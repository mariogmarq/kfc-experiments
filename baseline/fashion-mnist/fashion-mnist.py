import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from flex.data import Dataset, FedDataDistribution, FedDataset, FedDatasetConfig
from flex.datasets import load
from flex.model import FlexModel
from flex.pool import (FlexPool, fed_avg, init_server_model)
from attacks.utils import *
from flexBlock.pool import (BlockchainPool, PoFLBlockchainPool,
                            PoWBlockchainPool)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

CLIENTS_PER_ROUND = 30
EPOCHS = 10
STOPPING_VAL_DELTA = 0.005
N_MINERS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset():
    train_data = FashionMNIST(
        root=".",
        train=True,
        download=True,
        transform=None
    )

    test_data = FashionMNIST(
        root=".",
        train=False,
        download=True,
        transform=None
    )
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    data_threshold = 30
    # Get users with more than 30 items
    print("All users", len(flex_dataset))
    cids = list(flex_dataset.keys())
    for k in cids:
        if len(flex_dataset[k]) < data_threshold:
            del flex_dataset[k]

    print("Filtered users", len(flex_dataset))

    assert isinstance(flex_dataset, FedDataset)

    return flex_dataset, test_data

flex_dataset, test_data = get_dataset()

mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

@dataclass
class PoFLMetric:
    aggregated: bool
    target_acc: float

@dataclass
class Metrics:
    loss: float
    accuracy: float
    n_round: int
    pofl: Optional[PoFLMetric] = None

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(14*14*64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        return self.fc(x)

@init_server_model
def build_server_model():
    server_flex_model = FlexModel()
    server_flex_model["model"] = CNNModel()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model

def train(client_flex_model: FlexModel, client_data: Dataset):
    train_dataset = client_data.to_torchvision_dataset(transform=mnist_transforms)
    client_dataloader = DataLoader(train_dataset, batch_size=20)
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(EPOCHS):
        for imgs, labels in client_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

def obtain_accuracy(server_flex_model: FlexModel, test_data: Dataset):
    model = server_flex_model["model"]
    model.eval()
    test_acc = 0
    total_count = 0
    model = model.to(device)
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset(transform=mnist_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, pin_memory=False
    )
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_acc /= total_count
    return test_acc

def obtain_metrics(server_flex_model: FlexModel, data):
    if data is None:
        data = test_data
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataset = data.to_torchvision_dataset(transform=mnist_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc

def clean_up_models(client_model: FlexModel, _):
    import gc

    client_model.clear()
    gc.collect()

def train_pofl(pool: BlockchainPool, target_acc: float, n_rounds = 20):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS*3, delta=0.01)

    i = 0
    target_acc = target_acc

    for i in tqdm(range(n_rounds)):
        selected_clients = pool.clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients_block, selected_clients)
        selected_clients.map(train)
        pool.aggregators.map(get_clients_weights_block, selected_clients)
        aggregated = pool.aggregate(fed_avg, set_agreggated_weights_to_server, eval_function=obtain_accuracy, eval_dataset=test_data, accuracy=target_acc)

        selected_clients.map(clean_up_models)

        if aggregated:
            a = max(list(map(lambda x: x[1], pool.servers.map(obtain_metrics))))
            target_acc = a + (1 - a)*0.1
        
        round_metrics = pool.servers.map(obtain_metrics)
        
        print(f"Aggregated? {'yes' if aggregated else 'no':3} target_acc: {target_acc}")
        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i, PoFLMetric(aggregated, target_acc)))
            stopper(loss)
        
        if False:
            print("Early stopping at {i}")
            break
    
    return metrics
        


def train_pos_pow(pool: BlockchainPool, n_rounds = 20):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS*3, delta=0.01)

    for i in tqdm(range(n_rounds), "POS/POW"):
        selected_clients = pool.clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients_block, selected_clients)
        selected_clients.map(train)
        pool.aggregators.map(get_clients_weights_block, selected_clients)
        pool.aggregate(fed_avg, set_agreggated_weights_to_server)

        selected_clients.map(clean_up_models)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if False:
            print("Early stopping at {i}")
            break
    
    return metrics


def train_base(pool: FlexPool, n_rounds = 20):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(5, delta=0.01)

    for i in tqdm(range(n_rounds), "BASE"):
        selected_clients = pool.clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients, selected_clients)
        selected_clients.map(train)
        pool.aggregators.map(get_clients_weights, selected_clients)
        pool.aggregators.map(fed_avg)
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)


        selected_clients.map(clean_up_models)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if False:
            print("Early stopping at {i}")
            break
    
    return metrics

def dump_metric(file_name: str, metrics: List[Metrics]):
    with open(file_name, "w") as f:
        json.dump(list(map(lambda x: asdict(x), metrics)), f)

def run_server_pool():
    global flex_dataset
    global test_data
    flex_dataset["server"] = test_data
    for i in range(3):
        print(f"[BASE] Experiment round {i}")
        pool = FlexPool.client_server_pool(flex_dataset, build_server_model)
        metrics = train_base(pool)
        dump_metric(f"base-{i}.json", metrics)

def run_pow():
    for i in range(3):
        print(f"[POW] Experiment round {i}")
        pool = PoWBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics = train_pos_pow(pool)
        dump_metric(f"pow-{i}.json", metrics)

def run_pofl():
    for i in range(3):
        print(f"[POFL] Experiment round {i}")
        pool = PoFLBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics = train_pofl(pool, target_acc=0.4)
        dump_metric(f"pofl-{i}.json", metrics)

def main():
    run_pofl()
    run_pow()
    run_server_pool()
        
if __name__ == "__main__":
    main()
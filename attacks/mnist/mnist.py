from typing import List

import torch
import torch.nn as nn
from flex.data import Dataset, FedDataset, FedDatasetConfig, FedDataDistribution
from flex.datasets import load
from flex.model import FlexModel
from flex.pool import (FlexPool, collect_clients_weights, fed_avg, init_server_model)
from attacks.utils import *
from flexBlock.pool import (BlockchainPool, PoFLBlockchainPool,
                            PoWBlockchainPool,
                            collect_to_send_wrapper)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

CLIENTS_PER_ROUND = 30
EPOCHS = 10
N_MINERS = 3
NUM_POISONED = 100
POISONED_PER_ROUND = 1
SANE_PER_ROUND = CLIENTS_PER_ROUND - POISONED_PER_ROUND
DEFAULT_BOOSTING = float(CLIENTS_PER_ROUND) / float(POISONED_PER_ROUND)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset():
    flex_dataset, test_data = load("emnist")
    assert isinstance(flex_dataset, Dataset)

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

assert isinstance(flex_dataset, FedDataset)

poisoned_clients_ids = list(flex_dataset.keys())[:NUM_POISONED]
print(f"From a total of {len(flex_dataset.keys())} there is {NUM_POISONED} poisoned clients")

flex_dataset = flex_dataset.apply(label_flipping, node_ids=poisoned_clients_ids)

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


@collect_clients_weights
def get_poisoned_weights(client_flex_model: FlexModel, boosting=None):
    boosting_coef = boosting[client_flex_model.actor_id] if boosting is not None else DEFAULT_BOOSTING
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return apply_boosting([weight_dict[name] - server_dict[name].to(dev) for name in weight_dict], boosting_coef)

get_poisoned_weights_block = collect_to_send_wrapper(get_poisoned_weights)

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

    poisoned_clients = pool.clients.select(lambda client_id, _: client_id in poisoned_clients_ids)
    clean_clients = pool.clients.select(lambda client_id, _: client_id not in poisoned_clients_ids)       

    i = 0
    target_acc = target_acc

    for i in tqdm(range(n_rounds)):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = poisoned_clients.select(POISONED_PER_ROUND)

        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        pool.servers.map(copy_server_model_to_clients_block, selected_poisoned)

        selected_clean.map(train)
        selected_poisoned.map(train)

        boosting = get_boosting_coef(pool, selected_poisoned, selected_clean)

        pool.aggregators.map(get_clients_weights_block, selected_clean)
        pool.aggregators.map(get_poisoned_weights_block, selected_poisoned, boosting=boosting)

        aggregated = pool.aggregate(fed_avg, set_agreggated_weights_to_server, eval_function=obtain_accuracy, eval_dataset=test_data, accuracy=target_acc)

        selected_clean.map(clean_up_models)
        selected_poisoned.map(clean_up_models)

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
            print(f"Early stopping at {i}")
            break
    
    return metrics
        


def train_pos_pow(pool: BlockchainPool, n_rounds = 20):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS*3, delta=0.01)

    poisoned_clients = pool.clients.select(lambda client_id, _: client_id in poisoned_clients_ids)
    clean_clients = pool.clients.select(lambda client_id, _: client_id not in poisoned_clients_ids)       

    for i in tqdm(range(n_rounds), "POS/POW"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = poisoned_clients.select(POISONED_PER_ROUND)

        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        pool.servers.map(copy_server_model_to_clients_block, selected_poisoned)

        selected_clean.map(train)
        selected_poisoned.map(train)

        pool.aggregators.map(get_clients_weights_block, selected_clean)
        pool.aggregators.map(get_poisoned_weights_block, selected_poisoned)

        pool.aggregate(fed_avg, set_agreggated_weights_to_server)

        selected_clean.map(clean_up_models)
        selected_poisoned.map(clean_up_models)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if False:
            print(f"Early stopping at {i}")
            break
    
    return metrics


def train_base(pool: FlexPool, n_rounds = 20):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(5, delta=0.01)

    poisoned_clients = pool.clients.select(lambda client_id, _: client_id in poisoned_clients_ids)
    clean_clients = pool.clients.select(lambda client_id, _: client_id not in poisoned_clients_ids)       

    for i in tqdm(range(n_rounds), "BASE"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = poisoned_clients.select(POISONED_PER_ROUND)

        pool.servers.map(copy_server_model_to_clients, selected_clean)
        pool.servers.map(copy_server_model_to_clients, selected_poisoned)

        selected_clean.map(train)
        selected_poisoned.map(train)

        pool.aggregators.map(get_clients_weights, selected_clean)
        pool.aggregators.map(get_poisoned_weights, selected_poisoned)

        pool.aggregators.map(fed_avg)
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)


        selected_clean.map(clean_up_models)
        selected_poisoned.map(clean_up_models)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if False:
            print(f"Early stopping at {i}")
            break
    
    return metrics

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
    run_pow()
    run_pofl()
    run_server_pool()
        
if __name__ == "__main__":
    main()
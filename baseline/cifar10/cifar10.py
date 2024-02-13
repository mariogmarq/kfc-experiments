import copy
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
from torchvision import datasets
from flex.data import Dataset, FedDatasetConfig, FedDataDistribution
from flex.model import FlexModel
from flex.pool import (FlexPool, collect_clients_weights, deploy_server_model,
                       fed_avg, init_server_model, set_aggregated_weights)
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from tqdm import tqdm

from flexBlock.pool import (BlockchainPool, PoFLBlockchainPool,
                            PoWBlockchainPool,
                            collect_to_send_wrapper, deploy_server_to_miner)
from attacks.utils import EarlyStopping

CLIENTS_PER_ROUND = 30
EPOCHS = 5
N_MINERS = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.CIFAR10(
    root=".",
    train=True,
    download=True,
    transform=None,  # Note that we do not specify transforms here, we provide them later in the training process
)

test_data = datasets.CIFAR10(
    root=".",
    train=False,
    download=True,
    transform=None
)

test_data = Dataset.from_torchvision_dataset(test_data)

config = FedDatasetConfig(seed=0)
config.replacement = False
config.n_nodes = 100

flex_dataset = FedDataDistribution.from_config(
    centralized_data=Dataset.from_torchvision_dataset(train_data), config=config
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


cifar_transforms = EfficientNet_B4_Weights.DEFAULT.transforms()

def get_model(num_classes=10):
    efficient_model = efficientnet_b4(weights='DEFAULT')
    for p in efficient_model.parameters():
        p.requires_grad = False
    efficient_model.classifier[1] = torch.nn.Linear(efficient_model.classifier[1].in_features, num_classes)
    return efficient_model

@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    server_flex_model["model"] = get_model()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.functional.cross_entropy
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model

@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    new_flex_model = FlexModel()
    new_flex_model["model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    new_flex_model["optimizer_func"] = copy.deepcopy(server_flex_model["optimizer_func"])
    new_flex_model["optimizer_kwargs"] = copy.deepcopy(server_flex_model["optimizer_kwargs"])
    return new_flex_model

copy_server_model_to_clients_block = deploy_server_to_miner(copy_server_model_to_clients)

def train(client_flex_model: FlexModel, client_data: Dataset):
    train_dataset = client_data.to_torchvision_dataset(transform=cifar_transforms)
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
def get_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].classifier[1].state_dict()
    return [weight_dict[name] for name in weight_dict]

get_clients_weights_block = collect_to_send_wrapper(get_clients_weights)

def obtain_accuracy(server_flex_model: FlexModel, test_data: Dataset):
    model = server_flex_model["model"]
    model.eval()
    test_acc = 0
    total_count = 0
    model = model.to(device)
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset(transform=cifar_transforms)
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
    test_dataset = data.to_torchvision_dataset(transform=cifar_transforms)
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

@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    with torch.no_grad():
        weight_dict = server_flex_model["model"].classifier[1].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(new)


def clean_up_models(clients: FlexPool):
    import gc
    clients.clients.map(lambda model, _: model.clear())
    gc.collect()
    torch.cuda.empty_cache()

def train_pofl(pool: BlockchainPool, target_acc: float, n_rounds = 100):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS*5)

    i = 0
    target_acc = target_acc


    for i in tqdm(range(n_rounds)):
        selected_clients = pool.clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients_block, selected_clients)
        selected_clients.map(train)
        pool.aggregators.map(get_clients_weights_block, selected_clients)
        
        aggregated = pool.aggregate(fed_avg, set_agreggated_weights_to_server, eval_function=obtain_accuracy, eval_dataset=test_data, accuracy=target_acc)

        clean_up_models(selected_clients)

        if aggregated:
            a = max(list(map(lambda x: x[1], pool.servers.map(obtain_metrics))))
            target_acc = a + 0.05
        
        round_metrics = pool.servers.map(obtain_metrics)
        
        print(f"Aggregated? {'yes' if aggregated else 'no':3} target_acc: {target_acc}")
        
        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i, PoFLMetric(aggregated, target_acc)))
            stopper(loss)
        
        if stopper.early_stop:
            print("Early stopping at {i}")
            break

    
    return metrics
        


def train_pos_pow(pool: BlockchainPool, n_rounds=100):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS*5)

    for i in tqdm(range(n_rounds), "POS/POW"):
        selected_clients = pool.clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients_block, selected_clients)
        selected_clients.map(train)
        pool.aggregators.map(get_clients_weights_block, selected_clients)
        pool.aggregate(fed_avg, set_agreggated_weights_to_server)

        clean_up_models(selected_clients)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if stopper.early_stop:
            print("Early stopping at {i}")
            break
    
    return metrics


def train_base(pool: FlexPool, n_rounds = 100):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(5)

    for i in tqdm(range(n_rounds), "BASE"):
        selected_clients = pool.clients.select(CLIENTS_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients, selected_clients)
        selected_clients.map(train)
        pool.aggregators.map(get_clients_weights, selected_clients)
        pool.aggregators.map(fed_avg)
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)

        clean_up_models(selected_clients)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if stopper.early_stop:
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
        metrics = train_pofl(pool, target_acc=0.3)
        dump_metric(f"pofl-{i}.json", metrics)

def main():
    #run_pofl()
    #run_pow()
    run_server_pool()
        
if __name__ == "__main__":
    main()
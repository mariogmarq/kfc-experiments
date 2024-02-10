import torch
from torchvision import transforms
from flex.data import Dataset, FedDatasetConfig, FedDataDistribution
from flex.model import FlexModel
from flex.pool import FlexPool, collect_clients_weights, fed_avg, init_server_model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.models import resnet18
from tqdm import tqdm

from flexBlock.pool import (BlockchainPool, PoFLBlockchainPool,
                            PoWBlockchainPool,
                            collect_to_send_wrapper, deploy_server_to_miner)
from attacks.utils import *

CLIENTS_PER_ROUND = 30
NUM_POISONED = 10
EPOCHS = 5
N_MINERS = 3
BATCH_SIZE = 14
POISONED_PER_ROUND = 1 # With model replacement
DEFAULT_BOOSTING = float(CLIENTS_PER_ROUND) / float(POISONED_PER_ROUND)

SANE_PER_ROUND = CLIENTS_PER_ROUND - POISONED_PER_ROUND
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} as device")

celeba_transforms = transforms.Compose([])

initial_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((224,224)),
    ]
)

class FormatLabel:
    def __call__(self, data):
        return (data[0].item(), data[1])


@data_poisoner
def label_flipping(img_array, label):
    return img_array, label[::-1]


def load_celeba():
    # Load and federate data
    out_dir = "."
    preliminary_transforms = initial_transforms
    # Load and federate data
    train_celeba = CelebA(
        root=out_dir,
        split="train",
        download=True,
        transform=preliminary_transforms,
        target_transform=FormatLabel(),
        target_type=["identity", "attr"],
    )

    test_celeba = CelebA(
        root=out_dir,
        split="test",
        download=True,
        transform=preliminary_transforms,
        target_transform=None,
        target_type="attr",
    )

    print("Creating from torch vision dataset")
    celeba_train_dataset = Dataset.from_torchvision_dataset(train_celeba)
    celeba_test_dataset = Dataset.from_torchvision_dataset(test_celeba)

    config = FedDatasetConfig(seed=0, group_by_label_index=0)

    print("Federating...")
    fed_celeba = FedDataDistribution.from_config(
                    centralized_data=celeba_train_dataset, #####
                    config=config
                )


    data_threshold = 30
    # Get users with more than 30 items
    print("All users", len(fed_celeba))
    cids = list(fed_celeba.keys())
    for k in cids:
        if len(fed_celeba[k]) < data_threshold:
            del fed_celeba[k]

    smiling_index = -9

    def select_label(dataset: Dataset):
        y_data = [np.eye(2)[label[smiling_index]] for label in dataset.y_data]
        return Dataset(X_data=dataset.X_data, y_data=y_data)

    print("Filtering labels")
    fed_celeba = fed_celeba.apply(select_label)
    celeba_test_dataset = select_label(celeba_test_dataset)

    print("Poisoning...")
    poisoned_clients_ids = list(fed_celeba.keys())[:NUM_POISONED]
    fed_celeba = fed_celeba.apply(label_flipping, node_ids=poisoned_clients_ids)


    print("Done!")
    return fed_celeba, celeba_test_dataset, poisoned_clients_ids


def get_model(num_classes=2):
    resnet_model = resnet18(weights="DEFAULT")
    for p in resnet_model.parameters():
        p.requires_grad = False
    resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
    return resnet_model


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    server_flex_model["model"] = get_model()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.functional.binary_cross_entropy_with_logits
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model

copy_server_model_to_clients_block = deploy_server_to_miner(copy_server_model_to_clients)

def train(client_flex_model: FlexModel, client_data: Dataset):
    train_dataset = client_data.to_torchvision_dataset(transform=celeba_transforms)
    client_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
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
    weight_dict = client_flex_model["model"].fc.state_dict()
    server_dict = client_flex_model["server_model"].fc.state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return [weight_dict[name] - server_dict[name].to(dev) for name in weight_dict]

get_clients_weights_block = collect_to_send_wrapper(get_clients_weights)

@collect_clients_weights
def get_poisoned_weights(client_flex_model: FlexModel, boosting=None):
    boosting_coef = boosting[client_flex_model.actor_id] if boosting is not None else DEFAULT_BOOSTING
    weight_dict = client_flex_model["model"].fc.state_dict()
    server_dict = client_flex_model["server_model"].fc.state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return apply_boosting([weight_dict[name] - server_dict[name].to(dev) for name in weight_dict], boosting_coef)

get_poisoned_weights_block = collect_to_send_wrapper(get_poisoned_weights)

@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    dev = aggregated_weights[0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    with torch.no_grad():
        weight_dict = server_flex_model["model"].fc.state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(dev) + new)

def obtain_metrics(server_flex_model: FlexModel, data: Dataset):
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
    test_dataset = data.to_torchvision_dataset(transform=transforms.Compose([
    ]))
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            total_count += len(target)
            test_acc += (pred.argmax(1) == target.argmax(1)).sum().cpu().item()
            losses.append(criterion(pred, target).item())

    test_loss = np.mean(losses)
    test_acc /= total_count
    return test_loss, test_acc

def obtain_accuracy(server_flex_model: FlexModel, test_data: Dataset):
    return obtain_metrics(server_flex_model, test_data)[1]

def clean_up_models(clients: FlexPool):
    import gc
    clients.clients.map(lambda model, _: model.clear())
    gc.collect()
    torch.cuda.empty_cache()


def train_pofl(pool: BlockchainPool, target_acc: float, n_rounds = 100):
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

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

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
            print(f"Early stopping at {i}")
            break
    
    return metrics
        


def train_pos_pow(pool: BlockchainPool, n_rounds=100):
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

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if stopper.early_stop:
            print(f"Early stopping at {i}")
            break
    
    return metrics


def train_base(pool: FlexPool, n_rounds = 100):
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


        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_metrics)

        for (loss, acc) in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)
        
        if stopper.early_stop:
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
        metrics = train_pofl(pool, target_acc=0.3)
        dump_metric(f"pofl-{i}.json", metrics)

def main():
    run_pofl()
    run_pow()
    run_server_pool()
        
if __name__ == "__main__":
    flex_dataset, test_data, poisoned_clients_ids = load_celeba()
    main()

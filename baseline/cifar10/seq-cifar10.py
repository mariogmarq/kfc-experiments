import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
from utils import EarlyStopping
from torchvision import transforms, datasets
from torchvision.models import resnet18
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
cifar_transforms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_data = datasets.CIFAR10(
    root=".",
    train=True,
    download=True,
    transform=cifar_transforms,  # Note that we do not specify transforms here, we provide them later in the training process
)

test_data = datasets.CIFAR10(
    root=".",
    train=False,
    download=True,
    transform=cifar_transforms
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


def get_model(num_classes=10):
    resnet_model = resnet18(weights="DEFAULT")
    for p in resnet_model.parameters():
        p.requires_grad = False
    resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
    return resnet_model

trainloader = torch.utils.data.DataLoader(train_data, batch_size=20,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(test_data, batch_size=20,
                                         shuffle=False, num_workers=2)

def dump_metric(file_name: str, metrics: List[Metrics]):
    with open(file_name, "w") as f:
        json.dump(list(map(lambda x: asdict(x), metrics)), f)

if __name__ == "__main__":
    metrics = []
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    stopper = EarlyStopping(5)

    for i in tqdm(range(100)):
        model.train()
        model.to(device)
        for _ in range(5):
            for imgs, labels in trainloader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                pred = model(imgs)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
        
        model.eval()
        test_loss = 0
        test_acc = 0
        total_count = 0
        model = model.to(device)
        # get test data as a torchvision object
        losses = []
        with torch.no_grad():
            for data, target in testloader:
                total_count += target.size(0)
                data, target = data.to(device), target.to(device)
                output = model(data)
                losses.append(criterion(output, target).item())
                pred = output.data.max(1, keepdim=True)[1]
                test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

        test_loss = sum(losses) / len(losses)
        test_acc /= total_count
        stopper(test_loss)

        if stopper.early_stop:
            print(f"end at {i}")
            break

        metrics.append(Metrics(test_loss, test_acc, i))
    
    dump_metric("seq.json", metrics)
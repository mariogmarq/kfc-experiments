import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flex.datasets import load
from attacks.utils import EarlyStopping
from torchvision import transforms
from tqdm import tqdm

CLIENTS_PER_ROUND = 30
EPOCHS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
flex_dataset, test_data = load("federated_emnist", return_test=True)
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

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=mnist_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=mnist_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=2)

def dump_metric(file_name: str, metrics: List[Metrics]):
    with open(file_name, "w") as f:
        json.dump(list(map(lambda x: asdict(x), metrics)), f)

if __name__ == "__main__":
    metrics = []
    model = CNNModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    stopper = EarlyStopping(5, 0.01)

    for i in tqdm(range(100)):
        model.train()
        model.to(device)
        for _ in range(EPOCHS):
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
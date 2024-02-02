import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import train_tfm, test_tfm, TestingDataset

myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


class Resnet(nn.Module):
    def __init__(self, num_classes=200):
        super(Resnet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    train_set = ImageFolder("../../data/train", train_tfm)
    test_set = TestingDataset("../../data/test", test_tfm)

    print(f"Total training data: {train_set.__len__()}")
    print(f"Total testing data: {test_set.__len__()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    epochs = 100

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = Resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_accs = []

        with tqdm(total=len(train_loader), unit="batch") as tqdm_bar:
            tqdm_bar.set_description(f"Epoch {epoch+1:03d}/{epochs}")
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                logits = model(images)

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                acc = (logits.argmax(dim=1) == labels).float().mean()
                train_accs.append(acc)
                tqdm_bar.set_postfix(
                    loss=f"{sum(train_loss)/len(train_loss):.5f}",
                    acc=f"{sum(train_accs)/len(train_accs):.5f}",
                )
                tqdm_bar.update(1)

            scheduler.step()

            tqdm_bar.set_postfix(
                loss=f"{sum(train_loss)/len(train_loss):.5f}",
                acc=f"{sum(train_accs)/len(train_accs):.5f}",
            )
            tqdm_bar.close()

            if sum(train_accs) / len(train_accs) > best_acc:
                best_acc = sum(train_accs) / len(train_accs)
                torch.save(model.state_dict(), f"../weights/resnet50.pth")

    model = Resnet().to(device)
    model.load_state_dict(torch.load("../weights/resnet50.pth"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader)):
            images = images.to(device)
            logits = model(images)
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    class_dic = {}
    for i, class_dir in enumerate(train_set.classes):
        class_dic[i] = os.path.basename(class_dir)

    predictions = [class_dic[pred] for pred in predictions]

    submission = pd.DataFrame({"id": test_set.__getnames__(), "label": predictions})
    submission.to_csv("../results/submission_resnet50.csv", index=False)

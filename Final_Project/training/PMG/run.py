import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.network import PMG
from models.resnet import resnet50, resnet101
from inference import inference
from dataset import TestingDataset
from utils import train_tfm, test_tfm, jigsaw_generator, cosine_anneal_schedule


# seed
myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


# train
def train(cfg, train_loader, valid_loader, model, criterion, optimizer, epochs=10):
    best_acc = 0.0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_accs = []

        with tqdm(total=len(train_loader), unit="batch") as tqdm_bar:
            tqdm_bar.set_description(f"Epoch {epoch+1:03d}/{epochs}")
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                images, labels = Variable(images), Variable(labels)

                # update learning rate
                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]["lr"] = cosine_anneal_schedule(
                        epoch, epochs, lr[nlr]
                    )

                # step 1
                optimizer.zero_grad()
                inputs1 = jigsaw_generator(images, 8).to(device)
                output1, _, _, _ = model(inputs1)
                loss1 = criterion(output1, labels)
                loss1.backward()
                optimizer.step()

                # step 2
                optimizer.zero_grad()
                inputs2 = jigsaw_generator(images, 4).to(device)
                _, output2, _, _ = model(inputs2)
                loss2 = criterion(output2, labels)
                loss2.backward()
                optimizer.step()

                # step 3
                optimizer.zero_grad()
                inputs3 = jigsaw_generator(images, 2).to(device)
                _, _, output3, _ = model(inputs3)
                loss3 = criterion(output3, labels)
                loss3.backward()
                optimizer.step()

                # step 4
                optimizer.zero_grad()
                _, _, _, output_concat = model(images)
                concat_loss = criterion(output_concat, labels) * 2
                concat_loss.backward()
                optimizer.step()

                loss = loss1 + loss2 + loss3 + concat_loss

                train_loss.append(loss.item())
                acc = (output_concat.argmax(dim=1) == labels).float().mean()
                train_accs.append(acc)
                tqdm_bar.set_postfix(
                    loss=f"{sum(train_loss)/len(train_loss):.5f}",
                    acc=f"{sum(train_accs)/len(train_accs):.5f}",
                    val_loss=0.0,
                    val_acc=0.0,
                )
                tqdm_bar.update(1)

            tqdm_bar.set_postfix(
                loss=f"{sum(train_loss)/len(train_loss):.5f}",
                acc=f"{sum(train_accs)/len(train_accs):.5f}",
                val_loss=0.0,
                val_acc=0.0,
            )

            model.eval()
            valid_loss = []
            valid_accs = []

            for _, (images, labels) in enumerate(valid_loader):
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    images, labels = Variable(images), Variable(labels)
                    output1, output2, output3, output_concat = model(images)
                    output_combine = output1 + output2 + output3 + output_concat

                    loss = criterion(output_combine, labels)

                valid_loss.append(loss.item())
                acc = (output_combine.argmax(dim=1) == labels).float().mean()
                valid_accs.append(acc)

            tqdm_bar.set_postfix(
                loss=f"{sum(train_loss)/len(train_loss):.5f}",
                acc=f"{sum(train_accs)/len(train_accs):.5f}",
                val_loss=f"{sum(valid_loss)/len(valid_loss):.5f}",
                val_acc=f"{sum(valid_accs)/len(valid_accs):.5f}",
            )

            if sum(valid_accs) / len(valid_accs) > best_acc:
                best_acc = sum(valid_accs) / len(valid_accs)
                torch.save(model.state_dict(), cfg["model_weight_path"])

            tqdm_bar.close()


if __name__ == "__main__":
    # config
    with open("config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # device
    device = "cuda"
    torch.cuda.set_device(cfg["GPU"])

    # dataset
    train_set = ImageFolder(cfg["train_dir"], train_tfm)
    test_set = TestingDataset(cfg["test_dir"], test_tfm)
    valid_set = ImageFolder(cfg["valid_dir"], test_tfm)

    print(f"Total training data: {train_set.__len__()}")
    print(f"Total testing data: {test_set.__len__()}")
    print(f"Total validation data: {valid_set.__len__()}")

    class_dic = {}
    for i, class_dir in enumerate(train_set.classes):
        class_dic[i] = os.path.basename(class_dir)

    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    if cfg["backbone"] == "resnet50":
        net = resnet50(pretrained=True)
    elif cfg["backbone"] == "resnet101":
        net = resnet101(pretrained=True)
    else:
        raise ValueError("backbone must be resnet50 or resnet101")

    model = PMG(net, 512, 200).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [
            {"params": model.classifier_concat.parameters(), "lr": 0.002},
            {"params": model.conv_block1.parameters(), "lr": 0.002},
            {"params": model.classifier1.parameters(), "lr": 0.002},
            {"params": model.conv_block2.parameters(), "lr": 0.002},
            {"params": model.classifier2.parameters(), "lr": 0.002},
            {"params": model.conv_block3.parameters(), "lr": 0.002},
            {"params": model.classifier3.parameters(), "lr": 0.002},
            {"params": model.features.parameters(), "lr": 0.0002},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )

    train(cfg, train_loader, valid_loader, model, criterion, optimizer, epochs=epochs)

    inference(cfg, net, test_set, test_loader, class_dic)

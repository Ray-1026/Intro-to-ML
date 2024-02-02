import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler

from models.network import VisionTransformer, CONFIGS
from utils.scheduler import WarmupCosineSchedule
from utils.data_utils import get_loader
from inference import inference

# seed
myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(myseed)
np.random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


def setup(cfg):
    # Prepare model
    config = CONFIGS[cfg["model"]]
    config.split = cfg["split"]
    config.slide_step = 12

    num_classes = 200

    model = VisionTransformer(
        config,
        448,
        zero_head=True,
        num_classes=num_classes,
        smoothing_value=0.0,
    )

    model.load_from(np.load(cfg["pretrained_dir"]))
    model.to("cuda")

    return model


def train(cfg):
    epochs = cfg["epochs"]
    train_loader, _, valid_loader = get_loader(cfg)

    model = setup(cfg)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=cfg["warmup_steps"], t_total=epochs
    )

    model.zero_grad()
    best_acc = 0.0
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_acc = []

        with tqdm(total=len(train_loader), unit="batch") as tqdm_bar:
            tqdm_bar.set_description(f"Epoch {epoch+1:03d}/{epochs}")
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                with autocast():
                    loss, output = model(images, labels, epoch, epochs)
                    # loss = loss.mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # loss, output = model(images, labels, epoch, epochs)
                # loss = loss.mean()

                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # optimizer.step()
                # optimizer.zero_grad()
                # scheduler.step()

                train_loss.append(loss.item())
                acc = (output.argmax(dim=1) == labels).float().mean()
                train_acc.append(acc)
                tqdm_bar.set_postfix(
                    loss=f"{sum(train_loss)/len(train_loss):.5f}",
                    acc=f"{sum(train_acc)/len(train_acc):.5f}",
                    val_loss=0.0,
                    val_acc=0.0,
                )
                tqdm_bar.update(1)

            tqdm_bar.set_postfix(
                loss=f"{sum(train_loss)/len(train_loss):.5f}",
                acc=f"{sum(train_acc)/len(train_acc):.5f}",
                val_loss=0.0,
                val_acc=0.0,
            )

            train_losses.append(sum(train_loss) / len(train_loss))
            train_accs.append((sum(train_acc) / len(train_acc)).cpu().numpy())

            model.eval()
            criterion = nn.CrossEntropyLoss()
            valid_loss = []
            valid_acc = []

            for _, (images, labels) in enumerate(valid_loader):
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    output = model(images)
                    loss = criterion(output, labels)

                valid_loss.append(loss.item())
                acc = (output.argmax(dim=1) == labels).float().mean()
                valid_acc.append(acc)

            tqdm_bar.set_postfix(
                loss=f"{sum(train_loss)/len(train_loss):.5f}",
                acc=f"{sum(train_acc)/len(train_acc):.5f}",
                val_loss=f"{sum(valid_loss)/len(valid_loss):.5f}",
                val_acc=f"{sum(valid_acc)/len(valid_acc):.5f}",
            )

            valid_losses.append(sum(valid_loss) / len(valid_loss))
            valid_accs.append((sum(valid_acc) / len(valid_acc)).cpu().numpy())

            if sum(valid_acc) / len(valid_acc) > best_acc:
                best_acc = sum(valid_acc) / len(valid_acc)
                torch.save(model.state_dict(), cfg["model_weight_path"])

            tqdm_bar.close()

    plot_acc_loss(train_losses, train_accs, valid_losses, valid_accs)


def plot_acc_loss(train_loss, train_accs, valid_loss, valid_accs):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.legend()
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(train_accs, label="train_accs")
    plt.plot(valid_accs, label="valid_accs")
    plt.legend()
    plt.savefig("SIM-Trans.png")


if __name__ == "__main__":
    # config
    with open("config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # device
    device = "cuda"
    torch.cuda.set_device(cfg["GPU"])

    train(cfg)
    inference(cfg)

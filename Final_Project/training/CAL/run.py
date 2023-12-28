import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.network import WSDAN_CAL, batch_augment
from dataset import TestingDataset
from utils import CenterLoss, adjust_learning, train_tfm, test_tfm
from inference import inference

import numpy as np

# seed
myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


# train
def train(cfg, train_loader, valid_loader, model, criterion, optimizer, epochs=10):
    best_acc = 0.0
    feature_center = torch.zeros(200, 32 * model.num_features).cuda()

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_accs = []

        with tqdm(total=len(train_loader), unit="batch") as tqdm_bar:
            tqdm_bar.set_description(f"Epoch {epoch+1:03d}/{epochs}")
            batch_len = len(train_loader)
            for i, (images, labels) in enumerate(train_loader):
                float_iter = float(i) / batch_len
                adjust_learning(optimizer, epoch, float_iter)
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                y_pred_raw, y_pred_aux, feature_matrix, attention_map = model(images)

                feature_center_batch = F.normalize(feature_center[labels], dim=-1)
                feature_center[labels] += 0.05 * (
                    feature_matrix.detach() - feature_center_batch
                )

                with torch.no_grad():
                    crop_images = batch_augment(
                        images,
                        attention_map[:, :1, :, :],
                        mode="crop",
                        theta=(0.4, 0.6),
                        padding_ratio=0.1,
                    )
                    drop_images = batch_augment(
                        images,
                        attention_map[:, 1:, :, :],
                        mode="drop",
                        theta=(0.2, 0.5),
                    )
                aug_images = torch.cat([crop_images, drop_images], dim=0)
                y_aug = torch.cat([labels, labels], dim=0)

                y_pred_aug, y_pred_aux_aug, _, _ = model(aug_images)

                y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
                y_aux = torch.cat([labels, y_aug], dim=0)

                batch_loss = (
                    criterion(y_pred_raw, labels) / 3.0
                    + criterion(y_pred_aux, y_aux) * 3.0 / 3.0
                    + criterion(y_pred_aug, y_aug) * 2.0 / 3.0
                    + center_loss(feature_matrix, feature_center_batch)
                )

                batch_loss.backward()
                optimizer.step()

                train_loss.append(batch_loss.item())
                acc = (y_pred_raw.argmax(dim=1) == labels).float().mean()
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
                    y_pred_raw, y_pred_aux, _, attention_map = model(images)

                    crop_images3 = batch_augment(
                        images,
                        attention_map,
                        mode="crop",
                        theta=0.1,
                        padding_ratio=0.05,
                    )
                    y_pred_crop3, y_pred_aux_crop3, _, _ = model(crop_images3)

                    y_pred = (y_pred_raw + y_pred_crop3) / 2.0
                    y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.0

                    batch_loss = criterion(y_pred, labels)

                    valid_loss.append(batch_loss.item())
                    acc = (y_pred.argmax(dim=1) == labels).float().mean()
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

    model = WSDAN_CAL(num_classes=200, M=32, net=cfg["backbone"], pretrained=True)
    model = model.to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    center_loss = CenterLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=1e-5
    )

    train(cfg, train_loader, valid_loader, model, cross_entropy_loss, optimizer, epochs)
    inference(cfg, test_set, test_loader, class_dic)

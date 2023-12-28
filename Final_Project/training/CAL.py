import os
import glob
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.utils.model_zoo import load_url

# seed
myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# np.random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


# transforms
train_tfm = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_tfm = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# dataset
class TestingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        data
        ├── test
        |   ├── xxxxx.jpg
        |   ├── ...
        |   └── yyyyy.jpg
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.names = []

        self.images = sorted(glob.glob(f"{self.img_dir}/*"))
        self.names = [os.path.basename(image)[:-4] for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getnames__(self):
        return self.names

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        return image


# model
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        # spatial attention
        self.conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SPPLayer(nn.Module):
    def __init__(self, pool_size, pool=nn.MaxPool2d):
        super(SPPLayer, self).__init__()
        self.pool_size = pool_size
        self.pool = pool
        self.out_length = np.sum(np.array(self.pool_size) ** 2)

    def forward(self, x):
        B, C, H, W = x.size()
        for i in range(len(self.pool_size)):
            h_wid = int(math.ceil(H / self.pool_size[i]))
            w_wid = int(math.ceil(W / self.pool_size[i]))
            h_pad = (h_wid * self.pool_size[i] - H + 1) / 2
            w_pad = (w_wid * self.pool_size[i] - W + 1) / 2
            out = self.pool(
                (h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad)
            )(x)
            if i == 0:
                spp = out.view(B, -1)
            else:
                spp = torch.cat([spp, out.view(B, -1)], dim=1)
        return spp


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, cbam=None, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if cbam is not None:
            self.cbam = CBAMLayer(planes)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cbam=None, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if cbam is not None:
            self.cbam = CBAMLayer(planes * Bottleneck.expansion)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, cbam=None, num_classes=1000, stride=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], cbam, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cbam, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cbam, stride=stride)
        print("==> using resnet with stride=", 16 * stride)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.spp = SPPLayer(pool_size=[1, 2, 4], pool=nn.MaxPool2d)
        # self.fc = nn.Linear(512 * block.expansion * self.spp.out_length, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cbam=None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride=stride, cbam=cbam, downsample=downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cbam=cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.spp(x)
        x = self.fc(x)

        return x

    def get_features(self):
        return nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        if len(pretrained_dict) == len(state_dict):
            print("%s: All params loaded" % type(self).__name__)
        else:
            print("%s: Some params were not loaded:" % type(self).__name__)
            not_loaded_keys = [
                k for k in state_dict.keys() if k not in pretrained_dict.keys()
            ]
            print(("%s, " * (len(not_loaded_keys) - 1) + "%s") % tuple(not_loaded_keys))
        model_dict.update(pretrained_dict)
        super(ResNet, self).load_state_dict(model_dict)


def resnet101(pretrained=False, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet101"]))
    return model


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


EPSILON = 1e-6


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool="GAP"):
        super(BAP, self).__init__()
        assert pool in ["GAP", "GMP"]
        if pool == "GAP":
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (
                torch.einsum("imjk,injk->imn", (attentions, features)) / float(H * W)
            ).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i : i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(
            torch.abs(feature_matrix) + EPSILON
        )

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (
            torch.einsum("imjk,injk->imn", (fake_att, features)) / float(H * W)
        ).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON
        )

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


def batch_augment(images, attention_map, mode="crop", theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == "crop":
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index : batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(
                int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0
            )
            height_max = min(
                int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH
            )
            width_min = max(
                int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0
            )
            width_max = min(
                int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW
            )

            crop_images.append(
                F.upsample_bilinear(
                    images[
                        batch_index : batch_index + 1,
                        :,
                        height_min:height_max,
                        width_min:width_max,
                    ],
                    size=(imgH, imgW),
                )
            )
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == "drop":
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index : batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(
                F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d
            )
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError(
            "Expected mode in ['crop', 'drop'], but received unsupported augmentation method %s"
            % mode
        )


class WSDAN_CAL(nn.Module):
    def __init__(self, num_classes, M=32, net="inception_mixed_6e", pretrained=False):
        super(WSDAN_CAL, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        # if "inception" in net:
        #     if net == "inception_mixed_6e":
        #         self.features = inception_v3(
        #             pretrained=pretrained
        #         ).get_features_mixed_6e()
        #         self.num_features = 768
        #     elif net == "inception_mixed_7c":
        #         self.features = inception_v3(
        #             pretrained=pretrained
        #         ).get_features_mixed_7c()
        #         self.num_features = 2048
        #     else:
        #         raise ValueError("Unsupported net: %s" % net)
        # elif "resnet" in net:
        self.features = resnet101(pretrained=pretrained).get_features()
        self.num_features = 512 * self.features[-1][-1].expansion
        # elif "att" in net:
        #     print("==> Using MANet with resnet101 backbone")
        #     self.features = MANet()
        #     self.num_features = 2048
        # else:
        #     raise ValueError("Unsupported net: %s" % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool="GAP")

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        # logging.info(
        #     "WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}".format(
        #         net, self.num_classes, self.M
        #     )
        # )

    def visualize(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != "inception_mixed_7c":
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, : self.M, ...]

        feature_matrix, _ = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.0)

        return p, attention_maps

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != "inception_mixed_7c":
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, : self.M, ...]

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.0)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(
                    attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON
                )
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(
                attention_map
            )  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(
                attention_maps, dim=1, keepdim=True
            )  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.0), feature_matrix, attention_map


# utils
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction="sum")

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


def adjust_learning(optimizer, epoch, iter):
    """Decay the learning rate based on schedule"""
    base_lr = 0.001
    base_rate = 0.9
    base_duration = 2.0
    lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# train
def train(train_loader, valid_loader, model, criterion, optimizer, epochs=10):
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
                torch.save(model.state_dict(), f"CAL.pth")

            tqdm_bar.close()


# inference
def inference(test_loader, class_dic):
    model = WSDAN_CAL(num_classes=200, M=32, net="resnet101", pretrained=True).to(
        device
    )
    model.load_state_dict(torch.load("CAL.pth"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader)):
            images = images.to(device)
            images_m = torch.flip(images, dims=[3])
            y_pred_raw, y_pred_aux, _, attention_map = model(images)
            y_pred_raw_m, y_pred_aux_m, _, attention_map_m = model(images_m)

            crop_images = batch_augment(
                images, attention_map, mode="crop", theta=0.3, padding_ratio=0.1
            )
            y_pred_crop, y_pred_aux_crop, _, _ = model(crop_images)

            crop_images2 = batch_augment(
                images, attention_map, mode="crop", theta=0.2, padding_ratio=0.1
            )
            y_pred_crop2, y_pred_aux_crop2, _, _ = model(crop_images2)

            crop_images3 = batch_augment(
                images, attention_map, mode="crop", theta=0.1, padding_ratio=0.05
            )
            y_pred_crop3, y_pred_aux_crop3, _, _ = model(crop_images3)

            crop_images_m = batch_augment(
                images_m, attention_map_m, mode="crop", theta=0.3, padding_ratio=0.1
            )
            y_pred_crop_m, y_pred_aux_crop_m, _, _ = model(crop_images_m)

            crop_images2_m = batch_augment(
                images_m, attention_map_m, mode="crop", theta=0.2, padding_ratio=0.1
            )
            y_pred_crop2_m, y_pred_aux_crop2_m, _, _ = model(crop_images2_m)

            crop_images3_m = batch_augment(
                images_m, attention_map_m, mode="crop", theta=0.1, padding_ratio=0.05
            )
            y_pred_crop3_m, y_pred_aux_crop3_m, _, _ = model(crop_images3_m)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.0
            y_pred_m = (
                y_pred_raw_m + y_pred_crop_m + y_pred_crop2_m + y_pred_crop3_m
            ) / 4.0
            y_pred = (y_pred + y_pred_m) / 2.0

            y_pred_aux = (
                y_pred_aux + y_pred_aux_crop + y_pred_aux_crop2 + y_pred_aux_crop3
            ) / 4.0
            y_pred_aux_m = (
                y_pred_aux_m
                + y_pred_aux_crop_m
                + y_pred_aux_crop2_m
                + y_pred_aux_crop3_m
            ) / 4.0
            y_pred_aux = (y_pred_aux + y_pred_aux_m) / 2.0

            predictions.extend(y_pred_aux.argmax(dim=1).cpu().numpy())

    predictions = [class_dic[pred] for pred in predictions]

    submission = pd.DataFrame({"id": test_set.__getnames__(), "label": predictions})
    submission.to_csv("../submission_CAL.csv", index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)

    train_set = ImageFolder("../data/train", train_tfm)
    test_set = TestingDataset("../data/test", test_tfm)
    valid_set = ImageFolder("../data/validation", test_tfm)

    print(f"Total training data: {train_set.__len__()}")
    print(f"Total testing data: {test_set.__len__()}")
    print(f"Total validation data: {valid_set.__len__()}")

    batch_size = 12
    epochs = 20

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
    )

    cross_entropy_loss = nn.CrossEntropyLoss()
    center_loss = CenterLoss()

    model = WSDAN_CAL(num_classes=200, M=32, net="resnet101", pretrained=True).to(
        device
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
    )

    train(train_loader, valid_loader, model, cross_entropy_loss, optimizer, epochs)

    class_dic = {}
    for i, class_dir in enumerate(train_set.classes):
        class_dic[i] = os.path.basename(class_dir)

    inference(test_loader, class_dic)

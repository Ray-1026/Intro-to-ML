import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch.utils.model_zoo import load_url as load_state_dict_from_url

# seed
myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
}


# transforms
train_tfm = transforms.Compose(
    [
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_tfm = transforms.Compose(
    [
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.avgpool(x5)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x1, x2, x3, x4, x5


def _resnet(net, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[net],
            progress=progress,
        )
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


class PMG(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMG, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(
                self.num_ftrs // 4,
                feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                relu=True,
            ),
            BasicConv(
                feature_size,
                self.num_ftrs // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                relu=True,
            ),
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(
                self.num_ftrs // 2,
                feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                relu=True,
            ),
            BasicConv(
                feature_size,
                self.num_ftrs // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                relu=True,
            ),
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(
                self.num_ftrs,
                feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                relu=True,
            ),
            BasicConv(
                feature_size,
                self.num_ftrs // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                relu=True,
            ),
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)

        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)
        return xc1, xc2, xc3, x_concat


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# utils
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n**2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[
            ...,
            x * block_size : (x + 1) * block_size,
            y * block_size : (y + 1) * block_size,
        ].clone()
        jigsaws[
            ...,
            x * block_size : (x + 1) * block_size,
            y * block_size : (y + 1) * block_size,
        ] = temp

    return jigsaws


# train
def train(train_loader, valid_loader, model, criterion, optimizer, epochs=10):
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
                torch.save(model.state_dict(), f"PMG_101.pth")

            tqdm_bar.close()


# inference
def inference(net, test_loader, class_dic):
    model = PMG(net, 512, 200).to(device)
    model.load_state_dict(torch.load("PMG_101.pth"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader)):
            images = images.to(device)
            images = Variable(images)
            output1, output2, output3, output_concat = model(images)
            output_combine = output1 + output2 + output3 + output_concat

            predictions.extend(output_combine.argmax(dim=-1).cpu().numpy().tolist())

    predictions = [class_dic[pred] for pred in predictions]

    submission = pd.DataFrame({"id": test_set.__getnames__(), "label": predictions})
    submission.to_csv("../submission_PMG.csv", index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)

    train_set = ImageFolder("../data/train", train_tfm)
    test_set = TestingDataset("../data/test", test_tfm)
    valid_set = ImageFolder("../data/validation", test_tfm)

    print(f"Total training data: {train_set.__len__()}")
    print(f"Total testing data: {test_set.__len__()}")
    print(f"Total validation data: {valid_set.__len__()}")

    batch_size = 8
    epochs = 10

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # net = resnet50(pretrained=True)
    net = resnet101(pretrained=True)
    # net = resnext50_32x4d(pretrained=True)
    # net = resnext101_32x8d(pretrained=True)
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

    train(train_loader, valid_loader, model, criterion, optimizer, epochs=epochs)

    class_dic = {}
    for i, class_dir in enumerate(train_set.classes):
        class_dic[i] = os.path.basename(class_dir)

    inference(net, test_loader, class_dic)

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


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


def resnet34(pretrained=False, num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, num_classes=1000, stride=1):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stride=stride)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet101"]))
    return model


def resnet34_cbam(pretrained=False, num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], cbam=True, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet34"]))
    return model


def resnet50_cbam(pretrained=False, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], cbam=True, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet50"]))
    return model


def resnet101_cbam(pretrained=False, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], cbam=True, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet101"]))
    return model

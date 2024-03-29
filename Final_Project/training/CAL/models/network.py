import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import *


EPSILON = 1e-6


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


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
        if self.net == "resnet101":
            self.features = resnet101(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif self.net == "resnet50":
            self.features = resnet50(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif self.net == "resnet101_cbam":
            self.features = resnet101_cbam(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif self.net == "resnet50_cbam":
            self.features = resnet50_cbam(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif self.net == "resnet34":
            self.features = resnet34(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError("Unsupported Net: %s" % self.net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool="GAP")

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

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

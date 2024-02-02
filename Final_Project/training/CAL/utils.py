import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def add_noise(image, noise_level=0.1):
    noise = torch.rand_like(image) * noise_level
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image


# transform
train_tfm = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: add_noise(x, 0.1)),
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


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


class LabelSmoothing(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


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

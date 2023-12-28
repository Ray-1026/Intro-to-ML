import torch.nn as nn
import torchvision.transforms as transforms

# transform
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

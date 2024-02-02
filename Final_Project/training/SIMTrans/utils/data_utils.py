from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import ImageFolder

from .dataset import TestingDataset as CUB


def get_loader(cfg):
    train_transform = transforms.Compose(
        [
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # train
    if cfg["train_dir"] is not None:
        trainset = ImageFolder(cfg["train_dir"], transform=train_transform)
        train_sampler = RandomSampler(trainset)
        train_loader = DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=cfg["batch_size"],
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )
    else:
        train_loader = None

    # test
    testset = CUB(cfg["test_dir"], transform=test_transform)
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(
        testset,
        sampler=test_sampler,
        batch_size=cfg["batch_size"],
        num_workers=4,
        pin_memory=True,
    )

    # valid
    if cfg["valid_dir"] is not None:
        validset = ImageFolder(cfg["valid_dir"], transform=test_transform)
        valid_sampler = SequentialSampler(validset)
        valid_loader = DataLoader(
            validset,
            sampler=valid_sampler,
            batch_size=cfg["batch_size"],
            num_workers=4,
            pin_memory=True,
        )
    else:
        valid_loader = None

    return train_loader, test_loader, valid_loader

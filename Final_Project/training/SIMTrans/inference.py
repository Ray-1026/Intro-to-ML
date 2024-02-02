import pandas as pd
import os
import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from models.network import VisionTransformer, CONFIGS
from utils.data_utils import get_loader
from utils.dataset import TestingDataset


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

    # model.load_from(np.load(cfg["pretrained_dir"]))
    model.to("cuda")

    return model


def inference(cfg):
    device = "cuda"

    _, test_loader, _ = get_loader(cfg)
    train_set = ImageFolder(cfg["train_dir"])
    test_set = TestingDataset(cfg["test_dir"])

    class_dic = {}
    for i, class_dir in enumerate(train_set.classes):
        class_dic[i] = os.path.basename(class_dir)

    model = setup(cfg)
    model.load_state_dict(torch.load(cfg["model_weight_path"]))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader)):
            images = images.to(device)
            # output = model(images)

            with autocast():
                output = model(images)

            predictions.extend(output.argmax(dim=1).cpu().numpy().tolist())

    predictions = [class_dic[pred] for pred in predictions]

    submission = pd.DataFrame({"id": test_set.__getnames__(), "label": predictions})
    submission.to_csv(cfg["submission_path"], index=False)


if __name__ == "__main__":
    pass

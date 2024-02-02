import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

from CAL.models.network import WSDAN_CAL, batch_augment
from CAL.dataset import TestingDataset
from CAL.utils import test_tfm as test_tfm_cal
from SIMTrans.models.network import VisionTransformer, CONFIGS
from SIMTrans.utils.data_utils import get_loader

# seed
myseed = 666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(myseed)
np.random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cal = WSDAN_CAL(num_classes=200, M=32, net="resnet101", pretrained=True)
        self.sim_trans = self.setup()

    def setup(self):
        # Prepare model
        config = CONFIGS["ViT-B_16"]
        config.split = "overlap"
        config.slide_step = 12

        num_classes = 200

        model = VisionTransformer(
            config,
            448,
            zero_head=True,
            num_classes=num_classes,
            smoothing_value=0.0,
        )

        return model

    def forward(self, x_c, x_s):
        # SIM-Trans
        y_s = self.sim_trans(x_s)

        # CAL
        images_m = torch.flip(x_c, dims=[3])
        y_pred_raw, y_pred_aux, _, attention_map = self.cal(x_c)
        y_pred_raw_m, y_pred_aux_m, _, attention_map_m = self.cal(images_m)

        crop_images = batch_augment(
            x_c, attention_map, mode="crop", theta=0.3, padding_ratio=0.1
        )
        y_pred_crop, y_pred_aux_crop, _, _ = self.cal(crop_images)

        crop_images2 = batch_augment(
            x_c, attention_map, mode="crop", theta=0.2, padding_ratio=0.1
        )
        y_pred_crop2, y_pred_aux_crop2, _, _ = self.cal(crop_images2)

        crop_images3 = batch_augment(
            x_c, attention_map, mode="crop", theta=0.1, padding_ratio=0.05
        )
        y_pred_crop3, y_pred_aux_crop3, _, _ = self.cal(crop_images3)

        crop_images_m = batch_augment(
            images_m, attention_map_m, mode="crop", theta=0.3, padding_ratio=0.1
        )
        y_pred_crop_m, y_pred_aux_crop_m, _, _ = self.cal(crop_images_m)

        crop_images2_m = batch_augment(
            images_m, attention_map_m, mode="crop", theta=0.2, padding_ratio=0.1
        )
        y_pred_crop2_m, y_pred_aux_crop2_m, _, _ = self.cal(crop_images2_m)

        crop_images3_m = batch_augment(
            images_m, attention_map_m, mode="crop", theta=0.1, padding_ratio=0.05
        )
        y_pred_crop3_m, y_pred_aux_crop3_m, _, _ = self.cal(crop_images3_m)

        y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.0
        y_pred_m = (
            y_pred_raw_m + y_pred_crop_m + y_pred_crop2_m + y_pred_crop3_m
        ) / 4.0
        y_pred = (y_pred + y_pred_m) / 2.0

        # ensemble
        y_ensemble = F.softmax(y_pred, dim=1) + F.softmax(y_s, dim=1)

        return y_ensemble


def inference(test_set, test_loader_c, test_loader_t, class_dic):
    device = "cuda"

    ensemble = EnsembleModel().to(device)
    ensemble.load_state_dict(torch.load("weights/ensemble.pth"))
    ensemble.eval()

    predictions = []
    with torch.no_grad():
        for i, (images_c, images_t) in enumerate(
            tqdm(zip(test_loader_c, test_loader_t))
        ):
            images_cal = images_c.to(device)
            images_trans = images_t.to(device)

            y_ensemble = ensemble(images_cal, images_trans)

            predictions.extend(y_ensemble.argmax(dim=1).cpu().numpy().tolist())

    predictions = [class_dic[pred] for pred in predictions]

    submission = pd.DataFrame({"id": test_set.__getnames__(), "label": predictions})
    submission.to_csv("ensemble_SIMTrans_CAL.csv", index=False)


if __name__ == "__main__":
    # device
    device = "cuda"
    # torch.cuda.set_device(0)

    # load weight and save
    model = EnsembleModel().to(device)
    model.cal.load_state_dict(torch.load("weights/CAL.pth"))
    model.sim_trans.load_state_dict(torch.load("weights/SIM-Trans.pth"))
    torch.save(model.state_dict(), "weights/ensemble.pth")
    print("weights saved!")

    # dataset
    train_set = ImageFolder("../data/train")
    test_set_c = TestingDataset("../data/test", test_tfm_cal)

    print(f"Total testing data: {test_set_c.__len__()}")

    class_dic = {}
    for i, class_dir in enumerate(train_set.classes):
        class_dic[i] = os.path.basename(class_dir)

    batch_size = 8

    test_loader_c = DataLoader(test_set_c, batch_size=batch_size, shuffle=False)
    _, test_loader_t, _ = get_loader(
        {
            "batch_size": batch_size,
            "test_dir": "../data/test",
            "valid_dir": "../data/validation",
            "train_dir": "../data/train",
        }
    )

    inference(test_set_c, test_loader_c, test_loader_t, class_dic)

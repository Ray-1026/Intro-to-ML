import pandas as pd
import torch
from tqdm import tqdm
from models.network import WSDAN_CAL, batch_augment


def inference(cfg, test_set, test_loader, class_dic):
    device = "cuda"

    model = WSDAN_CAL(num_classes=200, M=32, net=cfg["backbone"], pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(cfg["model_weight_path"]))
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

            predictions.extend(y_pred.argmax(dim=1).cpu().numpy())

    predictions = [class_dic[pred] for pred in predictions]

    submission = pd.DataFrame({"id": test_set.__getnames__(), "label": predictions})
    submission.to_csv(cfg["submission_path"], index=False)


if __name__ == "__main__":
    pass

import pandas as pd
import torch
from tqdm import tqdm
from torch.autograd import Variable

from models.network import PMG


def inference(cfg, net, test_set, test_loader, class_dic):
    device = "cuda"

    model = PMG(net, 512, 200).to(device)
    model.load_state_dict(torch.load(cfg["model_weight_path"]))
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
    submission.to_csv(cfg["submission_path"], index=False)

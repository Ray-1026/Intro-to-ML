# import pandas as pd
# import os
# import numpy as np
# import cv2
# import sys

# # generate validation set
# if os.path.exists("../data/validation"):
#     print("Validation set already exists")
# else:
#     os.mkdir("../data/validation")

# # read directory
# directory = "../data/train/"
# files = os.listdir(directory)
# for i in files:
#     imgs = os.listdir(directory + i)
#     img = cv2.imread(directory + i + "/" + imgs[0])
#     os.remove(directory + i + "/" + imgs[0])

#     if os.path.exists("../data/validation/" + i):
#         print("Validation set already exists")
#     else:
#         os.mkdir("../data/validation/" + i)

#     cv2.imwrite("../data/validation/" + i + "/" + imgs[0], img)

import torch
from torch import nn
from torchvision import models

model = models.resnet101()
print(model)

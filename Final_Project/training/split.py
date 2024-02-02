import os
import cv2

os.mkdir("../data/validation")

for i in os.listdir("../data/train"):
    os.mkdir("../data/validation/" + i)
    for j in os.listdir("../data/train/" + i):
        img = cv2.imread("../data/train/" + i + "/" + j)
        cv2.imwrite("../data/validation/" + i + "/" + j, img)
        os.remove("../data/train/" + i + "/" + j)
        break

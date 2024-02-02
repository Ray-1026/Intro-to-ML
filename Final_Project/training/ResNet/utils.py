import os
from PIL import Image
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset


train_tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(
            45, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
    ]
)

test_tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class TestingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        data
        ├── test
        |   ├── xxxxx.jpg
        |   ├── ...
        |   └── yyyyy.jpg
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.names = []

        self.images = sorted(glob.glob(f"{self.img_dir}/*"))
        self.names = [os.path.basename(image)[:-4] for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getnames__(self):
        return self.names

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        return image

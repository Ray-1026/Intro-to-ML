import random
import numpy as np
import torchvision.transforms as transforms


train_tfm = transforms.Compose(
    [
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_tfm = transforms.Compose(
    [
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# utils
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n**2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[
            ...,
            x * block_size : (x + 1) * block_size,
            y * block_size : (y + 1) * block_size,
        ].clone()
        jigsaws[
            ...,
            x * block_size : (x + 1) * block_size,
            y * block_size : (y + 1) * block_size,
        ] = temp

    return jigsaws

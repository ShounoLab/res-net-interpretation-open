# https://arxiv.org/pdf/1409.1556.pdf
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os

import torchvision.transforms as transforms

from .my_folder import MyImageFolder
from .path import ILSVRC2012_DATASET_PATH

PYTORCH_IMAGENET_MEAN = [0.485, 0.456, 0.406]
PYTORCH_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_ilsvrc2012(mode="train", transform_type="vgg", image_size=224, val_txt=None):
    normalize = transforms.Normalize(
        mean=PYTORCH_IMAGENET_MEAN, std=PYTORCH_IMAGENET_STD
    )
    if transform_type is None:
        transform = None
    elif transform_type == "vgg_train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif transform_type == "vgg" or transform_type == "vgg_test":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif transform_type == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif transform_type == "test":
        transform = transforms.Compose(
            [
                transforms.Resize(int(8 / 7 * image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif isinstance(transform_type, transforms.Compose):
        transform = transform_type

    if mode == "train":
        txt_path = os.path.join(ILSVRC2012_DATASET_PATH, "train.txt")
        train_dataset = MyImageFolder(ILSVRC2012_DATASET_PATH, txt_path, transform)
        return train_dataset

    elif mode == "test":
        if val_txt is None:
            txt_path = os.path.join(ILSVRC2012_DATASET_PATH, "val.txt")
        elif os.path.exists(val_txt):
            txt_path = val_txt
        else:
            raise TypeError(val_txt)

        test_dataset = MyImageFolder(ILSVRC2012_DATASET_PATH, txt_path, transform)
        return test_dataset

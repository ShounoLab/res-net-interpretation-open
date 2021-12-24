# https://arxiv.org/pdf/1409.1556.pdf
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os

import chainercv.transforms as chainer_transforms
import numpy as np
import torchvision.transforms as transforms
from chainer.datasets import TransformDataset
from chainercv.datasets import CUBLabelDataset

from .my_folder import MyImageFolder
from .path import ILSVRC2012_DATASET_PATH

PYTORCH_IMAGENET_MEAN = [0.485, 0.456, 0.406]
PYTORCH_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_ilsvrc2012(mode="train", transform_type="vgg", image_size=224, val_txt=None):
    #  TODO: 訓練の前処理を正しくする。
    #  TODO: 評価データをロードできるようにする。
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


def get_imagenet100(mode="train", transform_type="vgg", image_size=224):
    #  TODO: 訓練の前処理を正しくする。
    #  TODO: 評価データをロードできるようにする。
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

    data_path = "/data1/dataset/imagenet100/"
    if mode == "train":
        txt_path = os.path.join(data_path, "train_imagenet100.txt")
        dataset = MyImageFolder(data_path, txt_path, transform)
        return dataset

    elif mode == "test":
        txt_path = os.path.join(data_path, "val_imagenet100.txt")
        dataset = MyImageFolder(data_path, txt_path, transform)
        return dataset


def get_CUB(mode="vanilla"):
    #  TODO: 訓練の前処理を正しくする。
    #  TODO: 評価データをロードできるようにする。
    dataset = CUBLabelDataset()
    if mode == "vanilla":

        def transform(in_data):
            img, label = in_data
            img = chainer_transforms.resize(img, (256, 256))
            img = chainer_transforms.center_crop(img, (224, 224))
            img -= np.asarray([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
            img /= np.asarray([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]
            return img, label

        return TransformDataset(dataset, transform)
    elif mode == "train":
        pass
    elif mode == "test":
        pass

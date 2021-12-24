import os
import pickle

from utils import config
from utils.load_model import TransferModel, get_model, get_submodel

resnet34_model_names = [
    "20200409_resnet34",
    "resnet34-skip_",
    "resnet34-skip2_",
    "resnet34-skip3_",
]

plainnet34_model_names = [
    "20200409_plainnet34",
    "resnet34-plain_",
    "resnet34-plain2_",
    "resnet34-plain3_",
]

resnet34_model_keys = [
    "resnet34-skip4",
    "resnet34-skip",
    "resnet34-skip2",
    "resnet34-skip3",
]

plainnet34_model_keys = [
    "resnet34-plain4",
    "resnet34-plain",
    "resnet34-plain2",
    "resnet34-plain3",
]

labels_resnets = [
    "ResNet34",
    "ResNet34-1",
    "ResNet34-2",
    "ResNet34-3",
]
labels_plainnets = [
    "PlainNet34",
    "PlainNet34-1",
    "PlainNet34-2",
    "PlainNet34-3",
]

analyis_layers = [
    "maxpool",
    "layer1.0",
    "layer1.1",
    "layer1.2",
    "layer2.0",
    "layer2.1",
    "layer2.2",
    "layer2.3",
    "layer3.0",
    "layer3.1",
    "layer3.2",
    "layer3.3",
    "layer3.4",
    "layer3.5",
    "layer4.0",
    "layer4.1",
    "layer4.2",
]

exts = ("png", "pdf")


def get_model_from_keywords(arch, wandb_flag, no_grad_fe=False):
    if wandb_flag:
        config_path = os.path.join(arch, "config.yaml")
        wandb_config = config.dict_from_config_file(config_path)
        feature_extractor = get_submodel(wandb_config["feature_extractor"]["value"])
        classifier = get_submodel(wandb_config["classifier"]["value"], num_classes=1000)
        model = TransferModel(feature_extractor, classifier, no_grad_fe=no_grad_fe)
    else:
        model = get_model(arch)

    return model


def get_arch_name(arch, wandb_flag):
    if wandb_flag:
        config_path = os.path.join(arch, "config.yaml")
        wandb_config = config.dict_from_config_file(config_path)
        feature_extractor_tmp = wandb_config["feature_extractor"]["value"]
        classifier_tmp = wandb_config["classifier"]["value"]
        arch_name = "FE_{}_CL_{}".format(feature_extractor_tmp, classifier_tmp)
    elif os.path.exists(arch):
        dname = os.path.dirname(arch)
        arch_name = os.path.basename(dname)
    else:
        arch_name = arch

    return arch_name


def save_arr(out_path, arr):
    with open(out_path, "wb") as f:
        pickle.dump(arr, f)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

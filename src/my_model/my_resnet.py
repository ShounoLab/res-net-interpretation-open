import os
from collections import OrderedDict

import torch
import torch.utils.model_zoo as model_zoo

from .blocks import BasicBlock, BasicBlockGeneral, Bottleneck, ResNet

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, map_location=torch.device("cpu"), **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        plain(bool): If True, return a plain model
        block_mode(str): If select number of mode, apply general block
        num_classes(int): output dimension
        channel(int or list): default = 64. If int value, set initial channel.
                              If list value with length 5, set each list value to channels.
    """
    if "block_mode" in kwargs:
        print("Block mode: {}".format(kwargs["block_mode"]))
        model = ResNet(BasicBlockGeneral, [3, 4, 6, 3], **kwargs)
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if isinstance(pretrained, bool):
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    elif isinstance(pretrained, str):
        currenct_path = os.path.dirname(os.path.abspath(__file__))
        path = None
        if kwargs["plain"]:
            if pretrained == "my":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.plain.final.model"
                )
                model.load_state_dict(torch.load(path, map_location=map_location))
            elif pretrained == "my2":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.plain20200124.final.model"
                )
                model.load_state_dict(torch.load(path, map_location=map_location))
            elif pretrained == "my3":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.plain20200814.final.model"
                )
                model.load_state_dict(
                    fix_model_state_dict(torch.load(path, map_location=map_location))
                )
            elif pretrained == "my4":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.plain20200409.final.model"
                )
                model.load_state_dict(
                    fix_model_state_dict(torch.load(path, map_location=map_location))
                )

        else:
            if pretrained == "my":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.skip.final.model"
                )
                model.load_state_dict(torch.load(path, map_location=map_location))
            elif pretrained == "my2":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.skip20200115.final.model"
                )
                model.load_state_dict(torch.load(path, map_location=map_location))
            elif pretrained == "my3":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.skip20200120.final.model"
                )
                model.load_state_dict(torch.load(path, map_location=map_location))
            elif pretrained == "my4":
                path = os.path.join(
                    currenct_path, "./trained_model/resnet34.skip20200409.final.model"
                )
                model.load_state_dict(
                    fix_model_state_dict(torch.load(path, map_location=map_location))
                )

            elif pretrained == "pytorch":
                path = model_urls["resnet34"]
                model.load_state_dict(model_zoo.load_url(path))
        if not pretrained:
            # if pretrained == False then do not loading weight
            path = 0

        if path is None:
            raise ValueError(
                "unkown words; pretrained:{}, plain:{}".format(
                    pretrained, kwargs["plain"]
                )
            )

    return model


def resnet50(pretrained=False, path=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if path is not None:
        model.load_state_dict(fix_model_state_dict(torch.load(path)))
        return model

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model

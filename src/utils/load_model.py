import os
from collections import OrderedDict

import numpy as np
import torch
import torchvision

import my_model


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class TransferModel(torch.nn.Module):
    def __init__(self, feature_extractor, classifier, no_grad_fe=True):
        super(TransferModel, self).__init__()
        self._no_grad_fe = no_grad_fe
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = classifier

    def forward(self, x, return_feature=False, layer=None):
        fe_layer = None
        cl_layer = None
        if layer is not None:
            index = layer.find(".")
            if index == -1:
                k = layer
                prop_layer = ""
            else:
                k = layer[:index]
                prop_layer = layer[index + len(".") :]
            if k == "classifier":
                fe_layer = None
                cl_layer = prop_layer
            elif k == "feature_extractor":
                fe_layer = prop_layer
                cl_layer = None

        if self._no_grad_fe:
            with torch.no_grad():
                feature = self.feature_extractor(x, layer=fe_layer)
        else:
            feature = self.feature_extractor(x, layer=fe_layer)

        out = self.classifier(feature, layer=cl_layer)
        if return_feature:
            return out, feature
        return out


def save_params(model):
    save_dir = "./save_{}/".format(type(model).__name__)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for param in model.named_parameters():
        print(param[0])
        fname = "{}".format(param[0])
        save_path = os.path.join(save_dir, fname)
        np_data = param[1].data.numpy()
        np.save(save_path, np_data)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[len("module.") :]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def get_model(arch, fix=False, verbose=True, onto_device="cpu", **kwargs):

    """
    enable archs
    {
        resnet34-pytorch:  model of pytorch
        resnet34, resnet34-skip: my resnet34 model 1
        resnet34-skip2: my resnet34 model 2
        resnet34-skip3: my resnet34 model 3
        resnet34-skip4: my resnet34 model 4
        resnet34-sikp-random: resnet34 random weight model
        resnet34-plain: my plainnet34 model 1
        resnet34-plain: my plainnet34 model 1
        resnet34-plain2: my plainnet34 model 2
        resnet34-plain3: my plainnet34 model 3
        resnet34-plain4: my plainnet34 model 4
        resnet34-plain-random: plainnet34 random weight model
        resnet50-pytorch:  model of pytorch
    }

    If arch is a path of the model, load model from the path
    """
    if os.path.exists(arch):
        if verbose:
            print("load ==> {}".format(arch))
        state_dict = fix_model_state_dict(
            torch.load(arch, map_location=torch.device(onto_device))
        )

        if "plain" in arch:
            plain = True
        else:
            plain = False
        model = my_model.resnet34(pretrained=False, plain=plain, **kwargs)
        model.load_state_dict(state_dict)
        return model

    currenct_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(currenct_path, "../my_model/trained_model/")
    if arch == "resnet34-pytorch":
        model = my_model.resnet34(
            pretrained=True, map_location=torch.device(onto_device), **kwargs
        )
    elif arch == "resnet50-pytorch":
        model = my_model.resnet50(
            pretrained=True, map_location=torch.device(onto_device), **kwargs
        )
    elif arch == "resnet34-plain-random":
        model = my_model.resnet34(
            pretrained=False,
            plain=True,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34-skip-random":
        model = my_model.resnet34(
            pretrained=False,
            plain=False,
            map_location=torch.device(onto_device),
            **kwargs
        )
    # trained weight models
    elif arch == "resnet34-plain":
        model = my_model.resnet34(
            pretrained="my",
            plain=True,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34" or arch == "resnet34-skip":
        model = my_model.resnet34(
            pretrained="my",
            plain=False,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34-skip2":
        model = my_model.resnet34(
            pretrained="my2",
            plain=False,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34-skip3":
        model = my_model.resnet34(
            pretrained="my3",
            plain=False,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34-skip4":
        model = my_model.resnet34(
            pretrained="my4",
            plain=False,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34-plain2":
        model = my_model.resnet34(
            pretrained="my2",
            plain=True,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34-plain3":
        model = my_model.resnet34(
            pretrained="my3",
            plain=True,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet34-plain4":
        model = my_model.resnet34(
            pretrained="my4",
            plain=True,
            map_location=torch.device(onto_device),
            **kwargs
        )
    elif arch == "resnet50-skip":
        path = os.path.join(dir_path, "resnet50.skip20200821.final.model")
        model = my_model.resnet50(path=path, plain=False, **kwargs)
    elif arch == "resnet50-plain":
        path = os.path.join(dir_path, "resnet50.plain20200821.final.model")
        model = my_model.resnet50(path=path, plain=True, **kwargs)
    else:
        model = None
    if fix:
        state_dict = fix_model_state_dict(model.state_dict())
        model.load_state_dict(state_dict)

    return model


def name_convetor(m_name):
    if "resnet34" in m_name:
        num = m_name.split(".")[-1]
        if num.isdigit():
            if int(num) < 1:
                return "resnet34-pytorch"
            elif int(num) == 1:
                return "resnet34-skip"
            else:
                return "resnet34-skip{}".format(num)
        else:
            return "resnet34-skip-random"
    elif "plainnet34" in m_name:
        num = m_name.split(".")[-1]
        if num.isdigit():
            if int(num) < 1:
                raise ValueError(m_name)
            if int(num) == 1:
                return "resnet34-plain"
            else:
                return "resnet34-plain{}".format(num)
        else:
            return "resnet34-plain-random"

    return m_name


def get_submodel(name, split_key="-", **kwargs):
    """
    style of name:
        {model name}-{first layer}-{last layer}
        (Can change split keyword "-" to use split_key.)

    get sub modules (children) from first layer until last layer
    of get_model().

    If first layer is layer1 and last layer is layer3 then
    return modules consits of layer1, layer2 and layer3

    NOTE: Not working for sub layer. e.g. layer3.4, layer3.4.relu1 etc.
    """
    model_name, first_layer, last_layer = name.split(split_key)
    model_name = name_convetor(model_name)
    base_model = get_model(model_name, **kwargs)
    assert isinstance(
        base_model, my_model.blocks.ResNet
    ), "should be ResNet or PlainNet"
    if last_layer == "fc":
        need_flatten = True
    else:
        need_flatten = False

    nn_list = []
    set_flag = False
    for c_name, child in base_model.named_children():
        if first_layer == c_name:
            set_flag = True
        if need_flatten and c_name == "fc":
            nn_list.append(("flatten", Flatten()))
        if set_flag:
            nn_list.append((c_name, child))
        if last_layer == c_name:
            set_flag = False

    model = my_model.blocks.MySequential(OrderedDict(nn_list))

    return model


def get_resblock(model, layer_name):
    """
    model is a class resnet
    e.g. layer_name is "layer1.0.relu2"
    """
    obj_name, b_idx, act_name = layer_name.split(".")
    b_idx = int(b_idx)
    block = getattr(model, obj_name)[b_idx]
    return block


def get_prelayer_name(layer_num, block_num, model_type="34"):
    if model_type != "34":
        msg = "only support resnet34 or plainnet34. not {}".format(model_type)
        raise ValueError(msg)

    base_name = "layer{}.{}.relu2"
    if block_num == 0:
        if layer_num == 1:
            prelayer_name = "maxpool"
        else:
            if layer_num == 2:
                j = 2
            elif layer_num == 3:
                j = 3
            elif layer_num == 4:
                j = 5
            else:
                msg = "MUST: number of layer <= {}, not {}".format(4, layer_num)
                raise ValueError(msg)
            prelayer_name = base_name.format(layer_num - 1, j)
    else:
        prelayer_name = base_name.format(layer_num, block_num - 1)

    return prelayer_name


if __name__ == "__main__":
    model = torchvision.models.resnet34(pretrained=True)
    save_params(model)

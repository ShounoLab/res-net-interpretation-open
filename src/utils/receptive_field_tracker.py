"""
    MUST using original resnet model (adding AddFunction class)
    >>> import my_resnet

    Example:
    # get final receptive field size
    >>> model = my_resnet34()
    >>> image_size = (1, 3, 224, 224)
    >>> input_data = torch.Tensor(np.random.normal(size=image_size))
    >>> handler = HookHandler(ReceptiveFieldHook())
    >>> handler.register_hooks(model)
    >>> output_data = model(input_data)
    >>> hander.hook.final_rf
    (7, 32, 899, 31.5)

    # the information is for my_model.get_receptive_field()

    # reset hook
    >>> handler.reset()

    # remove all handler of hooks
    >>> handler.remove()


    # get corner points of receptive field by back propagation
    >>> model = my_resnet34()
    >>> image_size = (1, 3, 224, 224)
    >>> input_data = torch.Tensor(np.random.normal(size=image_size))
    >>> input_data.requires_grad = True
    >>> handler = HookHandler(ReceptiveFieldHookGrad())
    >>> handler.register_hooks(model, backward=True)
    >>> output_data = model.layer_forward(input_data, layers=['layer1.0.conv1'])
    >>> onehot = np.zeros(output_data.shape)
    >>> onehot[0, 0, output_data.shape[2]//2, output_data.shape[3]//2] = 1
    >>> onehot = torch.Tensor(onehot)
    >>> output_data.backward(gradient=onehot)
    >>> tl_index, br_index =  get_corner_point(input_data.grad[0])

"""


import math
from collections import OrderedDict
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from my_model.blocks import ResNet

from .receptive_field import get_receptive_field, get_rf_region


def get_rf_layer_info(model, img, layer, outmap=True):

    rf_tracker = RFTracker(model)
    with torch.no_grad():
        if isinstance(model, (ResNet,)):
            out = model(img, layers=[layer])[layer]
        else:
            out = model(img, layer=layer)

    rf_info = rf_tracker.get_rf_info(layer)
    rf_tracker.remove_hook()

    if outmap:
        return rf_info, out
    return rf_info


# https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
def outFromIn(conv, layerIn, cover_all=False):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]
    if cover_all:
        n_out = math.floor((n_in - k + 2 * p + s - 1) / s) + 1
    else:
        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        actualP = (n_out - 1) * s - n_in + k
        # pR = math.ceil(actualP / 2)
        pL = math.floor(actualP / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - pL) * j_in
        # start_out = start_in + ((k-1)/2 - p)*j_in
    return n_out, j_out, r_out, start_out


def all_child(func):
    def wrapper(*args, **kwargs):
        parent = args[0]
        if hasattr(parent, "named_children"):
            for name, child in parent.named_children():
                if hasattr(child, "named_children"):
                    if len(list(child.named_children())) == 0:
                        func(child, name, **kwargs)
                    else:
                        wrapper(child, **kwargs)

    return wrapper


def get_corner_point(grad_img):
    if isinstance(grad_img, torch.Tensor):
        grad_img = grad_img.detach().numpy()

    assert isinstance(grad_img, np.ndarray)
    assert grad_img.ndim == 3
    # (channel, height, width)

    rf_index = (grad_img != 0).sum(axis=0).reshape(-1)
    result = np.where(rf_index > 0)[0]
    if len(result) > 0:
        top_left_index = np.unravel_index(
            result[0], (grad_img.shape[1], grad_img.shape[2])
        )
        bottom_right_index = np.unravel_index(
            result[-1], (grad_img.shape[1], grad_img.shape[2])
        )
    else:
        top_left_index = (0, 0)
        bottom_right_index = (grad_img.shape[1], grad_img.shape[2])
        # raise ValueError

    return top_left_index, bottom_right_index


class RFTracker(object):
    def __init__(self, model, candidate_layers=None, device="cpu"):
        """
        To track receptive field information of pytorch model.
        If any candidates are not specified, the hook is registered to all the layers.
        Eg.
            tracker = TensorTracker(model)
            output_data = model(input_data)
            rf_info = tracker.find_receptive_field("layer")
        """
        super(RFTracker, self).__init__()
        self.model = model
        self.handlers = []
        self.rf_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list
        self._device = device
        self.rfs = {}
        self.done_in_ids = set()
        self.done_out_ids = set()
        self.id_names = {}

        def forward_hook(key):
            def forward_hook_(module, input_data, output_data):
                # input_shape is only used for adaptive pooling
                info = self._parse_module(module, input_data[0].shape)
                flag = True
                for in_data in input_data:
                    flag = flag & (id(in_data) in self.done_in_ids)
                if id(output_data) in self.done_out_ids and flag:
                    pass
                elif len(input_data) == 1:
                    in_data = input_data[0]
                    # N, C, H, W
                    size = tuple(in_data.size())
                    # first process
                    if len(self.rfs) == 0:
                        self.rfs[id(in_data)] = (size[-1], 1, 1, 0.5)
                    # use information of module
                    if info is not None:
                        self.rfs[id(output_data)] = outFromIn(
                            info, self.rfs[id(in_data)], False
                        )
                    else:
                        self.rfs[id(output_data)] = self.rfs[id(in_data)]

                # pass maximum input size
                elif len(input_data) == 2:
                    assert info is None
                    max_var = (-1, -1, -1, -1)
                    for in_data in input_data:
                        if id(in_data) in self.done_in_ids:
                            continue
                        tmp_var = self.rfs[id(in_data)]
                        if max_var[2] < tmp_var[2]:
                            max_var = tmp_var
                    if max_var[2] == -1:
                        return
                    self.rfs[id(output_data)] = max_var
                else:
                    raise ValueError(len(input_data))

                for in_data in input_data:
                    pass
                # print(id(input_data[0].data), id(output_data.data))
                # print(len(input_data), id(input_data[0]), id(output_data), self.rfs[id(output_data)], key)

                self.rf_pool[key] = copy(self.rfs[id(output_data)])

            return forward_hook_

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Sequential,)):
                self.handlers.append(module.register_forward_hook(forward_hook(name)))

    def __del__(self):
        self.remove()

    def remove(self):
        self.remove_hook()
        del self.rf_pool
        del self.candidate_layers

    def find_receptive_field(self, neuron_index, target_layer, **kwargs):
        """
        Args:
            neuron_index: tuple of int
                the position of the neuron where you want to find the receptive field.
            target_layer: string
                the layer name
        Return:
            center: tuple or int
            receptive_field: tuple or int
        """
        rf_info = self.get_rf_info(target_layer)
        return get_receptive_field(neuron_index, rf_info, **kwargs)

    def find_receptive_field_region(
        self, neuron_index, target_layer, image_size, **kwargs
    ):
        """
        Args:
            neuron_index: tuple of int
                the position of the neuron where you want to find the receptive field.
            target_layer: string
                the layer name
            image_size: int
                input image size
        Return:
            heights: tuple of int
            widths: tuple of int
        """
        center, rf = self.find_receptive_field(neuron_index, target_layer)
        return get_rf_region(center, rf, **kwargs)

    def get_rf_info(self, target_layer):
        """
        get the information of the receptive field
        target_layer: string
            the layer name
        """
        return self._find(self.rf_pool, target_layer)

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _parse_module(self, module, input_shape):
        class_name = str(type(module))
        if "Conv2d" in class_name:
            w = tuple(module.weight.size())[-1]
            s = module.stride[-1]
            p = module.padding[-1]
            info = [w, s, p]
        elif "Adaptive" in class_name:
            # adaptive average pooling or max pooling
            w = input_shape[-1]
            s = 1
            p = 0
            info = [w, s, p]
        elif "Pool2d" in class_name:
            w = module.kernel_size
            s = module.stride
            p = module.padding
            info = [w, s, p]
        else:
            info = None

        return info

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

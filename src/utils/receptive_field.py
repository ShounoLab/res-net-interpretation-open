import math

import numpy as np
import torch
from torch.nn import functional as F

from .ReceptiveFieldHook import HookHandler, ReceptiveFieldHook


def cut_rf_from_img_helper(
    img,
    neuron_index,
    _layer_info,
    clipping=False,
    is_numpy=False,
    pad_size=None,
    **kwargs
):
    if pad_size is None:
        pad_size = _layer_info[2]

    layer_info = (_layer_info[0], _layer_info[1], pad_size, _layer_info[3])
    center, rf_ration = get_receptive_field(neuron_index, layer_info)

    image_size = img.shape[-1]
    xs, ys = get_rf_region(
        center, rf_ration, clipping=clipping, image_size=image_size, **kwargs
    )

    out = cut_rf_from_img(img, xs, ys, pad_size, is_numpy=is_numpy)

    return out


#  my_model/receptive_field.py
def get_receptive_field(neuron_index, layer_info):
    """
    neuron_index: tuple of length 2 or int represented x axis and y axis
    layer_info: tuple of length 4 consist of the information of receptive_field
    """
    n, j, rf, start = layer_info
    if isinstance(neuron_index, tuple):
        center_y = start + (neuron_index[1]) * (j)
        center_x = start + (neuron_index[0]) * (j)
    else:
        center_y = start + (neuron_index // n) * (j)
        center_x = start + (neuron_index % n) * (j)
    return (center_x, center_y), (rf / 2, rf / 2)


def cut_rf_from_img(img, xs, ys, pad_size, is_numpy=True):
    """
    padding last 2 dim same pad_size
    """
    assert isinstance(img, torch.Tensor)
    pad_img = F.pad(img, (pad_size, pad_size, pad_size, pad_size)).to("cpu")
    if len(pad_img.shape) == 3:
        pad_img = pad_img[
            :, xs[0] + pad_size : xs[1] + pad_size, ys[0] + pad_size : ys[1] + pad_size
        ]
    elif len(pad_img.shape) == 2:
        pad_img = pad_img[
            xs[0] + pad_size : xs[1] + pad_size, ys[0] + pad_size : ys[1] + pad_size
        ]

    if is_numpy:
        return pad_img.detach().numpy()
    else:
        return pad_img


def clip(x, min_val, max_val, clip_flag=False):
    flag = 0
    if x < min_val:
        x = min_val
        flag = -1
    elif max_val < x:
        x = max_val
        flag = 1
    if clip_flag:
        return x, flag
    else:
        return x


def get_rf_region(
    center, rf, image_size=224, detect_remain_corner=False, clipping=True
):
    k = 0
    x1 = math.floor(center[k] - rf[k])
    x2 = math.floor(center[k] + rf[k])
    if clipping:
        if detect_remain_corner:
            top, t_flow = clip(x1, 0, image_size, True)
            bottom, b_flow = clip(x2, 0, image_size, True)
            xs = (top, bottom)
        else:
            xs = (clip(x1, 0, image_size), clip(x2, 0, image_size))
    else:
        xs = (x1, x2)

    k = 1
    x1 = math.floor(center[k] - rf[k])
    x2 = math.floor(center[k] + rf[k])
    if clipping:
        if detect_remain_corner:
            left, l_flow = clip(x1, 0, image_size, True)
            right, r_flow = clip(x2, 0, image_size, True)
            ys = (left, right)
        else:
            ys = (clip(x1, 0, image_size), clip(x2, 0, image_size))
    else:
        ys = (x1, x2)

    if clipping and detect_remain_corner:
        corners = {"top left", "top right", "bottom left", "bottom right"}
        if t_flow > 0:
            raise ValueError(
                "top of receptive field is greater than {}. receptive field is None".format(
                    image_size
                )
            )
        if b_flow < 0:
            raise ValueError(
                "bottom of receptive field is negative value. receptive field is None"
            )
        if l_flow > 0:
            raise ValueError(
                "left of receptive field is greater than {}. receptive field is None".format(
                    image_size
                )
            )
        if r_flow < 0:
            raise ValueError(
                "right of receptive field is negative value. receptive field is None"
            )

        if t_flow < 0:
            for x in set(filter(lambda x: "top" in x, corners)):
                corners.remove(x)
        if b_flow > 0:
            for x in set(filter(lambda x: "bottom" in x, corners)):
                corners.remove(x)
        if l_flow < 0:
            for x in set(filter(lambda x: "left" in x, corners)):
                corners.remove(x)
        if r_flow > 0:
            for x in set(filter(lambda x: "right" in x, corners)):
                corners.remove(x)

        return (xs, ys), corners
    else:
        return xs, ys


def get_rf_layer_info(model, img, layer, outmap=True):
    handler = HookHandler(ReceptiveFieldHook())
    handler.register_hooks(model)
    handler.reset()
    with torch.no_grad():
        out = model(img, layers=[layer])[layer]
    layer_info = handler.hook.final_rf
    handler.reset()
    handler.remove()
    if outmap:
        return layer_info, out
    return layer_info


def get_downconv(np_conv, np_down_weight):
    assert isinstance(np_conv, np.ndarray)
    assert isinstance(np_down_weight, np.ndarray)
    high_conv = np_down_weight.copy()
    if high_conv.ndim == 4:
        if high_conv.shape[-1] > 1:
            out_ch = high_conv.shape[0]
            in_ch = high_conv.shape[1]
            high_conv = high_conv.reshape(int(out_ch * in_ch), -1)
            index = np.abs(high_conv).argmax(axis=-1)
            high_conv = high_conv[range(len(index)), index]
            high_conv = high_conv.reshape(out_ch, in_ch)
        else:
            high_conv = high_conv[:, :, 0, 0]
    v_array = np.einsum("ki,ijnm->kjnm", high_conv, np_conv)
    return v_array

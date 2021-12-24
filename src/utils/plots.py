import os

import matplotlib.pyplot as plt
import numpy as np
import torch

PYTORCH_IMAGENET_MEAN = [0.485, 0.456, 0.406]
PYTORCH_IMAGENET_STD = [0.229, 0.224, 0.225]


def norm_img(x):
    h = x.copy()
    h -= h.min()
    h /= h.max()
    return h


def plot_imshows(
    imgs,
    nrow="auto",
    ncol=None,
    im_mode="rgb",
    out_dir=".",
    normalize="all",
    out_name=None,
    show_flag=False,
    title=None,
    scale=None,
    fontsize=None,
    vrange=None,
    exts=("png",),
):

    """
    plot images.

    parameters:
        imgs:
            numpy.ndarray
            or
            (list, tuple) of numpy.ndarray

            shape N C H W
            or
            shape N H W
    """
    if isinstance(imgs, np.ndarray):
        if normalize == "all":
            img_max = imgs.max()
            img_min = imgs.min()

    elif isinstance(imgs, list) or isinstance(imgs, tuple):
        assert isinstance(imgs[0], np.ndarray)
        if normalize == "all":
            img_max = max([img.max() for img in imgs])
            img_min = min([img.min() for img in imgs])
    else:
        raise TypeError("Unknown type of imgs: {}".format(type(imgs)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if nrow == "auto":
        nrow = int(np.ceil(np.sqrt(len(imgs))))
        ncol = int(np.ceil(np.sqrt(len(imgs))))
    elif isinstance(nrow, int):
        nrow = min(nrow, len(imgs))
        if ncol is None:
            ncol = int(np.ceil(float(len(imgs)) / nrow))

    if scale is None:
        scale = 10 / 8
    plt.figure(figsize=(ncol * scale, nrow * scale))
    for i, w in enumerate(imgs):
        plt.subplot(nrow, ncol, i + 1)
        if w.ndim == 3:
            img = np.transpose(w, (1, 2, 0))
        elif w.ndim == 2:
            img = w
        else:
            raise ValueError(len(w.ndim))

        if normalize == "all":
            img = (img - img_min) / (img_max - img_min)
        elif normalize == "each":
            img = norm_img(img)
        elif callable(normalize):
            img = normalize(img)
        if vrange is None:
            plt.imshow(img)
        else:
            plt.imshow(img, vmin=0, vmax=1)
        plt.axis("off")
        if title is None:
            pass
        elif title == "count":
            if fontsize is None:
                plt.title(i)
            else:
                plt.title(i, fontsize=fontsize)
        elif callable(title):
            plt.title(title(i, w))
    plt.tight_layout()
    if out_name is not None:
        if "png" in out_name:
            path = os.path.join(out_dir, "{}".format(out_name))
            plt.savefig(path, transparent=True)
        elif exts is not None:
            for ext in exts:
                path = os.path.join(out_dir, "{}.{}".format(out_name, ext))
                plt.savefig(path, transparent=True)
        else:
            path = os.path.join(out_dir, "{}.png".format(out_name))
            plt.savefig(path, transparent=True)
    if show_flag:
        plt.show()
    plt.close()


def imshow_helper(x, cmap=None):
    assert isinstance(x, np.ndarray)
    if x.ndim == 3:
        if x.shape[0] == 3:
            # CHW
            x = np.transpose(x, (1, 2, 0))
        elif x.shape[-1] == 3:
            # HWC
            pass
        else:
            raise ValueError("unknown format. need CHW or HWC")

    if cmap is None:
        plt.imshow(x)
    else:
        plt.imshow(x, cmap=cmap)


PYTORCH_IMAGENET_MEAN = [0.485, 0.456, 0.406]
PYTORCH_IMAGENET_STD = [0.229, 0.224, 0.225]


def input2image(data, clip=True, img_format="CHW", mean=None, std=None):
    if mean is None:
        mean = PYTORCH_IMAGENET_MEAN
    if std is None:
        std = PYTORCH_IMAGENET_STD
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    elif isinstance(data, np.ndarray):
        data = data.copy()

    mean = np.asarray(mean)
    std = np.asarray(std)

    if data.ndim == 3:
        if img_format == "CHW":
            img = data * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
        elif img_format == "HWC":
            img = data * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    elif data.ndim == 4:
        if img_format == "CHW":
            img = data * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
        elif img_format == "HWC":
            img = data * std.reshape(1, 1, 1, -1) + mean.reshape(1, 1, 1, -1)
    else:
        raise ValueError(data.ndim)

    return np.clip(img, 0, 1)


def normalize_inputspace(
    x,
    vmax=1,
    vmin=0,
    mean=PYTORCH_IMAGENET_MEAN,
    std=PYTORCH_IMAGENET_STD,
    each=True,
    img_format="CHW",
):
    """
    Args:
        x: numpy.ndarray
            format is CHW or BCHW
        each: bool
            if x has dimension B
            then apply each input x.
    Returns:
        normalized x: numpy.ndarray
    """
    if len(x.shape) == 3:
        return normalize3d_inputspace(x, vmax, vmin, mean, std, img_format=img_format)
    elif len(x.shape) == 4:
        if each:
            return np.array(
                [
                    normalize_inputspace(
                        _x, vmax, vmin, mean, std, img_format=img_format
                    )
                    for _x in x
                ]
            )
        else:
            # TODO:
            raise ValueError(each)


def normalize3d_inputspace(
    x,
    vmax=1,
    vmin=0,
    mean=PYTORCH_IMAGENET_MEAN,
    std=PYTORCH_IMAGENET_STD,
    img_format="CHW",
):
    """
    Args:
        x:  numpy.ndarray
        format is CHW
    Returns:
        normalized x: numpy.ndarray

    Note:
    if input_vmax < hmax * input_vmin / hmin
        => hmax / hmin < input_vmax / input_vmin
        => h_ratio < m_ratio
        then input_vmax / hmax

    if hmin * input_vmax / hmax < input_vmin
        => input_vmax / input_vmin < hmax / hmin
        => m_ratio < h_ratio
        then input_vmin / hmin
    """
    if img_format == "CHW":
        assert len(x.shape) == 3 and x.shape[0] == 3
    elif img_format == "HWC":
        assert len(x.shape) == 3 and x.shape[-1] == 3
        x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError(img_format)

    mean = np.array(mean)
    std = np.array(std)
    input_vmin = (vmin - mean) / std
    input_vmax = (vmax - mean) / std

    h = np.clip(x, input_vmin[..., None, None], input_vmax[..., None, None])
    hmin = np.minimum(h.min((-2, -1)), 0)
    hmax = np.maximum(h.max((-2, -1)), 0)
    min0 = hmin == 0
    max0 = hmax == 0
    # to void zero div warnings
    hmin[min0] = 1
    hmax[max0] = 1

    h_ratio = hmax / hmin
    m_ratio = input_vmax / input_vmin
    flags = m_ratio < h_ratio
    coeff_max = input_vmax / hmax
    coeff_min = input_vmin / hmin
    coeffs = []
    for cnt, flag in enumerate(flags):
        if max0[cnt] and min0[cnt]:
            coeffs.append(1)
        elif max0[cnt]:
            coeffs.append(coeff_min[cnt])
        elif min0[cnt]:
            coeffs.append(coeff_max[cnt])
        elif flag:
            coeffs.append(coeff_min[cnt])
        else:
            coeffs.append(coeff_max[cnt])
    coeffs = np.asarray(coeffs)
    hs = h * coeffs[..., None, None]
    if img_format == "CHW":
        return hs
    elif img_format == "HWC":
        return np.transpose(hs, (1, 2, 0))


# https://pytorch.org/docs/stable/_modules/torchvision/utils.html#make_grid
def make_grid(imgs, nrow=8, padding=2, pad_value=0, im_mode="rgb", copy=True):
    assert isinstance(imgs, np.ndarray)
    imgs = imgs.copy()

    if im_mode == "rgb":
        # N C H W
        assert imgs.ndim == 3 or imgs.ndim == 4
        if imgs.ndim == 3:
            imgs = imgs[None, ...]

        # C = 3
        assert imgs.shape[1] == 3

    elif im_mode == "gray":
        assert imgs.nim == 2 or imgs.ndim == 3
        if imgs.ndim == 2:
            imgs = imgs[None, ...]
        grid = np.zeros(())
    else:
        raise ValueError(im_mode)

    nmaps = len(imgs)
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))

    height = imgs.shape[-2] + padding
    width = imgs.shape[-1] + padding

    if im_mode == "rgb":
        grid = (
            np.zeros((imgs.shape[1], height * ymaps + padding, width * xmaps + padding))
            + pad_value
        )
    elif im_mode == "gray":
        grid = np.zeros((height * ymaps + padding, width * xmaps + padding)) + pad_value

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break

            hslice = slice(y * height + padding, (y + 1) * height)
            wslice = slice(x * width + padding, (x + 1) * width)
            if im_mode == "rgb":
                grid[:, hslice, wslice] = imgs[k]
            elif im_mode == "gray":
                grid[hslice, wslice] = imgs[k]
            k += 1

    return grid


def get_colors(cmap_name="viridis", N=100):
    the_cmap = plt.get_cmap(cmap_name)
    colors = [the_cmap(i) for i in np.linspace(0, 1, N)]
    return colors


def scatter_linear_colors(data, colors=None, reverse=False, **kwargs):
    assert isinstance(data, np.ndarray) and data.ndim == 2

    if colors is None:
        colors = get_colors(cmap_name="viridis", N=len(data))
    elif isinstance(colors, str):
        colors = get_colors(cmap_name=colors, N=len(data))

    if reverse:
        colors = reversed(colors)

    for cnt, color in enumerate(colors):
        plt.scatter(data[cnt, 0], data[cnt, 1], color=color, **kwargs)


def clip_on_imagespace(
    inp, mean=PYTORCH_IMAGENET_MEAN, std=PYTORCH_IMAGENET_STD, image_min=0, image_max=1
):
    assert isinstance(inp, (np.ndarray, torch.Tensor))

    is_tensor = False
    if isinstance(inp, torch.Tensor):
        if len(inp.shape) == 4:
            out = clip_on_imagespace4d_tensor(
                inp, mean=mean, std=std, image_min=image_min, image_max=image_max
            )
            return out
        _inp = inp.to("cpu").detach().numpy()
        is_tensor = True
    else:
        _inp = inp.copy()

    if len(_inp.shape) == 3:
        out = clip_on_imagespace3d(
            _inp, mean=mean, std=std, image_min=image_min, image_max=image_max
        )
    elif len(_inp.shape) == 4:
        out = clip_on_imagespace4d(
            _inp, mean=mean, std=std, image_min=image_min, image_max=image_max
        )

    if is_tensor:
        with torch.no_grad():
            inp.data = torch.as_tensor(out, device=inp.device)
        return inp
    return out


def clip_on_imagespace4d_tensor(
    inp, mean=PYTORCH_IMAGENET_MEAN, std=PYTORCH_IMAGENET_STD, image_min=0, image_max=1
):
    assert isinstance(inp, (torch.Tensor))
    assert inp.shape[1] == 3, "image format should be NCHW"

    if isinstance(mean, (list, tuple)):
        mean = torch.Tensor(mean)

    if isinstance(std, (list, tuple)):
        std = torch.Tensor(std)

    with torch.no_grad():
        vmax = torch.ones_like(mean) * image_max
        vmin = torch.ones_like(mean) * image_min

        input_max = (vmax - mean) / std
        input_min = (vmin - mean) / std

        for i_dim in range(len(mean)):
            inp[:, i_dim] = torch.clamp(
                inp[:, i_dim], input_min[i_dim], input_max[i_dim]
            )

    return inp


def clip_on_imagespace4d(
    inp, mean=PYTORCH_IMAGENET_MEAN, std=PYTORCH_IMAGENET_STD, image_min=0, image_max=1
):
    assert isinstance(inp, (np.ndarray))
    assert inp.ndim == 4
    assert inp.shape[1] == 3, "image format should be NCHW"

    _inp = inp.copy()

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)

    if isinstance(std, (list, tuple)):
        std = np.array(std)

    vmax = np.ones_like(mean) * image_max
    vmin = np.ones_like(mean) * image_min

    input_max = (vmax - mean) / std
    input_min = (vmin - mean) / std

    for i_dim in range(len(mean)):
        _inp[:, i_dim] = np.clip(inp[:, i_dim], input_min[i_dim], input_max[i_dim])

    return _inp


def clip_on_imagespace3d(
    inp, mean=PYTORCH_IMAGENET_MEAN, std=PYTORCH_IMAGENET_STD, image_min=0, image_max=1
):
    assert isinstance(inp, (np.ndarray))
    assert inp.ndim == 3
    assert inp.shape[0] == 3, "image format should be CHW"

    _inp = inp.copy()

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)

    if isinstance(std, (list, tuple)):
        std = np.array(std)

    vmax = np.ones_like(mean) * image_max
    vmin = np.ones_like(mean) * image_min

    input_max = (vmax - mean) / std
    input_min = (vmin - mean) / std

    for i_dim in range(len(mean)):
        _inp[i_dim] = np.clip(inp[i_dim], input_min[i_dim], input_max[i_dim])

    return _inp

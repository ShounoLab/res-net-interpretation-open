import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .make_image_from_vis_data import norm_img
from .make_vis_data import get_model
from .utils import colors


def make_image_all_filter(
    weight,
    out_dir=".",
    normalize="image",
    colormode="RGB",
    save_flag=True,
    out_name="all_n_fmap",
    save_format="png",
    each_title=None,
):
    """
    plot or show weight image of filter.

    parameters:
        weight:
            torch.Tensor or numpy.ndarray
        normalize:
            'image' or 'channel'
            if 'image' then normalize over whale weights.
            if 'channel' then normailze each weight.
        colormode:
            RGB or YUV
        save_flag:
            if True then output images to out_dir else show images not output
    """
    if isinstance(weight, torch.Tensor):
        weight = weight.numpy()

    assert isinstance(weight, np.ndarray)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    nrow = int(np.ceil(np.sqrt(len(weight))))
    ncol = int(np.ceil(np.sqrt(len(weight))))
    scale = nrow / 8
    plt.figure(figsize=(10 * scale, 10 * scale))
    for i, w in enumerate(weight):
        plt.subplot(nrow, ncol, i + 1)
        img = np.transpose(w, (1, 2, 0))
        if normalize == "image":
            img = (img - weight.min()) / (weight.max() - weight.min())
        elif normalize == "channel":
            img = norm_img(img)
        if each_title is not None:
            if each_title == "number":
                plt.title(each_title)
            elif callable(each_title):
                plt.title(each_title(i, w))
            elif isinstance(each_title, str):
                plt.title(each_title)
            else:
                raise ValueError(each_title)

        plt.imshow(img)
        plt.axis("off")
    path = os.path.join(out_dir, "{}.{}".format(out_name, save_format))
    plt.tight_layout()
    if save_flag:
        plt.savefig(path, transparent=True)
    else:
        plt.show()
    plt.close()


def make_image_filter(
    weight, out_dir=".", normalize="image", colormode="RGB", title=False, save_flag=True
):
    """
    plot or show each weight image of filter.

    parameters:
        weight:
            torch.Tensor or numpy.ndarray
        normalize:
            'image' or 'channel'
            if 'image' then normalize over whale weights.
            if 'channel' then normailze each weight.
        colormode:
            RGB or YUV
        save_flag:
            if True then output images to out_dir else show images not output
    """
    if isinstance(weight, torch.Tensor):
        weight = weight.numpy()

    assert isinstance(weight, np.ndarray)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if weight.ndim == 4:
        # (out channel, in channel, height, width)

        for i_fmap, w in enumerate(weight):
            plt.figure()
            img = np.transpose(w, (1, 2, 0))
            if normalize == "image":
                img = (img - weight.min()) / (weight.max() - weight.min())
            elif normalize == "channel":
                img = norm_img(img)
            plt.imshow(img)
            path = os.path.join(out_dir, "n_fmap-{}.png".format(i_fmap))
            if save_flag:
                plt.savefig(path, transparent=True)
            else:
                plt.show()
            plt.close()

            if colormode is None:
                continue
            if colormode == "RGB":
                channel_name = ["R", "G", "B"]
            elif colormode == "YCbCr" or colormode == "YUV":
                channel_name = ["Y", "Cb", "Cr"]
            else:
                raise ValueError("{}".format(colormode))

            nrow = len(colormode) + 1
            ncol = 1
            scale = 3
            plt.figure(figsize=(scale * ncol, scale * nrow))

            plt.subplot(nrow, ncol, 1)
            plt.imshow(img)
            for i in range(nrow - 1):
                plt.subplot(nrow, ncol, i + 2)
                # tmp_img = norm_img(img.copy())
                tmp_img = img.copy()
                if colormode == "RGB":
                    if i == 0:
                        tmp_img[:, :, 1] = 0
                        tmp_img[:, :, 2] = 0
                    elif i == 1:
                        tmp_img[:, :, 0] = 0
                        tmp_img[:, :, 2] = 0
                    elif i == 2:
                        tmp_img[:, :, 0] = 0
                        tmp_img[:, :, 1] = 0
                elif colormode == "YCbCr" or colormode == "YUV":
                    rgb = [
                        tmp_img[:, :, 0].reshape(-1),
                        tmp_img[:, :, 1].reshape(-1),
                        tmp_img[:, :, 2].reshape(-1),
                    ]
                    yuv = colors.rgb2yuv(rgb[0], rgb[1], rgb[2])
                    # print(yuv[0, :].max(), yuv[1, :].max(), yuv[2, :].max())
                    # print(yuv[0, :].min(), yuv[1, :].min(), yuv[2, :].min())
                    if i == 0:
                        yuv[1, :] = 0
                        yuv[2, :] = 0
                    elif i == 1:
                        yuv[0, :] = 0
                        yuv[2, :] = 0
                    elif i == 2:
                        yuv[0, :] = 0
                        yuv[1, :] = 0
                    rgb = colors.yuv2rgb(yuv[0], yuv[1], yuv[2])
                    if i == 1 or i == 2:
                        if rgb.min() < 0:
                            rgb -= rgb.min()
                        if rgb.max() > 1:
                            rgb /= rgb.max()

                    tmp_img = np.transpose(
                        rgb.reshape((3,) + tmp_img.shape[:-1]), (1, 2, 0)
                    )

                mean = tmp_img.mean()

                plt.imshow(tmp_img)
                if title:
                    plt.title(
                        "mean:{:.3f}, \n{:.4f}, \n{:.4f}".format(
                            mean, tmp_img.max(), tmp_img.min()
                        )
                    )
            plt.tight_layout()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            path = os.path.join(
                out_dir, "n_fmap-{}_{}.png".format(i_fmap, "".join(channel_name))
            )
            if save_flag:
                plt.savefig(path, transparent=True)
            else:
                plt.show()
            plt.close()


def main(arch, out, normalize="channel", colormode="RGB"):

    print("get model: {}".format(arch))
    model = get_model(arch)
    if not hasattr(model, "conv1"):
        raise ValueError("Unkonw first layer of {}".format(arch))

    weight = None
    for name, param in model.conv1.named_parameters():
        if name == "weight":
            weight = param

    out_dir = os.path.join(out, "filters")
    print("make image filter")
    make_image_filter(
        weight.detach().numpy(), out_dir, normalize=normalize, colormode=colormode
    )

    make_image_all_filter(
        weight.detach().numpy(), out, normalize=normalize, colormode=colormode
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate receptive field of max activation neuron"
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="output directory name"
    )
    parser.add_argument(
        "-a", "--arch", type=str, default="resnet34-pytorch", help="model architecture"
    )
    parser.add_argument(
        "-n", "--normalize", type=str, default="channel", help="normalize mode"
    )
    parser.add_argument(
        "-c", "--colormode", type=str, default="RGB", help="select color mode"
    )
    args = parser.parse_args()

    main(
        arch=args.arch, out=args.out, normalize=args.normalize, colormode=args.colormode
    )

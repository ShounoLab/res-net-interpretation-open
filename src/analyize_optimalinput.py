import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from my_model import get_receptive_field
from utils import config, plots, receptive_field
from utils.analysis import exts
from utils.plots import clip_on_imagespace, input2image
from utils.receptive_field import get_downconv
from utils.receptive_field_tracker import RFTracker

if __name__ == "__main__":
    import argparse

    from utils.load_model import get_model

    parser = argparse.ArgumentParser(description="Generate mean receptive field")
    parser.add_argument(
        "-l",
        "--layer-name",
        type=str,
        default="layer1.0.conv1",
        help="layer name of feature map defined by user",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="analysis/optimal",
        help="output directory name",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=256, help="mini batch size"
    )
    parser.add_argument(
        "-m", "--max-iter", type=int, default=50, help="max itertion number"
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="resnet34",
        help="model architecture. using get_model()",
    )
    parser.add_argument("--device", type=str, default="cpu", help="model architecture")
    parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--mode",
        choices=["neuron", "layer", "other"],
        default="neuron",
        help="available choices",
    )
    parser.add_argument(
        "--random",
        choices=["normal", "uniform"],
        default="noraml",
        help="available random choices",
    )
    parser.add_argument(
        "--optim",
        choices=["adam", "lbfgs"],
        default="adam",
        help="available optimizers",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--zero-start", action="store_true", help="initial ")
    parser.add_argument("--channel-wise", action="store_true", help="initial ")
    parser.add_argument(
        "--off-sorted-channel",
        dest="is_sorted_channel",
        action="store_false",
        help="initial ",
    )
    parser.add_argument("--clip-inputspace", action="store_true", help="initial ")
    args = parser.parse_args()

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    mode = args.mode
    layer_name = args.layer_name
    batch_size = args.batch_size
    max_iter = args.max_iter
    lr = args.lr
    wd = args.wd
    device = args.device
    random_type = args.random
    channel_wise = args.channel_wise
    zero_start = args.zero_start
    optim_type = args.optim.lower()
    is_sorted_channel = args.is_sorted_channel
    clip_inputspace = args.clip_inputspace
    model = get_model(args.arch)
    model = model.to(device)
    model = model.eval()

    for m in model.parameters():
        m.require_grad = False

    tracker = RFTracker(model)
    img = torch.randn(1, 3, 224, 224, device=device)
    out = model(img, layers=[layer_name])[layer_name]
    # check arch is path
    if os.path.exists(args.arch):
        dname = os.path.dirname(args.arch)
        arch_name = os.path.basename(dname)
    else:
        arch_name = args.arch
    out_path = "{}_layer-{}".format(arch_name, args.layer_name)
    # path = '{}_layer-{}'.format(args.arch, args.layer_name)
    print(out_path)
    final_rf = tracker.rf_pool[layer_name]
    _, _, h, w = out.shape
    center, rf = get_receptive_field((h // 2, w // 2), final_rf)
    hs, ws = receptive_field.get_rf_region(center, rf)
    tracker.remove_hook()
    fmap_shape = out.shape

    channels = fmap_shape[1]
    # lr = channels * lr
    # wd = channels * wd

    def make_np_images():
        np_images = np.zeros((channels, 3, 224, 224))
        for b in range(0, channels, batch_size):
            print(b)
            bd = len(np_images[b : b + batch_size])
            if zero_start:
                images = torch.zeros(bd, 3, 224, 224, requires_grad=True, device=device)
            else:
                if random_type == "normal":
                    images = torch.randn(
                        bd, 3, 224, 224, requires_grad=True, device=device
                    )
                elif random_type == "uniform":
                    images = torch.empty(
                        bd, 3, 224, 224, requires_grad=True, device=device
                    )
                    torch.nn.init.uniform_(images, -1, 1)
                else:
                    raise ValueError(random_type)

            if optim_type == "adam":
                optimizer = optim.Adam([images], lr=lr, weight_decay=wd)
                for _ in tqdm(range(max_iter), total=max_iter):
                    optimizer.zero_grad()
                    out = model.layer_forward(images, layers=[layer_name])[layer_name]
                    _, c, h, w = out.shape
                    n_index = torch.arange(bd, dtype=torch.long, device=device)
                    ch_index = torch.arange(b, b + bd, dtype=torch.long, device=device)
                    if mode == "neuron":
                        o = out[n_index, ch_index, h // 2, w // 2]
                        loss = -torch.sum(o) / len(images)
                        loss.backward()
                    elif mode == "layer":
                        o = out[n_index, ch_index] / (h * w)
                        loss = -torch.sum(o) / len(images)
                        loss.backward()
                    else:
                        o = out[n_index, ch_index, h // 2, w // 2]
                        with torch.no_grad():
                            tensor = -0.01 * torch.ones_like(o)
                        with torch.no_grad():
                            loss = -torch.sum(o) / len(images)
                        o.backward(gradient=tensor)
                    optimizer.step()
                    if clip_inputspace:
                        images = clip_on_imagespace(images)
            elif optim_type == "lbfgs":
                optimizer = optim.LBFGS([images], lr=lr, max_iter=5)

                def closure():
                    out = model.layer_forward(images, layers=[layer_name])[layer_name]
                    _, c, h, w = out.shape
                    n_index = torch.arange(bd, dtype=torch.long, device=device)
                    ch_index = torch.arange(b, b + bd, dtype=torch.long, device=device)
                    if mode == "neuron":
                        o = out[n_index, ch_index, h // 2, w // 2]
                        loss = -torch.sum(o) / len(images) + wd * torch.mean(
                            torch.pow(images, 2)
                        )
                    elif mode == "layer":
                        o = out[n_index, ch_index] / (h * w)
                        loss = -torch.sum(o) / len(images) + wd * torch.mean(
                            torch.pow(images, 2)
                        )
                    loss.backward()
                    return loss

                for _ in tqdm(range(max_iter), total=max_iter):
                    optimizer.zero_grad()
                    optimizer.step(closure)
                    if clip_inputspace:
                        images = clip_on_imagespace(
                            images, image_min=0.3, image_max=0.7
                        )
            else:
                raise ValueError(optim_type)

            np_images[b : b + bd] = images.to("cpu").detach().numpy()
        if mode in ("neuron", "other"):
            np_images = np_images[:, :, hs[0] : hs[1], ws[0] : ws[1]]
        return np_images

    def make_np_images_channelwise():
        np_len = int(channels * batch_size)
        np_images = np.zeros((np_len, 3, 224, 224))
        for b in tqdm(range(0, np_len, batch_size), total=channels):
            cur_channel = b // batch_size
            if zero_start:
                images = torch.zeros(
                    batch_size, 3, 224, 224, requires_grad=True, device=device
                )
            else:
                if random_type == "normal":
                    images = torch.randn(
                        batch_size, 3, 224, 224, requires_grad=True, device=device
                    )
                elif random_type == "uniform":
                    images = torch.empty(
                        batch_size, 3, 224, 224, requires_grad=True, device=device
                    )
                    torch.nn.init.uniform_(images, -1, 1)
                else:
                    raise ValueError(random_type)

            if optim_type == "adam":
                optimizer = optim.Adam([images], lr=lr, weight_decay=wd)
                for _ in range(max_iter):
                    optimizer.zero_grad()
                    out = model.layer_forward(images, layers=[layer_name])[layer_name]
                    _, c, h, w = out.shape
                    n_index = torch.arange(batch_size, dtype=torch.long, device=device)
                    ch_index = cur_channel * torch.ones(
                        batch_size, dtype=torch.long, device=device
                    )
                    if mode == "neuron":
                        o = out[n_index, ch_index, h // 2, w // 2]
                        loss = -torch.sum(o) / len(images)
                        loss.backward()
                    elif mode == "layer":
                        o = out[n_index, ch_index] / (h * w)
                        loss = -torch.sum(o) / len(images)
                        loss.backward()
                    else:
                        o = out[n_index, ch_index, h // 2, w // 2]
                        with torch.no_grad():
                            tensor = -(channels / 64) * torch.ones_like(o)
                        with torch.no_grad():
                            loss = -torch.sum(o) / len(images)
                        o.backward(gradient=tensor)
                    optimizer.step()
                    if clip_inputspace:
                        images = clip_on_imagespace(images)
            elif optim_type == "lbfgs":
                optimizer = optim.LBFGS([images], lr=lr, max_iter=5)

                def closure():
                    out = model.layer_forward(images, layers=[layer_name])[layer_name]
                    _, c, h, w = out.shape
                    n_index = torch.arange(batch_size, dtype=torch.long, device=device)
                    ch_index = cur_channel * torch.ones(
                        batch_size, dtype=torch.long, device=device
                    )
                    if mode == "neuron":
                        o = out[n_index, ch_index, h // 2, w // 2]
                        loss = -torch.sum(o) / len(images) + wd * torch.mean(
                            torch.pow(images, 2)
                        )
                    elif mode == "layer":
                        o = out[n_index, ch_index] / (h * w)
                        loss = -torch.sum(o) / len(images) + wd * torch.mean(
                            torch.pow(images, 2)
                        )
                    loss.backward()
                    return loss

                for _ in range(max_iter):
                    optimizer.zero_grad()
                    optimizer.step(closure)
                    if clip_inputspace:
                        images = clip_on_imagespace(images)
            else:
                raise ValueError(optim_type)

            # print("loss: {}".format(loss.item()))
            np_images[b : b + batch_size] = images.to("cpu").detach().numpy()
        if mode in ("neuron", "other"):
            np_images = np_images[:, :, hs[0] : hs[1], ws[0] : ws[1]]
        return np_images

    if channel_wise:
        np_images = make_np_images_channelwise()
    else:
        np_images = make_np_images()

    np_down_weight = None
    down_weights = []
    for name, weight in model.named_parameters():
        if "downsample.0" in name:
            down_weights.append(weight.detach().to("cpu").numpy())

        if layer_name + ".weight" == name:
            np_down_weight = weight.detach().to("cpu").numpy()

    # model is of ResNets
    np_conv = model.conv1.weight.detach().to("cpu").numpy()
    if np_down_weight is not None and "plain" not in args.arch:
        # only ResNet
        for down_weight in down_weights:
            if np_conv.shape[0] == np_down_weight.shape[1]:
                break
            np_conv = get_downconv(np_conv, down_weight)
        np_downconv1 = get_downconv(np_conv, np_down_weight)
    else:
        # ResNet or PlainNet
        if len(down_weights) > 0:
            # only ResNEt
            for down_weight in down_weights:
                if np_conv.shape[0] == fmap_shape[1]:
                    break
                np_conv = get_downconv(np_conv, down_weight)
        np_downconv1 = np_conv

    if np_down_weight is not None and "plain" not in args.arch:
        # sort by down conv weight norm
        sorted_channels = np.argsort(
            np.linalg.norm(np_downconv1.reshape(len(np_downconv1), -1), axis=-1)
        )[::-1]
    else:
        sorted_channels = np.arange(fmap_shape[1])

    if not channel_wise and is_sorted_channel:
        np_images = np_images[sorted_channels]

    dir_path = os.path.join(args.out, out_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if seed is None:
        fname = "args.config"
    else:
        fname = "args.config{:08}".format(seed)
    config_path = os.path.join(dir_path, fname)
    config.save_config_file_from_args(config_path, args)

    if seed is not None:
        out_path = os.path.join(dir_path, "{}-images{:08}".format(args.mode, seed))
    else:
        out_path = os.path.join(dir_path, "{}-images".format(args.mode))
    np.save(out_path, np_images)

    if channel_wise:
        k = 0
        if seed is None:
            out_name = "{}-all-images-{}".format(mode, k)
        else:
            out_name = "{}-all-images-{}-{:08}".format(mode, k, seed)
        _, ch, h_size, w_size = np_images.shape
        imgs = np_images.reshape(channels, batch_size, ch, h_size, w_size)[:, k]

        def normalize_func(x):
            return input2image(x, img_format="HWC")

        plots.plot_imshows(
            imgs,
            normalize=normalize_func,
            title="count",
            exts=exts,
            out_dir=dir_path,
            out_name=out_name,
        )
        if seed is None:
            out_name = "{}-all-images-{}-norm".format(mode, k)
        else:
            out_name = "{}-all-images-{}-norm{:08}".format(mode, k, seed)

        def normalize_func(x):
            img_format = "HWC"
            return input2image(
                plots.normalize_inputspace(x, img_format=img_format),
                img_format=img_format,
            )

        plots.plot_imshows(
            imgs,
            normalize=normalize_func,
            title="count",
            exts=exts,
            out_dir=dir_path,
            out_name=out_name,
        )
    else:
        n = 2
        m = np.ceil(np.sqrt(channels))
        plt.figure(figsize=(n * m, n * m))
        for k in range(channels):
            tmp = input2image(np_images[k])
            plt.subplot(m, m, k + 1)
            plt.title(k)
            plt.imshow(np.transpose(np.clip(tmp, 0, 1), (1, 2, 0)))
            plt.axis("off")
        plt.tight_layout()
        for ext in exts:
            if seed is None:
                img_path = os.path.join(
                    dir_path, "{}-all-images.{}".format(args.mode, ext)
                )
            else:
                img_path = os.path.join(
                    dir_path, "{}-all-images{:08}.{}".format(args.mode, seed, ext)
                )
            plt.savefig(img_path, transparent=True)
        plt.close()

        plt.figure(figsize=(n * m, n * m))
        for k in range(channels):
            tmp = plots.normalize_inputspace(np_images[k])
            tmp = input2image(tmp)
            plt.subplot(m, m, k + 1)
            plt.title(k)
            plt.imshow(np.transpose(np.clip(tmp, 0, 1), (1, 2, 0)))
            plt.axis("off")
        plt.tight_layout()
        for ext in exts:
            if seed is None:
                img_path = os.path.join(
                    dir_path, "{}-all-images-norm.{}".format(args.mode, ext)
                )
            else:
                img_path = os.path.join(
                    dir_path, "{}-all-images-norm{:08}.{}".format(args.mode, seed, ext)
                )
            plt.savefig(img_path, transparent=True)
        plt.close()
    print("saved ==> ", dir_path)

#!/usr/bin/env python
# coding: utf-8
"""
    make datas for receptive field
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm as tqdm

from my_model.blocks import ResNet
from my_model.my_dataset import get_ilsvrc2012
from utils import analysis, config
from utils.load_model import TransferModel
from utils.plots import input2image, norm_img
from utils.receptive_field import cut_rf_from_img_helper, get_downconv
from utils.receptive_field_tracker import get_rf_layer_info

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="make data receptive field")
    parser.add_argument(
        "-l",
        "--layer-name",
        type=str,
        default="layer1.0.conv1",
        help="layer name of feature map defined by user",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="layer name of feature map counting activateions",
    )
    #    parser.add_argument('-d', '--dataset', type=str, default='imagenet',
    #                        help='dataset name')
    parser.add_argument(
        "--val-list", type=str, default=None, help="define list of validation dataset"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="analysis/e_receptive_field/",
        help="output directory name",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=256, help="mini batch size"
    )
    parser.add_argument("-c", "--channel", type=int, default=None, help="channel")
    parser.add_argument(
        "-m", "--max-iter", type=int, default=None, help="max itertion number"
    )
    parser.add_argument(
        "-a", "--arch", type=str, default="resnet34", help="model architecture"
    )
    parser.add_argument(
        "--wandb-flag", action="store_true", help="Is arch wandb directory?"
    )
    parser.add_argument("--device", type=str, default="cuda", help="model architecture")
    parser.add_argument(
        "-w", "--workers", type=int, default=6, help="worker number of using dataloader"
    )
    parser.add_argument("--max-ch-cnt", type=int, default=None, help="max channel size")
    parser.add_argument("--skip-counting", action="store_true", help="skip coutning")
    parser.add_argument("--skip-mean-rf", action="store_true", help="skip mean rf")
    args = parser.parse_args()

    device = args.device
    model = analysis.get_model_from_keywords(args.arch, args.wandb_flag)
    if args.wandb_flag:
        args.layer_name = "classifier." + args.layer_name

    model = model.to(device)
    model.eval()
    layer_name = args.layer_name

    image_size = 224
    rnd_dataset = get_ilsvrc2012(mode="test", val_txt=args.val_list)
    nclasses = 1000
    batch_size = args.batch_size
    workers = args.workers
    data_loader = torch.utils.data.DataLoader(
        rnd_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    img, _ = next(iter(data_loader))
    img = img.to(device)
    layer_info, out = get_rf_layer_info(model, img, layer_name)

    # check arch is path
    arch_name = analysis.get_arch_name(args.arch, args.wandb_flag)

    if args.activation is None:
        activation_name = layer_name
    else:
        activation_name = args.activation

    path = "{}_layer-{}_act-{}".format(arch_name, args.layer_name, activation_name)
    print("save ==> ", path)
    dir_path = os.path.join(args.out, path)

    configname = "config.yaml"
    path = os.path.join(dir_path, configname)
    d = {"script name": "make_rf_datas.py"}
    config.save_config_file_from_args(path, args, add_dict=d)

    fmap_shape = out.shape
    if args.max_ch_cnt is None:
        max_ch_cnt = fmap_shape[1]
    else:
        max_ch_cnt = args.max_ch_cnt

    np_down_weight = None
    down_weights = []
    for name, weight in model.named_parameters():
        if "downsample.0" in name:
            down_weights.append(weight.detach().to("cpu").numpy())

        if layer_name + ".weight" == name:
            np_down_weight = weight.detach().to("cpu").numpy()

    # model is of ResNets
    if isinstance(model, (ResNet,)):
        np_conv = model.conv1.weight.detach().to("cpu").numpy()
        if np_down_weight is not None and "plain" not in arch_name:
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
    elif isinstance(model, TransferModel):
        np_downconv1 = model.feature_extractor.conv1.weight.detach().to("cpu").numpy()
    else:
        raise TypeError(type(model))

    def forward_model(img, layer):
        if isinstance(model, ResNet):
            return model(img, layers=[layer])[layer]
        elif isinstance(model, TransferModel):
            return model(img, layer=layer)
        else:
            raise TypeError(type(model))

    activation_index = [np.array([], dtype=np.int32) for i in range(fmap_shape[1])]
    activation_value = [np.array([], dtype=np.float32) for i in range(fmap_shape[1])]
    if args.skip_counting:
        print("skip counting")
    else:
        print("counting activations ...")
        with torch.no_grad():
            for offset, (img, _) in tqdm(
                enumerate(data_loader), total=len(data_loader)
            ):
                img = img.to(device)
                #                 y = model(img, layers=[activation_name])
                #                 out = y[activation_name]
                out = forward_model(img, activation_name)
                out = out.transpose(1, 0)
                out = out.reshape(len(out), -1)
                n_out, dim = out.shape
                tmp = torch.repeat_interleave(
                    torch.arange(dim).to("cpu").reshape(1, -1), n_out, dim=0
                )
                for ch in range(n_out):
                    idx_ = tmp[ch][out[ch] > 0].to("cpu").numpy()
                    if len(idx_) == 0:
                        continue
                    out_ = out[ch][out[ch] > 0].to("cpu").numpy()
                    activation_index[ch] = np.append(
                        activation_index[ch],
                        idx_ + batch_size * offset * fmap_shape[-2] * fmap_shape[-1],
                    )
                    activation_value[ch] = np.append(activation_value[ch], out_)
        print("finished")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if np_down_weight is not None and "plain" not in args.arch:
        print("sort channnel by down layer")
        # sort by down conv weight norm
        sorted_channels = np.argsort(
            np.linalg.norm(np_downconv1.reshape(len(np_downconv1), -1), axis=-1)
        )[::-1]
    else:
        sorted_channels = np.arange(fmap_shape[1])

    if args.max_iter is None:
        max_iter = 256 * 10
    else:
        max_iter = args.max_iter

    # with open(os.path.join(dir_path, 'top_activation_value.pkl'), 'wb') as f:
    #     pickle.dump(activation_value, f)
    # with open(os.path.join(dir_path, 'top_activation_index.pkl'), 'wb') as f:
    #     pickle.dump(activation_index, f)

    rate = 0
    tmp_path = os.path.join(dir_path, "top_activation")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    rf_datas_path = os.path.join(dir_path, "top_rf_datas")
    if not os.path.exists(rf_datas_path):
        os.makedirs(rf_datas_path)
    for ch_cnt, ch in tqdm(enumerate(sorted_channels), total=len(sorted_channels)):
        if ch >= max_ch_cnt:
            break
        if args.channel is not None and ch != args.channel:
            continue

        np_xs = np.array([], dtype=np.int32)
        np_ys = np.array([], dtype=np.int32)
        np_img_idx = np.array([], dtype=np.int32)
        cut_grads = []
        cut_imgs = []

        total_value = 0
        channel_pkl_path = os.path.join(tmp_path, "channel-{:03}.pkl".format(ch))

        # print('count: ', ch_cnt)
        if args.skip_counting:
            with open(channel_pkl_path, "rb") as f:
                tmp_d = pickle.load(f)
            sorted_value = tmp_d["sorted_value"]
            sorted_index = tmp_d["sorted_index"]
            top_act_index = tmp_d["top_act_index"]
        else:
            sorted_value = np.sort(activation_value[ch])[::-1]
            sorted_index = np.argsort(activation_value[ch])[::-1]
            top_act_index = activation_index[ch][sorted_index[: max_iter + 1]]
            save_dict = {
                "sorted_value": sorted_value[: max_iter + 1],
                "sorted_index": sorted_index[: max_iter + 1],
                "top_act_index": top_act_index,
            }
            with open(channel_pkl_path, "wb") as f:
                pickle.dump(save_dict, f)

        # print('channel: {}, size: {}'.format(ch, len(sorted_index)))
        total_iter = len(sorted_index) if len(sorted_index) < max_iter else max_iter
        ori_cnt_map = torch.ones(image_size, image_size, dtype=torch.int32)
        pad_size = image_size if layer_info[2] > image_size else layer_info[2]
        cnt_map = np.zeros((pad_size, pad_size), dtype=np.int32)
        # save top rf image and grad
        # for cnt, idx in tqdm(enumerate(sorted_index), total=total_iter):
        for cnt, idx in enumerate(sorted_index):
            if sorted_value[cnt] < rate:
                break
            if max_iter > 0 and cnt > max_iter:
                break
            _, _, n_x, n_y = fmap_shape
            tmp = np.asarray(
                np.unravel_index(top_act_index[cnt], (len(rnd_dataset), n_x, n_y))
            )
            img_idx, x, y = tmp

            images = rnd_dataset[img_idx][0][None, ...]
            images = images.to(device)
            images = images.requires_grad_()
            model.zero_grad()
            #             out = model(images, layers=[activation_name])[activation_name]
            out = forward_model(images, activation_name)
            out[0, ch, x, y].backward(
                torch.Tensor(
                    [
                        1,
                    ]
                ).to(device)
            )
            grad = images.grad[0].detach().clone().to("cpu")

            cut_grad = cut_rf_from_img_helper(
                grad, (x, y), layer_info, is_numpy=True, pad_size=pad_size
            )
            image = images[0].detach().to("cpu")
            cut_img = cut_rf_from_img_helper(
                image, (x, y), layer_info, is_numpy=True, pad_size=pad_size
            )
            cut_grads.append(cut_grad)
            cut_imgs.append(cut_img)

            cnt_map += cut_rf_from_img_helper(
                ori_cnt_map, (x, y), layer_info, is_numpy=True, pad_size=pad_size
            )

        # save rf images and grads
        cut_grads = np.asarray(cut_grads)
        cut_imgs = np.asarray(cut_imgs)

        rfimgs_path = os.path.join(rf_datas_path, "rfimgs-{:03}".format(ch))
        np.save(rfimgs_path, cut_imgs)
        rfgrads_path = os.path.join(rf_datas_path, "rfgrads-{:03}".format(ch))
        np.save(rfgrads_path, cut_grads)
        rfcntmap_path = os.path.join(rf_datas_path, "rfcntmap-{:03}".format(ch))
        np.save(rfcntmap_path, cnt_map)

    save_dict = {
        "downconv": np_downconv1,
        "layer_info": layer_info,
        "image_size": image_size,
        "sorted_channels": sorted_channels,
        "fmap_shape": fmap_shape,
        "len_dataset": len(rnd_dataset),
    }
    with open(os.path.join(dir_path, "config.pkl"), "wb") as f:
        pickle.dump(save_dict, f)

    img_path = os.path.join(dir_path, "imgs")
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    print("save images ==> {}".format(img_path))
    for i, ch_idx in enumerate(sorted_channels):
        if i >= max_ch_cnt:
            break
        if args.channel is not None and ch != args.channel:
            continue
        n = 3
        plt.figure(figsize=(n, n))
        if len(np_downconv1) > ch_idx:
            tmp = np_downconv1[ch_idx].copy()
            tmp -= np_downconv1[ch_idx].min()
            tmp /= tmp.max()
            tmp_filter = tmp.copy()
        else:
            tmp_filter = np.zeros_like(np_downconv1[0])
        plt.figure(figsize=(n, n))
        plt.title("virtual filter {}".format(ch_idx))
        plt.imshow(np.transpose(tmp_filter, (1, 2, 0)))
        path = os.path.join(img_path, "vf-ch{:03}.png".format(ch_idx))
        plt.savefig(path, transparent=True)
        plt.close()

        plt.figure(figsize=(n, n))
        path = os.path.join(rf_datas_path, "rfcntmap-{:03}.npy".format(ch_idx))
        cnt_map = np.load(path)
        cnt_map[cnt_map == 0] = 1

        # mean RF image
        path = os.path.join(rf_datas_path, "rfimgs-{:03}.npy".format(ch_idx))
        rfimgs = np.load(path)
        tmp_img = np.sum(rfimgs / cnt_map, 0)
        plt.figure(figsize=(n, n))
        plt.title("mean rf image")
        plt.imshow(np.transpose(input2image(tmp_img), (1, 2, 0)))
        path = os.path.join(img_path, "mean-image-ch{:03}.png".format(ch_idx))
        plt.savefig(path, transparent=True)
        plt.close()

        tmp_img = np.sum(rfimgs / cnt_map, 0)
        plt.figure(figsize=(n, n))
        plt.title("normalized mean rf image")
        plt.imshow(np.transpose(norm_img(tmp_img), (1, 2, 0)))
        path = os.path.join(img_path, "mean-image-norm-ch{:03}.png".format(ch_idx))
        plt.savefig(path, transparent=True)
        plt.close()

        # mean RF grad
        path = os.path.join(rf_datas_path, "rfgrads-{:03}.npy".format(ch_idx))
        rfgrads = np.load(path)
        tmp_img = np.sum(rfgrads / cnt_map, 0)
        plt.figure(figsize=(n, n))
        plt.title("mean rf grads")
        plt.imshow(np.transpose(input2image(tmp_img), (1, 2, 0)))
        path = os.path.join(img_path, "mean-grad-ch{:03}.png".format(ch_idx))
        plt.savefig(path, transparent=True)
        plt.close()

        plt.figure(figsize=(n, n))
        plt.title("normalized mean rf grads")
        plt.imshow(np.transpose(norm_img(tmp_img), (1, 2, 0)))
        path = os.path.join(img_path, "mean-grad-norm-ch{:03}.png".format(ch_idx))
        plt.savefig(path, transparent=True)
        plt.close()

        # mean effective RF image
        tmp_img = np.sum(np.abs(rfgrads) * rfimgs / cnt_map, 0)
        plt.figure(figsize=(n, n))
        plt.title("mean erf image")
        plt.imshow(np.transpose(input2image(tmp_img), (1, 2, 0)))
        path = os.path.join(img_path, "mean-erfimage-ch{:03}.png".format(ch_idx))
        plt.savefig(path, transparent=True)
        plt.close()

        plt.figure(figsize=(n, n))
        plt.title("normalized mean erf image")
        plt.imshow(np.transpose(norm_img(tmp_img), (1, 2, 0)))
        path = os.path.join(img_path, "mean-erfimage-norm-ch{:03}.png".format(ch_idx))
        plt.savefig(path, transparent=True)
        plt.close()

        # mean enhanced RF image
        tmp_img = np.sum((rfgrads + rfimgs) / cnt_map, 0)
        plt.figure(figsize=(n, n))
        plt.title("mean enhanced rf image")
        plt.imshow(np.transpose(input2image(tmp_img), (1, 2, 0)))
        path = os.path.join(
            img_path, "mean-enhanced-rfimage-ch{:03}.png".format(ch_idx)
        )
        plt.savefig(path, transparent=True)
        plt.close()

        plt.figure(figsize=(n, n))
        plt.title("normalized mean enchanced rf image")
        plt.imshow(np.transpose(norm_img(tmp_img), (1, 2, 0)))
        path = os.path.join(
            img_path, "mean-enhanced-rfimage-norm-ch{:03}.png".format(ch_idx)
        )
        plt.savefig(path, transparent=True)
        plt.close()

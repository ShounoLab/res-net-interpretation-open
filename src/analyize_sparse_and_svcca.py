import itertools
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import svcca
import torch
from tqdm import tqdm as tqdm

from my_model import get_ilsvrc2012
# from utils.analysis import save_arr
from utils.analysis import (labels_plainnets, labels_resnets, make_dir,
                            plainnet34_model_keys, resnet34_model_keys)
from utils.load_model import get_model
from utils.tensortracker import TensorTracker


def main():
    dataset = get_ilsvrc2012(mode="test")
    targets = np.asarray(dataset.targets)
    out_dir = "sparse_and_svcca/imgs"
    device = "cuda"

    resnet_keys = resnet34_model_keys
    plainnet_keys = plainnet34_model_keys
    model_keys = resnet_keys + plainnet_keys

    candidate_layers = [
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

    def get_fmap_pools(images):
        fmap_pools = {}
        for name in model_keys:
            model = get_model(name)
            model = model.eval()
            model = model.to(device)
            tracker = TensorTracker(model, candidate_layers=candidate_layers)
            with torch.no_grad():
                model(images.to(device))
            fmap_pools[name] = deepcopy(tracker.fmap_pool)
            tracker.remove_hook()
        return fmap_pools

    def counting_non_zero_data(keys, fmap_pools, batch_size, channel_wise=False):
        count_non_zero_data = []
        for key in keys:
            tmp_data = []
            for layer_name, fmap in fmap_pools[key].items():
                if channel_wise:
                    tmp_data1 = np.count_nonzero(
                        fmap.numpy()[0].reshape(fmap.shape[1], -1), axis=1
                    )
                    tmp_data1 = tmp_data1 / np.prod(fmap.shape[2:])
                else:
                    tmp_data1 = np.count_nonzero(fmap.numpy().reshape(batch_size, -1))
                    tmp_data1 = tmp_data1 / np.prod(fmap.shape)
                tmp_data.append(tmp_data1)
            tmp_data = np.asarray(tmp_data)
            count_non_zero_data.append(tmp_data)
        return np.asarray(count_non_zero_data)

    def plot_helper(
        data,
        index=None,
        labels=None,
        colors=None,
        fillx=None,
        candidate_layers=candidate_layers,
        out_name=None,
    ):
        alpha = 0.7
        linestyles = ["solid", "dashed", "dotted", "dashdot"]
        # cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        # cmap = plt.get_cmap(cmaps[4])
        plt.figure(figsize=(14, 7))
        if index is None:
            index = np.arange(data.shape[1])
        if fillx is not None:
            assert hasattr(fillx, "__iter__")
            _cmap = plt.get_cmap("rainbow")
            _colors = [_cmap(i) for i in np.linspace(0, 1, len(fillx))]
            for cnt, x in enumerate(fillx):
                color = _colors[cnt]
                plt.fill_between(x, 0, 1, color=color, alpha=0.1)

        for cnt, d in enumerate(data):
            x = np.arange(len(index))
            if labels is None:
                name = ""
            else:
                name = labels[cnt]
            if colors is None:
                color = "k"
            else:
                color = colors[cnt]

            plt.plot(
                x,
                d[index],
                linestyle=linestyles[cnt % len(linestyles)],
                color=color,
                alpha=alpha,
                label=name,
            )

        if candidate_layers is not None:
            plt.xticks(x, np.asarray(candidate_layers)[index], rotation=90)
        plt.ylabel("Activation sparse rate")
        plt.ylim(0, 1)
        plt.grid()
        plt.legend()
        if out_name is not None:
            make_dir(out_dir)
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, transparent=True)
        else:
            plt.show()
        plt.close("all")

    def sparse_process(
        count_non_zero_res_data,
        count_non_zero_plain_data,
        candidate_layers,
        resnet_keys,
        plainnet_keys,
        out_prefix=None,
    ):
        fill_sepkeys = ["layer1", "layer2", "layer3", "layer4"]
        fill_index = []
        for key in fill_sepkeys:
            tmp = []
            for i, cl in enumerate(candidate_layers):
                if key in cl:
                    tmp.append(i)
            fill_index.append(np.asarray(tmp))

        if out_prefix is not None:
            out_name = "{}-{}.png".format(out_prefix, "resnets")
        else:
            out_name = None
        plot_helper(
            1 - count_non_zero_res_data,
            labels=labels_resnets,
            fillx=fill_index,
            out_name=out_name,
        )
        if out_prefix is not None:
            out_name = "{}-{}.png".format(out_prefix, "plainnets")
        else:
            out_name = None
        plot_helper(
            1 - count_non_zero_plain_data,
            labels=labels_plainnets,
            fillx=fill_index,
            out_name=out_name,
        )

    random_seed = 1210
    random_batch_size = 256
    focus_class = 0

    print("analysis sparse")

    np.random.seed(random_seed)
    batch_size = random_batch_size
    perm = np.random.permutation(len(targets))[:batch_size]
    images = torch.stack([dataset[i][0] for i in perm])

    fmap_pools = get_fmap_pools(images)
    count_non_zero_res_data = counting_non_zero_data(
        resnet_keys, fmap_pools, batch_size
    )
    count_non_zero_plain_data = counting_non_zero_data(
        plainnet_keys, fmap_pools, batch_size
    )
    out_prefix = "random"
    sparse_process(
        count_non_zero_res_data,
        count_non_zero_plain_data,
        candidate_layers,
        resnet_keys,
        plainnet_keys,
        out_prefix=out_prefix,
    )

    focus_images = torch.stack(
        [dataset[i][0] for i in np.where(targets == focus_class)[0]]
    )
    batch_size = len(focus_images)

    fmap_pools = get_fmap_pools(focus_images.clone())
    count_non_zero_res_data = counting_non_zero_data(
        resnet_keys, fmap_pools, batch_size
    )
    count_non_zero_plain_data = counting_non_zero_data(
        plainnet_keys, fmap_pools, batch_size
    )
    out_prefix = "focusclass{:03}".format(focus_class)
    sparse_process(
        count_non_zero_res_data,
        count_non_zero_plain_data,
        candidate_layers,
        resnet_keys,
        plainnet_keys,
        out_prefix=out_prefix,
    )

    batch_size = random_batch_size
    fmap_pools = get_fmap_pools(images)
    count_non_zero_res_data = counting_non_zero_data(
        resnet_keys, fmap_pools, batch_size, channel_wise=True
    )
    count_non_zero_plain_data = counting_non_zero_data(
        plainnet_keys, fmap_pools, batch_size, channel_wise=True
    )

    for i in range(len(candidate_layers)):
        tmp_array = np.stack(count_non_zero_res_data[:, i])
        tmp_array1 = np.stack(count_non_zero_plain_data[:, i])
        tmp_array = np.concatenate((tmp_array, tmp_array1))
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, tmp_array.shape[0] + 1)]
        tmp_array = np.sort(1 - tmp_array, -1)
        out_name = "random-{}.png".format(candidate_layers[i])
        plot_helper(
            tmp_array,
            candidate_layers=None,
            colors=colors,
            labels=labels_resnets + labels_plainnets,
            out_name=out_name,
        )

    print("analysis svcca")
    out_prefix = "svcca"
    fmap_pools = get_fmap_pools(images)
    svcca_process(
        fmap_pools,
        candidate_layers,
        resnet_keys,
        plainnet_keys,
        out_dir=out_dir,
        out_prefix=out_prefix,
    )


def svcca_process(
    fmap_pools,
    candidate_layers,
    resnet_keys,
    plainnet_keys,
    out_dir="out",
    out_prefix=None,
):
    def calc_mean_svcca(model_1, model_2, epsilon=0):
        table = np.zeros(len(candidate_layers))
        for cnt, layer in tqdm(
            enumerate(candidate_layers), total=len(candidate_layers)
        ):
            act1 = fmap_pools[model_1][layer].numpy()
            act1 = np.transpose(act1, (1, 0, 2, 3))
            act1 = act1.reshape(len(act1), -1).astype(np.float64)

            act2 = fmap_pools[model_2][layer].numpy()
            act2 = np.transpose(act2, (1, 0, 2, 3))
            act2 = act2.reshape(len(act2), -1).astype(np.float64)
            try:
                result = svcca.cca_core.get_cca_similarity(
                    act1, act2, epsilon=epsilon, verbose=False
                )
                value = sum(result["mean"]) / 2
            except Exception:
                value = 0
            table[cnt] = value
        return table

    svcca_datas = {}
    keys = resnet_keys + plainnet_keys
    labels = labels_resnets + labels_plainnets
    ij_list = list(itertools.combinations(range(len(keys)), 2))
    for i, j in tqdm(ij_list, total=len(ij_list)):
        key = "{} vs {}".format(labels[i], labels[j])
        svcca_datas[key] = calc_mean_svcca(keys[i], keys[j])

    n_color = 0
    for key in svcca_datas.keys():
        list_key = key.split("vs")
        if ("ResNet" in list_key[0] and "PlainNet" in list_key[1]) or (
            "ResNet" in list_key[1] and "PlainNet" in list_key[0]
        ):
            continue
        n_color += 1
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, n_color)]
    scale = 4 / 5
    plt.figure(figsize=(16 * scale, 9 * scale))
    cnt = 0
    for key, data in svcca_datas.items():
        list_key = key.split("vs")
        if ("ResNet" in list_key[0] and "PlainNet" in list_key[1]) or (
            "ResNet" in list_key[1] and "PlainNet" in list_key[0]
        ):
            continue
        plt.plot(data, label=key, color=colors[cnt])
        cnt += 1
    plt.legend()
    plt.xticks(
        ticks=np.arange(len(candidate_layers)), labels=candidate_layers, rotation=90
    )
    plt.grid()
    plt.ylim(0, 1)
    if out_prefix is not None:
        make_dir(out_dir)
        out_name = "{}-res-vs-plain.png".format(out_prefix)
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, transparent=True)
    else:
        plt.show()
    plt.close("all")

    #     cmap = plt.get_cmap("rainbow")
    #     colors = [cmap(i) for i in np.linspace(0, 1, len(svcca_datas))]
    #     scale = 4 / 5
    #     plt.figure(figsize=(16 * scale, 9 * scale))
    #     cnt = 0
    #     for key, data in svcca_datas.items():
    #         list_key = key.split("-")
    #         plt.plot(data, label=key, color=colors[cnt])
    #         cnt += 1
    #     plt.legend()
    #     plt.grid()
    #     plt.ylim(0, 1)
    #     if out_prefix is not None:
    #         make_dir(out_dir)
    #         out_name = "{}-all.png".format(out_prefix)
    #         out_path = os.path.join(out_dir, out_name)
    #         plt.savefig(out_path, transparent=True)
    #     else:
    #         plt.show()
    #     plt.close("all")

    def calc_mean_svcca_random(model_1, model_2, epsilon=0):
        table = np.zeros(len(candidate_layers))

        for cnt, layer in tqdm(
            enumerate(candidate_layers), total=len(candidate_layers)
        ):
            act1 = fmap_pools[model_1][layer].numpy()
            act1 = np.transpose(act1, (1, 0, 2, 3))
            act1 = act1.reshape(len(act1), -1).astype(np.float64)
            perm = np.random.permutation(act1.shape[-1])
            n_sample = 100 * len(act1)
            act1 = act1[:, perm[:n_sample]]

            act2 = fmap_pools[model_2][layer].numpy()
            act2 = np.transpose(act2, (1, 0, 2, 3))
            act2 = act2.reshape(len(act2), -1).astype(np.float64)
            n_sample = 100 * len(act2)
            act2 = act2[:, perm[:n_sample]]
            try:
                result = svcca.cca_core.get_cca_similarity(
                    act1, act2, epsilon=epsilon, verbose=False
                )
                value = sum(result["mean"]) / 2
            except Exception:
                value = 0
            table[cnt] = value
        return table

    svcca_datas_random = {}
    keys = resnet_keys
    labels = labels_resnets
    for i, j in itertools.combinations(range(len(keys)), 2):
        key = "{} vs {}".format(labels[i], labels[j])
        svcca_datas_random[key] = calc_mean_svcca_random(keys[i], keys[j])
    keys = plainnet_keys
    labels = labels_plainnets
    for i, j in itertools.combinations(range(len(keys)), 2):
        key = "{} vs {}".format(labels[i], labels[j])
        svcca_datas_random[key] = calc_mean_svcca_random(keys[i], keys[j])

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(svcca_datas_random))]
    scale = 4 / 5
    plt.figure(figsize=(16 * scale, 9 * scale))
    cnt = 0
    for key, data in svcca_datas_random.items():
        list_key = key.split("-")
        plt.plot(data, label=key, color=colors[cnt])
        cnt += 1
    plt.legend()
    plt.grid()
    plt.ylim(0, 1)
    plt.xticks(
        ticks=np.arange(len(candidate_layers)), labels=candidate_layers, rotation=90
    )
    if out_prefix is not None:
        make_dir(out_dir)
        out_name = "{}-random-sample.png".format(out_prefix)
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, transparent=True)
    else:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()

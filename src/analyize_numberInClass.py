import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from utils.dumpLoaders import RFdataLoader

# sample size
n_sample = 100
n_classes = 1000
exts = ("png", "pdf")


def main(
    analysis_root,
    rf_root,
    model_names,
    labels,
    key_layer_and_chmax_blockmax=None,
    vmin=None,
    vmax=None,
    out_dir=None,
):
    resnet34_model_names, plainnet34_model_names = model_names
    labels_resnets, labels_plainnets = labels
    keywords = {"channel", "config", "config-script"}
    rfdataloader = RFdataLoader(analysis_root, rf_root, keywords=keywords)
    data_resnets = {}
    data_plainnets = {}

    if out_dir is None:
        out_dir = "analysis_numberInClass"
    # ResNet34
    if key_layer_and_chmax_blockmax is None:
        key_layer_and_chmax_blockmax = [
            ("maxpool", 64, None),
            ("layer1", 64, 3),
            ("layer2", 128, 4),
            ("layer3", 256, 6),
            ("layer4", 512, 3),
        ]

    for key_layer, ch_max, block_max in key_layer_and_chmax_blockmax:

        def get_top_act_classes(model_name, key_layer, block_id):
            rfdataloader.set_vis_layer(model_name, key_layer, block_id)
            top_act_classes = []
            for ch in range(ch_max):
                rfdataloader.set_ch_data(ch)
                activation_indeces = rfdataloader.activation_indeces
                if activation_indeces is not None:
                    img_idxs = activation_indeces[0][:n_sample]
                    top_act_classes.append(
                        np.array(rfdataloader.dataset.targets)[img_idxs]
                    )
            top_act_classes = np.asarray(top_act_classes)
            return top_act_classes.copy()

        block_list = (
            range(block_max)
            if block_max is not None
            else [
                None,
            ]
        )
        for block_id in block_list:
            print(key_layer, block_id)
            for model_name in resnet34_model_names:
                res_top_act_classes = get_top_act_classes(
                    model_name, key_layer, block_id
                )
                res_uniques = [
                    np.unique(x, return_counts=True) for x in res_top_act_classes
                ]
                res_unique_len = np.asarray([len(unique) for unique, _ in res_uniques])

                key = "{}-{}-{}".format(model_name, key_layer, block_id)
                data_resnets[key] = [res_top_act_classes, res_uniques, res_unique_len]

            for model_name in plainnet34_model_names:
                res_top_act_classes = get_top_act_classes(
                    model_name, key_layer, block_id
                )
                res_uniques = [
                    np.unique(x, return_counts=True) for x in res_top_act_classes
                ]
                res_unique_len = np.asarray([len(unique) for unique, _ in res_uniques])

                key = "{}-{}-{}".format(model_name, key_layer, block_id)
                data_plainnets[key] = [res_top_act_classes, res_uniques, res_unique_len]

        show(
            data_resnets,
            data_plainnets,
            labels_resnets,
            labels_plainnets,
            key_layer,
            block_id,
            block_max,
            ch_max,
            vmin=vmin,
            vmax=vmax,
            out_dir=out_dir,
        )

    # save datas
    if len(resnet34_model_names) > 0:
        config = {
            "model_names": resnet34_model_names,
        }
        save_datas(data_resnets, config, "data_resnets.pkl", out_dir=out_dir)

    if len(plainnet34_model_names) > 0:
        config = {
            "model_names": plainnet34_model_names,
        }
        save_datas(data_plainnets, config, "data_plainnets.pkl", out_dir=out_dir)


def show(
    data_resnets,
    data_plainnets,
    labels_resnets,
    labels_plainnets,
    key_layer,
    block_id,
    block_max,
    ch_max,
    show_flag=False,
    out_dir="analysis_numberInClass",
    vmin=None,
    vmax=None,
):
    res_unique_lens = []
    for _, item in data_resnets.items():
        res_unique_lens.append(item[2])
    res_unique_lens = np.asarray(res_unique_lens)

    plain_unique_lens = []
    for _, item in data_plainnets.items():
        plain_unique_lens.append(item[2])
    plain_unique_lens = np.asarray(plain_unique_lens)

    if vmin is None or vmax is None:
        vmin = n_classes + 1
        vmax = -1
        for _, item in data_resnets.items():
            _vmin = item[2].min()
            _vmax = item[2].max()
            if _vmin < vmin:
                vmin = _vmin
            if _vmax > vmax:
                vmax = _vmax
        for _, item in data_plainnets.items():
            _vmin = item[2].min()
            _vmax = item[2].max()
            if _vmin < vmin:
                vmin = _vmin
            if _vmax > vmax:
                vmax = _vmax

    block_list = (
        range(block_max)
        if block_max is not None
        else [
            None,
        ]
    )
    for block_id in block_list:
        res_unique_lens = []
        for key, item in data_resnets.items():
            if "{}-{}".format(key_layer, block_id) in key:
                res_unique_lens.append(item[2])
        res_unique_lens = np.asarray(res_unique_lens)

        plain_unique_lens = []
        for key, item in data_plainnets.items():
            if "{}-{}".format(key_layer, block_id) in key:
                plain_unique_lens.append(item[2])
        plain_unique_lens = np.asarray(plain_unique_lens)

        data_res_size = len(res_unique_lens)
        data_plain_size = len(plain_unique_lens)
        alpha = 0.6
        channel_size = ch_max
        plt.figure(figsize=(10, 5))
        boxprops = {
            "facecolor": "tab:blue",
            "alpha": alpha,
        }
        if data_res_size > 0:
            plt.boxplot(
                res_unique_lens.T,
                positions=np.arange(data_res_size),
                patch_artist=True,
                boxprops=boxprops,
            )
        boxprops = {
            "facecolor": "tab:green",
            "alpha": alpha,
        }
        if data_plain_size > 0:
            plt.boxplot(
                plain_unique_lens.T,
                positions=np.arange(data_res_size, data_plain_size + data_res_size),
                patch_artist=True,
                boxprops=boxprops,
            )
        plt.xlim(0 - 1, data_res_size + data_plain_size)
        plt.xticks(
            ticks=np.arange(data_res_size + data_plain_size),
            labels=labels_resnets + labels_plainnets,
            rotation=90,
        )
        if block_max is None:
            msg = "{} ({} channels)".format(key_layer, channel_size)
        else:
            msg = "{}.{} ({} channels)".format(key_layer, block_id, channel_size)
        plt.title(msg)
        msg = "# of class in each channel"
        plt.ylabel(msg)
        plt.ylim(vmin - 1, vmax + 1)
        plt.grid()
        plt.tight_layout()
        if out_dir is not None:
            makedirs(out_dir)
            for ext in exts:
                if block_id is None:
                    out_name = "{}.{}".format(key_layer, ext)
                else:
                    out_name = "{}-{}.{}".format(key_layer, block_id, ext)
                out_path = os.path.join(out_dir, out_name)
                plt.savefig(out_path, transparent=True)
        if show_flag:
            plt.show()
        plt.close("all")


def makedirs(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


# save datas
def save_datas(data, config, out_name, out_dir="analysis_numberInClass"):
    makedirs(out_dir)

    save_files = {
        "data": data,
        "config": config,
    }

    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(save_files, f)


# save datas
def load_datas(in_name, in_dir="analysis_numberInClass"):
    in_path = os.path.join(in_dir, in_name)
    with open(in_path, "wb") as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    import yaml

    argv = sys.argv
    if len(argv) > 1:
        path_config_path = argv[1]
    else:
        path_config_path = "path-config.yaml"

    with open(path_config_path, "r") as conf_file:
        loaded = yaml.load(conf_file, Loader=yaml.FullLoader)
    analysis_root = loaded["analysis_root"]
    rf_root = loaded["rf_root"]

    if len(argv) > 2:
        import yaml

        path = argv[2]
        with open(path, "r") as conf_file:
            loaded = yaml.load(conf_file, Loader=yaml.FullLoader)
        resnet34_model_names = loaded["resnet34_model_names"]
        plainnet34_model_names = loaded["plainnet34_model_names"]
        labels_resnets = loaded["labels_resnets"]
        labels_plainnets = loaded["labels_plainnets"]
        key_layer_and_chmax_blockmax = loaded["key_layer_and_chmax_blockmax"]
        out_dir = loaded["out_dir"]
        vmin = loaded["vmin"]
        vmax = loaded["vmax"]
    else:
        from utils.analysis import (labels_plainnets, labels_resnets,
                                    plainnet34_model_names,
                                    resnet34_model_names)

        out_dir = None
        key_layer_and_chmax_blockmax = None
        vmin = 0
        vmax = 100
    main(
        analysis_root,
        rf_root,
        (resnet34_model_names, plainnet34_model_names),
        (labels_resnets, labels_plainnets),
        key_layer_and_chmax_blockmax=key_layer_and_chmax_blockmax,
        out_dir=out_dir,
        vmin=vmin,
        vmax=vmax,
    )

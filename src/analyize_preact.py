import gc
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm as tqdm
from umap import UMAP

from my_model import get_ilsvrc2012
from my_model.blocks import ResNet
from utils import analysis, plots
from utils.analysis import make_dir, save_arr
from utils.load_model import TransferModel, get_prelayer_name, get_resblock
from utils.receptive_field import get_rf_region
from utils.receptive_field_tracker import RFTracker
from utils.tensortracker import TensorTracker


def main(args):
    # main args
    root = args.root
    max_ch_cnt = args.max_ch_cnt
    model_name = analysis.get_arch_name(args.arch, args.wandb_flag)
    key_layer = args.layer_name
    layer_num = int(key_layer[-1])
    out_dir = os.path.join(args.out, model_name)
    N = args.N
    workers = args.workers
    specific_channel = args.specific_channel
    start_end_channel = args.start_end_channel

    # constant values
    fname_args = "config.yaml"
    fname_config = "config.pkl"
    fname_rf_data = "top_rf_datas"
    fname_top_activation = "top_activation"

    # temporary value
    arch = None

    if start_end_channel is None:
        ch_list = (
            tqdm(range(1, max_ch_cnt), total=max_ch_cnt)
            if specific_channel is None
            else (specific_channel,)
        )
    else:
        tmp_start, tmp_end = start_end_channel
        tmp_len = tmp_end - tmp_start
        assert tmp_len > 0
        ch_list = tqdm(range(tmp_start, tmp_end), total=tmp_len)

    for ch in ch_list:
        configs = []
        args_datas = []

        rfcntmaps = []
        rfimgs = []
        rfgrads = []
        top_channels = []
        search_key = "*{}*{}*".format(model_name, key_layer)
        for analysis_path in sorted(glob(os.path.join(root, search_key))):
            path = os.path.join(analysis_path, fname_config)
            # print(path)
            with open(path, "rb") as f:
                d = pickle.load(f)
            configs.append(d)
            path = os.path.join(analysis_path, fname_args)
            with open(path, "r") as f:
                args_data = yaml.load(f, Loader=yaml.SafeLoader)

            args_datas.append(args_data)

            for path in sorted(
                glob(
                    os.path.join(
                        analysis_path, fname_rf_data, "rfcntmap*{:03}*".format(ch)
                    )
                )
            ):
                rfcntmaps.append(np.load(path))

            for path in sorted(
                glob(
                    os.path.join(
                        analysis_path, fname_rf_data, "rfimg*{:03}*".format(ch)
                    )
                )
            ):
                rfimgs.append(np.load(path))

            for path in sorted(
                glob(
                    os.path.join(
                        analysis_path, fname_rf_data, "rfgrad*{:03}*".format(ch)
                    )
                )
            ):
                rfgrads.append(np.load(path))

            for path in sorted(
                glob(
                    os.path.join(
                        analysis_path, fname_top_activation, "channel*{:03}*".format(ch)
                    )
                )
            ):
                with open(path, "rb") as f:
                    top_channels.append(pickle.load(f))

        for i in range(len(rfimgs)):
            pre_layer_name = get_prelayer_name(layer_num, i, model_type="34")
            # resnet34 or plainnet34
            tmp_args = args_datas[i]["args"]
            out_fmap_name = "relu2"
            cur_arch = tmp_args["arch"]
            if arch is None or cur_arch != arch:
                arch = cur_arch
                if "wandb_dir" in tmp_args.keys() or "wandb_flag" in tmp_args.keys():
                    wandb_flag = True
                else:
                    wandb_flag = False
                model = analysis.get_model_from_keywords(
                    cur_arch, wandb_flag, no_grad_fe=True
                )
                model = model.to(args.device)
                model = model.eval()

            if isinstance(model, (TransferModel,)):
                # monkey patch for TransferModel consists of ResBlocks
                if len(model.classifier) == 4 and layer_num == 4 and i == 0:
                    # assumpt classifer is consits of layer4, avgpool, flatten, and fc
                    # calculated target is layer4.0
                    pre_layer_name = "feature_extractor"
                else:
                    pre_layer_name = "classifier." + pre_layer_name
            pre_acts, acts = get_preacts_and_acts(
                model,
                i,
                ch,
                pre_layer_name,
                args_datas,
                configs,
                top_channels,
                out_fmap_name,
                device=args.device,
            )
            if pre_acts is None or acts is None:
                continue
            tmp_n = N if N < pre_acts.shape[0] else pre_acts.shape[0]
            np_pre_vectors = pre_acts.to("cpu").numpy()[:tmp_n].reshape(tmp_n, -1)
            np_acts = acts.to("cpu").numpy()[:tmp_n]
            plt_args = {}
            if out_dir is not None:
                dname = "{}-{}".format(key_layer, i)
                make_dir(os.path.join(out_dir, dname))
                out_path = os.path.join(
                    out_dir, dname, "act_preact_{:03}.pkl".format(ch)
                )
                save_arr(out_path, [np_acts, np_pre_vectors])
                plt_args.update(
                    {
                        "outdir": os.path.join(out_dir, dname),
                        "prefix": "prevector",
                        "suffix": "ch{:03}".format(ch),
                    }
                )

            if tmp_n >= 4:
                plot_something(
                    np_pre_vectors, np_acts, N=tmp_n, plt_args=plt_args, workers=workers
                )

                np_rfimgs = rfimgs[i][:tmp_n].reshape(tmp_n, -1)
                do_flags = {"PCA"}
                plt_args.update(
                    {
                        "prefix": "rfimg",
                    }
                )
                plot_something(
                    np_rfimgs,
                    np_acts,
                    N=tmp_n,
                    do_flags=do_flags,
                    plt_args=plt_args,
                    workers=workers,
                )
            gc.collect()


def get_preacts_and_acts(
    model,
    i,
    ch,
    pre_layer_name,
    args_datas,
    configs,
    top_channels,
    out_fmap_name,
    device="cpu",
):

    conf_args = args_datas[i]["args"]
    val_list = conf_args["val_list"]
    rnd_dataset = get_ilsvrc2012(mode="test", val_txt=val_list)
    w, _, _, _ = configs[i]["layer_info"]
    shape = (len(rnd_dataset), w, w)
    tmp_index = np.unravel_index(top_channels[i]["top_act_index"], shape)
    img_idxs = []
    xs = []
    ys = []
    for img_idx, x, y in zip(*tmp_index):
        img_idxs.append(img_idx)
        xs.append(x)
        ys.append(y)

    if len(img_idxs) == 0:
        return None, None

    images = torch.stack([rnd_dataset[img_idx][0] for img_idx in img_idxs])

    layer_name = conf_args["layer_name"]
    tracker = TensorTracker(model, candidate_layers=[layer_name, pre_layer_name])
    if isinstance(model, (ResNet,)):
        rf_tracker = RFTracker(get_resblock(model, layer_name))
    elif isinstance(model, (TransferModel,)):
        rf_tracker = RFTracker(
            get_resblock(model.classifier, layer_name[len("classifier.") :])
        )
        # rf_tracker = RFTracker(get_resblock(model, layer_name))
    else:
        raise TypeError(type(model))

    _ = model(images.to(device))

    rf_tracker.remove_hook()
    pre_fmap = tracker.find_fmap(pre_layer_name)
    fmap = tracker.find_fmap(layer_name)
    acts = fmap[range(len(fmap)), ch, xs, ys]

    keys = list(rf_tracker.rf_pool.keys())
    block_image_size = rf_tracker.get_rf_info(keys[0])[0]
    tracker.remove_hook()

    pre_fmap_shape = pre_fmap.shape

    pre_vectors = []
    debug_pre_vectors = []
    rf_size = rf_tracker.get_rf_info(out_fmap_name)[2]
    tmp_shape = (
        pre_fmap_shape[0],
        pre_fmap_shape[1],
        pre_fmap_shape[2] + rf_size,
        pre_fmap_shape[3] + rf_size,
    )
    tmp_fmap = torch.zeros(tmp_shape)
    tmp_fmap[
        :, :, rf_size // 2 : -rf_size // 2, rf_size // 2 : -rf_size // 2
    ] = pre_fmap
    for cnt in range(len(fmap)):
        center, rf = rf_tracker.find_receptive_field((xs[cnt], ys[cnt]), out_fmap_name)
        dx, dy = get_rf_region(center, rf, image_size=block_image_size, clipping=False)
        pre_vector = pre_fmap[cnt, :, dx[0] : dx[1], dy[0] : dy[1]]
        debug_pre_vectors.append(pre_vector)
        dx = (dx[0] + rf_size // 2, dx[1] + rf_size // 2)
        dy = (dy[0] + rf_size // 2, dy[1] + rf_size // 2)
        pre_vector = tmp_fmap[cnt, :, dx[0] : dx[1], dy[0] : dy[1]]
        pre_vectors.append(pre_vector)

    pre_vectors = torch.stack(pre_vectors)
    return pre_vectors, acts


def plot_something(
    np_pre_vectors,
    np_acts,
    N=100,
    random_state=1119,
    do_flags=None,
    plt_args=dict(),
    workers=1,
):

    if do_flags is None:
        do_flags = {"PCA", "TSNE", "UMAP", "ACTS"}

    assert isinstance(do_flags, (list, set, tuple))

    if "scatter-figsize" not in plt_args:
        plt_args["scatter-figsize"] = (8, 8)
    if "pca-cumexp-figsize" not in plt_args:
        plt_args["pca-cumexp-figsize"] = (8, 4)
    if "act-figsize" not in plt_args:
        plt_args["act-figsize"] = (8, 4)

    plt.rcParams["font.size"] = 20
    alpha = 0.5

    def plt_savefig(key_outname, ext="png"):
        if "outdir" in plt_args:
            outdir = plt_args["outdir"]
            prefix = plt_args["prefix"]
            suffix = plt_args["suffix"]
            outname = "{}-{}-{}.{}".format(prefix, key_outname, suffix, ext)
            make_dir(os.path.join(outdir, "imgs"))
            path = os.path.join(outdir, "imgs", outname)
            plt.savefig(path, transparent=True)

    # PCA
    if "pca".upper() in do_flags:
        pca = PCA(svd_solver="full")

        pca.fit(np_pre_vectors)
        pca_data = pca.transform(np_pre_vectors)

        plt.figure(figsize=plt_args["pca-cumexp-figsize"])
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.ylim(0, 1)
        plt.ylabel("cumulative contribution rate")
        plt.xlabel("principal components")
        plt_savefig("pca-cumexp")
        plt.close()

        plt.figure(figsize=plt_args["scatter-figsize"])
        plots.scatter_linear_colors(pca_data, reverse=True, alpha=alpha)
        plt.xlabel("1st principal component")
        plt.ylabel("2nd principal component")
        plt_savefig("pca-scatter")
        plt.close()

    # t-sne
    if "tsne".upper() in do_flags:
        tsne = TSNE(random_state=random_state)
        tsne_data = tsne.fit_transform(np_pre_vectors)

        plt.figure(figsize=plt_args["scatter-figsize"])
        plots.scatter_linear_colors(tsne_data, reverse=True, alpha=alpha)
        plt_savefig("tsne-scatter")
        plt.close()

    # umap
    def helpler_umapshow(umap_model, mean_color="tab:red", mean_s=80):
        data2d = umap_model.fit_transform(np_pre_vectors)
        mean_v = umap_model.transform(np_pre_vectors.mean(0)[None, :])
        plots.scatter_linear_colors(data2d, reverse=True, alpha=alpha)
        plt.scatter(mean_v[:, 0], mean_v[:, 1], color=mean_color, s=mean_s)

    if "umap".upper() in do_flags:
        n_neighbor_list = [3, 5]
        while n_neighbor_list[-1] < N:
            n = n_neighbor_list[-1]
            n_neighbor_list.append(int(2 * n))

        if n_neighbor_list[-1] >= N:
            n_neighbor_list.pop(-1)
            n_neighbor_list.append(N - 1)

        # if n = 100 then n_neighbor_list = [3, 5, 10, 20, 40, 80, 100]
        for n_neighbors in n_neighbor_list:
            plt.figure(figsize=plt_args["scatter-figsize"])
            umap_model = UMAP(random_state=random_state, n_neighbors=n_neighbors)
            helpler_umapshow(umap_model)
            plt_savefig("umap-neighbors{:03}".format(n_neighbors))
            plt.close()

    # activation
    if "acts".upper() in do_flags:
        plt.figure(figsize=plt_args["act-figsize"])
        plt.plot(np_acts)
        plt.ylabel("activation value")
        plt.xlabel("index")
        plt_savefig("acts")
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="make data receptive field")
    parser.add_argument(
        "-l", "--layer-name", type=str, default="layer1", help="layer name"
    )
    parser.add_argument(
        "-r", "--root", type=str, required=True, help="rf-datas directory name"
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="output directory name"
    )
    parser.add_argument(
        "-a", "--arch", type=str, default="resnet34", help="model architecture"
    )
    parser.add_argument(
        "--wandb-flag", action="store_true", help="Is arch wandb directory?"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=6, help="worker number of using dataloader"
    )
    parser.add_argument("--max-ch-cnt", type=int, default=64, help="max channel size")
    parser.add_argument(
        "--specific-channel",
        type=int,
        default=None,
        help="only work at the channel. If it is None then all channels",
    )
    parser.add_argument(
        "--start-end-channel", type=int, nargs=2, default=None, help="for debuging. "
    )
    parser.add_argument("--N", type=int, default=100, help="")
    parser.add_argument("--device", type=str, default="cuda", help="model architecture")
    args = parser.parse_args()

    main(args)

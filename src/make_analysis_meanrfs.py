# coding: utf-8
import functools
import os
import pickle
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm as tqdm

from my_model.my_dataset import get_ilsvrc2012
from utils.performance_model import MySampler
from utils.plots import imshow_helper, input2image, plot_imshows

DO_SAMPLE = True
DO_PCA = True
DO_PCA_EXP = True

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) > 0:
        analysis_root = argv[1]
        glob_path = argv[2]
    else:
        analysis_root = "./analysis/receptive_field/"
        glob_path = "resnet34-*[p|n]_*layer[1-2]*"
    img_normalize = functools.partial(input2image, img_format="HWC")
    save_flag = True
    batch_size = 256
    workers = 4
    topk = 16
    pca_rate = 0.8
    key_dir = "top_activateion"
    rnd_dataset = None
    print("analyize {} {}".format(analysis_root, glob_path))
    file_list = sorted(glob(os.path.join(analysis_root, glob_path)))
    print(file_list)
    for _cnt, path in enumerate(file_list):
        config_path = os.path.join(path, "config.pkl")
        try:
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        except FileNotFoundError:
            print("Not Found {}".format(path))
            continue

        mean_rf_data_path = os.path.join(path, "mean_rf_datas.pkl")
        with open(mean_rf_data_path, "rb") as f:
            mean_rf_data = pickle.load(f)

        layer_info = config["layer_info"]
        sorted_channels = config["sorted_channels"]
        mean_rfs = mean_rf_data["mean_rfs"]

        out_dir_sample = os.path.join(path, "imgs", "samples")
        out_dir_pca = os.path.join(path, "imgs", "pca")
        out_dir_fft = os.path.join(path, "imgs", "fft")
        out_dir_pca_explained_v = os.path.join(path, "imgs", "pca_exp")
        if not os.path.exists(out_dir_sample):
            os.makedirs(out_dir_sample)
        if not os.path.exists(out_dir_pca_explained_v):
            os.makedirs(out_dir_pca_explained_v)
        #         if not os.path.exists(out_dir_fft):
        #             os.makedirs(out_dir_fft)
        if not os.path.exists(out_dir_pca):
            os.makedirs(out_dir_pca)
        bnames = os.listdir(os.path.join(path, key_dir))
        for bname in tqdm(bnames, total=len(bnames)):
            ch = int(bname[len("channel-") : -len(".pkl")])
            channel_path = os.path.join(path, key_dir, bname)
            with open(channel_path, "rb") as f:
                channel_data = pickle.load(f)

            val_list = channel_data["val_text"]
            np_img_idx = channel_data["image_index"]
            np_xs = channel_data["xs"]
            np_ys = channel_data["ys"]

            if rnd_dataset is None:
                rnd_dataset = get_ilsvrc2012(mode="test", val_txt=val_list)
            data_loader = torch.utils.data.DataLoader(
                rnd_dataset,
                sampler=MySampler(rnd_dataset, np_img_idx),
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=True,
            )
            clip_img_all = []
            for cnt, (imgs, _) in enumerate(data_loader):
                xs_i = cnt * batch_size
                xs = np_xs[xs_i : xs_i + batch_size]
                ys = np_ys[xs_i : xs_i + batch_size]
                clip_imgs = np.zeros((len(xs), 3, layer_info[2], layer_info[2]))
                for i in range(len(xs)):
                    pad_img = (
                        F.pad(
                            imgs[i],
                            (
                                layer_info[2],
                                layer_info[2],
                                layer_info[2],
                                layer_info[2],
                            ),
                        )
                        .to("cpu")
                        .numpy()
                    )
                    clip_imgs[i] = pad_img[:, xs[i][0] : xs[i][1], ys[i][0] : ys[i][1]]
                clip_img_all.append(clip_imgs)
            clip_img_all = np.asarray(clip_img_all)

            sorted_ch = ch
            if DO_SAMPLE:
                if len(clip_img_all) == 0:
                    print("skip DO SAMPLE ==> {}".format(channel_path))
                    continue
                plot_imshows(
                    clip_img_all[0][:topk],
                    normalize=img_normalize,
                    out_dir=out_dir_sample,
                    out_name="ch-{:03}.png".format(sorted_ch),
                )

            clip_img_all = clip_img_all.reshape(-1, 3, layer_info[2], layer_info[2])
            n, c, w, h = clip_img_all.shape

            if len(clip_img_all) == 0 or 1 >= n:
                print("skip DO PCA ==> {}".format(channel_path))
                continue
            pca = PCA(svd_solver="full")
            X = clip_img_all.reshape(len(clip_img_all), -1)
            pca.fit(X)

            pca_index = np.cumsum(pca.explained_variance_ratio_) > pca_rate
            if len(np.arange(len(pca.explained_variance_ratio_))[pca_index]) == 0:
                print(pca.explained_variance_ratio_)
                print(pca_index)
            n = np.arange(len(pca.explained_variance_ratio_))[pca_index][0]
            m = np.ceil(np.sqrt(n))
            if DO_PCA:
                plt.figure(figsize=(m, m))
                for i, component in enumerate(pca.components_):
                    if i >= n:
                        break
                    eign_vector = component.reshape(c, w, h)
                    plt.subplot(m, m, i + 1)
                    imshow_helper(
                        (eign_vector - eign_vector.min())
                        / (eign_vector - eign_vector.min()).max()
                    )
                    plt.axis("off")
                plt.tight_layout()
                if save_flag:
                    out_path = os.path.join(
                        out_dir_pca, "ch-{:03}.png".format(sorted_ch)
                    )
                    plt.savefig(out_path, transparent=True)
                plt.close()

            if save_flag and DO_PCA_EXP:
                pca_index = np.cumsum(pca.explained_variance_ratio_) <= pca_rate
                y_margin = 0.1
                out_path = os.path.join(
                    out_dir_pca_explained_v, "all-ch-{:03}.png".format(sorted_ch)
                )
                plt.figure(figsize=(4, 3))
                plt.plot(np.cumsum(pca.explained_variance_ratio_))
                plt.ylim(0 - y_margin, 1 + y_margin)
                plt.savefig(out_path, transparent=True)
                plt.close()
                out_path = os.path.join(
                    out_dir_pca_explained_v, "ch-{:03}.png".format(sorted_ch)
                )
                plt.figure(figsize=(4, 3))
                plt.plot(np.cumsum(pca.explained_variance_ratio_)[pca_index])
                plt.ylim(0 - y_margin, 1 + y_margin)
                plt.savefig(out_path, transparent=True)
                plt.close()

#             f_clip_imgs = np.fft.fftn(
#                 clip_img_all.reshape(-1, layer_info[2], layer_info[2]), axes=(-2, -1))
#             f_clip_imgs = f_clip_imgs.reshape(-1,
#                                               3, layer_info[2], layer_info[2])
#
#             plt.figure(figsize=(2 * 2, 2))
#             plt.subplot(1, 2, 1)
#             plt.title('amplitude')
#             f_clip = f_clip_imgs.mean(0)
#             tmp_img = np.fft.ifftn(f_clip.real, axes=(-2, -1)).real
#             tmp_img -= tmp_img.min()
#             tmp_img /= tmp_img.max()
#             imshow_helper(tmp_img)
#             plt.subplot(1, 2, 2)
#             plt.title('phase')
#             f_clip = f_clip - f_clip.real
#             tmp_img = np.fft.ifftn(f_clip, axes=(-2, -1)).real
#             tmp_img -= tmp_img.min()
#             tmp_img /= tmp_img.max()
#             imshow_helper(tmp_img)
#     for i, f_clip in enumerate(f_clip_imgs.mean(0)):
#         plt.subplot(1, 4, i + 2)
#         if i == 0:
#             cmap = 'Reds'
#         elif i == 1:
#             cmap = 'Greens'
#         elif i == 2:
#             cmap = 'Blues'
#         imshow_helper(np.fft.ifftn(f_clip.real, axes=(-2, -1)).real, cmap=cmap)
#         plt.colorbar()
#            plt.tight_layout()
#            if save_flag:
#                out_path = os.path.join(
#                    out_dir_fft, 'ch-{:03}.png'.format(sorted_ch))
#                plt.savefig(out_path, transparent=True)
#            plt.close()

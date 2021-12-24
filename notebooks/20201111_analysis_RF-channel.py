# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: dlb2-pytorch
#     language: python
#     name: dlb2-pytorch
# ---

# +
import os
import pickle
import sys
from glob import glob

import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from tqdm import tqdm as tqdm

# -


sys.path.append("../src")

# +
from my_model import get_ilsvrc2012
from utils import plots
from utils.load_model import get_model
from utils.receptive_field import cut_rf_from_img_helper
from utils.receptive_field_tracker import RFTracker
from utils.tensortracker import TensorTracker

# -

root = "/mnt/nas3/lab_member_directories/2021_genta/resnet/e_receptive_field/"

for analysis_path in glob(os.path.join(root, "*")):
    print(analysis_path)


fname_args = "config.yaml"
fname_config = "config.pkl"
fname_rf_data = "top_rf_datas"

# +
ch = 0
args_datas = []
configs = []

rfcntmaps = []
rfimgs = []
rfgrads = []

for analysis_path in glob(os.path.join(root, "*resnet*layer1*")):
    path = os.path.join(analysis_path, fname_args)
    with open(path, "r") as f:
        args_data = yaml.load(f)

    args_datas.append(args_data)

    path = os.path.join(analysis_path, fname_config)
    with open(path, "rb") as f:
        d = pickle.load(f)
    #         downconv = d["downconv"]
    #         layer_info = d["layer_info"]
    #         image_size = d["image_size"]
    #         sorted_channels = d["sorted_channels"]
    configs.append(d)

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfcntmap*{:03}*".format(ch)))
    ):
        rfcntmaps.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfimg*{:03}*".format(ch)))
    ):
        rfimgs.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfgrad*{:03}*".format(ch)))
    ):
        rfgrads.append(np.load(path))

# rfcntmaps = np.asarray(rfcntmaps)
# rfimgs = np.asarray(rfimgs)
# rfgrads = np.asarray(rfgrads)

# +
topk = 4
N = 100
nrow = 2

mean_rf_imgs = []
mean_rf_grads = []
mean_erfs = []

top_rfimgs = []
top_rfgrads = []
top_erfs = []

cluster_rfimgs = []
cluster_rfgrads = []
cluster_erfs = []


for rfcntmap, rfimg, rfgrad in zip(rfcntmaps, rfimgs, rfgrads):
    image_shape = rfimg.shape[1:]

    mean_rf_img = plots.input2image(rfimg[:N].sum(0) / rfcntmap)
    mean_rf_grad = plots.norm_img(rfgrad[:N].sum(0) / rfcntmap)
    mean_erf = plots.norm_img((rfimg[:N] * np.abs(rfgrad[:N])).sum(0) / rfcntmap)

    top_rfimg = plots.make_grid(plots.input2image(rfimg[:topk]), nrow=nrow)
    top_rfgrad = plots.make_grid(plots.norm_img(rfgrad[:topk]), nrow=nrow)
    top_erf = plots.make_grid(
        plots.norm_img(np.abs(rfgrad[topk]) * rfimg[:topk]), nrow=nrow
    )

    mean_rf_imgs.append(mean_rf_img)
    mean_rf_grads.append(mean_rf_grad)
    mean_erfs.append(mean_erf)

    top_rfimgs.append(top_rfimg)
    top_rfgrads.append(top_rfgrad)
    top_erfs.append(top_erf)

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfimg[:N].reshape(N, -1))
    cluster_rfimg = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfimg = plots.make_grid(
        plots.input2image(cluster_rfimg), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfgrad[:N].reshape(N, -1))
    cluster_rfgrad = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfgrad = plots.make_grid(
        plots.norm_img(cluster_rfgrad), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit((rfimg[:N] * np.abs(rfgrad[:N])).reshape(N, -1))
    cluster_erf = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_erf = plots.make_grid(plots.norm_img(cluster_erf), copy=False, nrow=nrow)

    cluster_rfimgs.append(cluster_rfimg)
    cluster_rfgrads.append(cluster_rfgrad)
    cluster_erfs.append(cluster_erf)


tmp_img = []
tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
plots.plot_imshows(tmp_img, show_flag=True, scale=2, nrow=9, normalize=None)
# -


# +
# resnet34 ch, 173(dog ?), 137(dog ?), 76 (human?)

# +
out_root = "/data2/genta/resnet/e_receptive_field/out/"

key_layer = "layer3"

# +
ch = 76
args_datas = []
configs = []

rfcntmaps = []
rfimgs = []
rfgrads = []
nrow = 2
topk = 4


for analysis_path in sorted(glob(os.path.join(root, "*resnet*{}*".format(key_layer)))):
    print(analysis_path)
    path = os.path.join(analysis_path, fname_args)
    with open(path, "r") as f:
        args_data = yaml.load(f)

    args_datas.append(args_data)

    path = os.path.join(analysis_path, fname_config)
    with open(path, "rb") as f:
        d = pickle.load(f)
    configs.append(d)

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfcntmap*{:03}*".format(ch)))
    ):
        rfcntmaps.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfimg*{:03}*".format(ch)))
    ):
        rfimgs.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfgrad*{:03}*".format(ch)))
    ):
        rfgrads.append(np.load(path))

# -

out_path = os.path.join(out_root, key_layer)
if not os.path.join(out_path):
    os.makedirs(out_path)

# +

N = 100

mean_rf_imgs = []
mean_rf_grads = []
mean_erfs = []

top_rfimgs = []
top_rfgrads = []
top_erfs = []

cluster_rfimgs = []
cluster_rfgrads = []
cluster_erfs = []


for rfcntmap, rfimg, rfgrad in zip(rfcntmaps, rfimgs, rfgrads):
    image_shape = rfimg.shape[1:]

    mean_rf_img = plots.input2image(rfimg[:N].sum(0) / rfcntmap)
    mean_rf_grad = plots.norm_img(rfgrad[:N].sum(0) / rfcntmap)
    mean_erf = plots.norm_img((rfimg[:N] * np.abs(rfgrad[:N])).sum(0) / rfcntmap)

    top_rfimg = plots.make_grid(plots.input2image(rfimg[:topk]), nrow=nrow)
    top_rfgrad = plots.make_grid(plots.norm_img(rfgrad[:topk]), nrow=nrow)
    top_erf = plots.make_grid(
        plots.norm_img(np.abs(rfgrad[topk]) * rfimg[:topk]), nrow=nrow
    )

    mean_rf_imgs.append(mean_rf_img)
    mean_rf_grads.append(mean_rf_grad)
    mean_erfs.append(mean_erf)

    top_rfimgs.append(top_rfimg)
    top_rfgrads.append(top_rfgrad)
    top_erfs.append(top_erf)

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfimg[:N].reshape(N, -1))
    cluster_rfimg = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfimg = plots.make_grid(
        plots.input2image(cluster_rfimg), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfgrad[:N].reshape(N, -1))
    cluster_rfgrad = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfgrad = plots.make_grid(
        plots.norm_img(cluster_rfgrad), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit((rfimg[:N] * np.abs(rfgrad[:N])).reshape(N, -1))
    cluster_erf = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_erf = plots.make_grid(plots.norm_img(cluster_erf), copy=False, nrow=nrow)

    cluster_rfimgs.append(cluster_rfimg)
    cluster_rfgrads.append(cluster_rfgrad)
    cluster_erfs.append(cluster_erf)


tmp_img = []
tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
# -

plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=2,
    nrow=9,
    normalize=None,
    out_dir=out_path,
    out_name="all_{:03}.png".format(ch),
)

fname = "top_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)

fname = "mean_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)

fname = "kmeans_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)


# +
ch = 173
args_datas = []
configs = []

rfcntmaps = []
rfimgs = []
rfgrads = []
nrow = 2
topk = 4


for analysis_path in sorted(glob(os.path.join(root, "*resnet*{}*".format(key_layer)))):
    print(analysis_path)
    path = os.path.join(analysis_path, fname_args)
    with open(path, "r") as f:
        args_data = yaml.load(f)

    args_datas.append(args_data)

    path = os.path.join(analysis_path, fname_config)
    with open(path, "rb") as f:
        d = pickle.load(f)
    configs.append(d)

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfcntmap*{:03}*".format(ch)))
    ):
        rfcntmaps.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfimg*{:03}*".format(ch)))
    ):
        rfimgs.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfgrad*{:03}*".format(ch)))
    ):
        rfgrads.append(np.load(path))


# +

N = 100

mean_rf_imgs = []
mean_rf_grads = []
mean_erfs = []

top_rfimgs = []
top_rfgrads = []
top_erfs = []

cluster_rfimgs = []
cluster_rfgrads = []
cluster_erfs = []


for rfcntmap, rfimg, rfgrad in zip(rfcntmaps, rfimgs, rfgrads):
    image_shape = rfimg.shape[1:]

    mean_rf_img = plots.input2image(rfimg[:N].sum(0) / rfcntmap)
    mean_rf_grad = plots.norm_img(rfgrad[:N].sum(0) / rfcntmap)
    mean_erf = plots.norm_img((rfimg[:N] * np.abs(rfgrad[:N])).sum(0) / rfcntmap)

    top_rfimg = plots.make_grid(plots.input2image(rfimg[:topk]), nrow=nrow)
    top_rfgrad = plots.make_grid(plots.norm_img(rfgrad[:topk]), nrow=nrow)
    top_erf = plots.make_grid(
        plots.norm_img(np.abs(rfgrad[topk]) * rfimg[:topk]), nrow=nrow
    )

    mean_rf_imgs.append(mean_rf_img)
    mean_rf_grads.append(mean_rf_grad)
    mean_erfs.append(mean_erf)

    top_rfimgs.append(top_rfimg)
    top_rfgrads.append(top_rfgrad)
    top_erfs.append(top_erf)

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfimg[:N].reshape(N, -1))
    cluster_rfimg = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfimg = plots.make_grid(
        plots.input2image(cluster_rfimg), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfgrad[:N].reshape(N, -1))
    cluster_rfgrad = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfgrad = plots.make_grid(
        plots.norm_img(cluster_rfgrad), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit((rfimg[:N] * np.abs(rfgrad[:N])).reshape(N, -1))
    cluster_erf = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_erf = plots.make_grid(plots.norm_img(cluster_erf), copy=False, nrow=nrow)

    cluster_rfimgs.append(cluster_rfimg)
    cluster_rfgrads.append(cluster_rfgrad)
    cluster_erfs.append(cluster_erf)


tmp_img = []
tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
# -

plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=2,
    nrow=9,
    normalize=None,
    out_dir=out_path,
    out_name="all_{:03}.png".format(ch),
)

fname = "top_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)

fname = "mean_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)

fname = "kmeans_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)


# +
ch = 137
args_datas = []
configs = []

rfcntmaps = []
rfimgs = []
rfgrads = []
nrow = 2
topk = 4


for analysis_path in sorted(glob(os.path.join(root, "*resnet*{}*".format(key_layer)))):
    print(analysis_path)
    path = os.path.join(analysis_path, fname_args)
    with open(path, "r") as f:
        args_data = yaml.load(f)

    args_datas.append(args_data)

    path = os.path.join(analysis_path, fname_config)
    with open(path, "rb") as f:
        d = pickle.load(f)
    configs.append(d)

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfcntmap*{:03}*".format(ch)))
    ):
        rfcntmaps.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfimg*{:03}*".format(ch)))
    ):
        rfimgs.append(np.load(path))

    for path in sorted(
        glob(os.path.join(analysis_path, fname_rf_data, "rfgrad*{:03}*".format(ch)))
    ):
        rfgrads.append(np.load(path))


# +

N = 100

mean_rf_imgs = []
mean_rf_grads = []
mean_erfs = []

top_rfimgs = []
top_rfgrads = []
top_erfs = []

cluster_rfimgs = []
cluster_rfgrads = []
cluster_erfs = []


for rfcntmap, rfimg, rfgrad in zip(rfcntmaps, rfimgs, rfgrads):
    image_shape = rfimg.shape[1:]

    mean_rf_img = plots.input2image(rfimg[:N].sum(0) / rfcntmap)
    mean_rf_grad = plots.norm_img(rfgrad[:N].sum(0) / rfcntmap)
    mean_erf = plots.norm_img((rfimg[:N] * np.abs(rfgrad[:N])).sum(0) / rfcntmap)

    top_rfimg = plots.make_grid(plots.input2image(rfimg[:topk]), nrow=nrow)
    top_rfgrad = plots.make_grid(plots.norm_img(rfgrad[:topk]), nrow=nrow)
    top_erf = plots.make_grid(
        plots.norm_img(np.abs(rfgrad[topk]) * rfimg[:topk]), nrow=nrow
    )

    mean_rf_imgs.append(mean_rf_img)
    mean_rf_grads.append(mean_rf_grad)
    mean_erfs.append(mean_erf)

    top_rfimgs.append(top_rfimg)
    top_rfgrads.append(top_rfgrad)
    top_erfs.append(top_erf)

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfimg[:N].reshape(N, -1))
    cluster_rfimg = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfimg = plots.make_grid(
        plots.input2image(cluster_rfimg), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit(rfgrad[:N].reshape(N, -1))
    cluster_rfgrad = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_rfgrad = plots.make_grid(
        plots.norm_img(cluster_rfgrad), copy=False, nrow=nrow
    )

    kmeans = KMeans(n_clusters=topk, random_state=0, n_jobs=8)
    kmeans.fit((rfimg[:N] * np.abs(rfgrad[:N])).reshape(N, -1))
    cluster_erf = kmeans.cluster_centers_.reshape((-1,) + image_shape)
    cluster_erf = plots.make_grid(plots.norm_img(cluster_erf), copy=False, nrow=nrow)

    cluster_rfimgs.append(cluster_rfimg)
    cluster_rfgrads.append(cluster_rfgrad)
    cluster_erfs.append(cluster_erf)


tmp_img = []
tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
# -

plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=2,
    nrow=9,
    normalize=None,
    out_dir=out_path,
    out_name="all_{:03}.png".format(ch),
)

fname = "top_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)

fname = "mean_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)

fname = "kmeans_{:03}.png".format(ch)
tmp_img = []
tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
plots.plot_imshows(
    tmp_img,
    show_flag=True,
    scale=3,
    nrow=3,
    normalize=None,
    out_dir=out_path,
    out_name=fname,
)

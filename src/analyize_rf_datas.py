import os
import pickle
from glob import glob

import numpy as np
from sklearn.cluster import KMeans
# import yaml
from tqdm import tqdm as tqdm

from utils import plots

exts = ("png", "pdf")


def main(args):
    # fname_args = "config.yaml"
    fname_config = "config.pkl"
    fname_rf_data = "top_rf_datas"

    root = args.root
    out_root = args.out

    model_name = args.arch
    key_layer = args.layer_name
    N = 100

    out_path = os.path.join(out_root, model_name, key_layer)
    if not os.path.join(out_path):
        os.makedirs(out_path)

    max_ch_cnt = args.max_ch_cnt
    topk = args.topk

    for ch in tqdm(range(max_ch_cnt), total=max_ch_cnt):
        configs = []
        #        args_datas = []

        rfcntmaps = []
        rfimgs = []
        rfgrads = []
        nrow = 2
        padding = 4
        search_key = "*{}*{}*".format(model_name, key_layer)
        for analysis_path in sorted(glob(os.path.join(root, search_key))):
            path = os.path.join(analysis_path, fname_config)
            # print(path)
            with open(path, "rb") as f:
                d = pickle.load(f)
            configs.append(d)
            #             path = os.path.join(analysis_path, fname_args)
            #             with open(path, "r") as f:
            #                 args_data = yaml.load(f)
            #
            #             args_datas.append(args_data)

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

            if not args.skip_mean:
                if args.normalize_mean:
                    mean_rf_img = plots.input2image(
                        plots.normalize_inputspace((rfimg[:N].sum(0) / rfcntmap))
                    )
                else:
                    mean_rf_img = plots.input2image(rfimg[:N].sum(0) / rfcntmap)
                mean_rf_grad = plots.input2image(
                    plots.normalize_inputspace(rfgrad[:N].sum(0) / rfcntmap)
                )

                coeff = np.abs(plots.normalize_inputspace(rfgrad[:N]))
                if args.normalize_mean:
                    meanimg = plots.normalize_inputspace(
                        (rfimg[:N] * coeff).sum(0) / rfcntmap
                    )
                else:
                    meanimg = (rfimg[:N] * coeff).sum(0) / rfcntmap
                mean_erf = plots.input2image(meanimg)

                mean_rf_imgs.append(mean_rf_img)
                mean_rf_grads.append(mean_rf_grad)
                mean_erfs.append(mean_erf)

            if not args.skip_topk:
                meanimgs = plots.input2image(rfimg[:topk])
                top_rfimg = plots.make_grid(meanimgs, nrow=nrow, padding=padding)
                top_rfgrad = plots.make_grid(
                    plots.input2image(plots.normalize_inputspace(rfgrad[:topk])),
                    nrow=nrow,
                    padding=padding,
                )
                coeff = np.abs(plots.normalize_inputspace(rfgrad[:topk]))
                top_erf = plots.make_grid(
                    plots.input2image(coeff * rfimg[:topk]), nrow=nrow, padding=padding
                )

                top_rfimgs.append(top_rfimg)
                top_rfgrads.append(top_rfgrad)
                top_erfs.append(top_erf)

            if not args.skip_kmeans:
                kmeans = KMeans(
                    n_clusters=args.n_clusters, random_state=0, n_jobs=args.workers
                )
                kmeans.fit(rfimg[:N].reshape(N, -1))
                cluster_rfimg = kmeans.cluster_centers_.reshape((-1,) + image_shape)
                cluster_rfimg = plots.make_grid(
                    plots.input2image(cluster_rfimg),
                    copy=False,
                    nrow=nrow,
                    padding=padding,
                )

                kmeans = KMeans(
                    n_clusters=args.n_clusters, random_state=0, n_jobs=args.workers
                )
                kmeans.fit(rfgrad[:N].reshape(N, -1))
                cluster_rfgrad = kmeans.cluster_centers_.reshape((-1,) + image_shape)
                cluster_rfgrad = plots.make_grid(
                    plots.input2image(plots.normalize_inputspace(cluster_rfgrad)),
                    copy=False,
                    nrow=nrow,
                    padding=padding,
                )

                kmeans = KMeans(
                    n_clusters=args.n_clusters, random_state=0, n_jobs=args.workers
                )
                coeff = np.abs(plots.normalize_inputspace(rfgrad[:N]))
                kmeans.fit((rfimg[:N] * coeff).reshape(N, -1))
                cluster_erf = kmeans.cluster_centers_.reshape((-1,) + image_shape)
                cluster_erf = plots.make_grid(
                    plots.input2image(cluster_erf),
                    copy=False,
                    nrow=nrow,
                    padding=padding,
                )

                cluster_rfimgs.append(cluster_rfimg)
                cluster_rfgrads.append(cluster_rfgrad)
                cluster_erfs.append(cluster_erf)

        if not args.skip_topk:
            fname = "top_{:03}".format(ch)
            tmp_img = []
            nrow = 0
            if not args.skip_meanrf:
                tmp_img = tmp_img + top_rfimgs
                nrow += 1
            if not args.skip_gradrf:
                tmp_img = tmp_img + top_rfgrads
                nrow += 1
            if not args.skip_meanerf:
                tmp_img = tmp_img + top_erfs
                nrow += 1
            # tmp_img = tmp_img + top_rfimgs + top_rfgrads + top_erfs
            plots.plot_imshows(
                tmp_img,
                show_flag=False,
                scale=3,
                nrow=nrow,
                normalize=None,
                out_dir=out_path,
                out_name=fname,
                exts=exts,
            )

        if not args.skip_mean:
            if args.normalize_mean:
                fname = "normlize_mean_{:03}".format(ch)
            else:
                fname = "mean_{:03}".format(ch)
            tmp_img = []
            nrow = 0
            if not args.skip_meanrf:
                tmp_img = tmp_img + mean_rf_imgs
                nrow += 1
            if not args.skip_gradrf:
                tmp_img = tmp_img + mean_rf_grads
                nrow += 1
            if not args.skip_meanerf:
                tmp_img = tmp_img + mean_erfs
                nrow += 1
            # tmp_img = tmp_img + mean_rf_imgs + mean_rf_grads + mean_erfs
            plots.plot_imshows(
                tmp_img,
                show_flag=False,
                scale=3,
                nrow=nrow,
                normalize=None,
                out_dir=out_path,
                out_name=fname,
                exts=exts,
            )

        if not args.skip_kmeans:
            fname = "kmeans_{:03}".format(ch)
            tmp_img = []
            nrow = 0
            if not args.skip_meanrf:
                tmp_img = tmp_img + cluster_rfimgs
                nrow += 1
            if not args.skip_gradrf:
                tmp_img = tmp_img + cluster_rfgrads
                nrow += 1
            if not args.skip_meanerf:
                tmp_img = tmp_img + cluster_erfs
                nrow += 1
            # tmp_img = tmp_img + cluster_rfimgs + cluster_rfgrads + cluster_erfs
            plots.plot_imshows(
                tmp_img,
                show_flag=False,
                scale=3,
                nrow=nrow,
                normalize=None,
                out_dir=out_path,
                out_name=fname,
                exts=exts,
            )


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
        "-w", "--workers", type=int, default=6, help="worker number of using dataloader"
    )
    parser.add_argument("--max-ch-cnt", type=int, default=64, help="max channel size")
    parser.add_argument("--topk", type=int, default=4, help="")
    parser.add_argument("--normalize-mean", action="store_true", help="")
    parser.add_argument("--skip-mean", action="store_true", help="skip mean data")
    parser.add_argument("--skip-topk", action="store_true", help="skip top k")
    parser.add_argument("--skip-kmeans", action="store_true", help="skip kmeans")
    parser.add_argument("--skip-meanrf", action="store_true", help="skip mean rf")
    parser.add_argument("--skip-meanerf", action="store_true", help="skip mean erf")
    parser.add_argument("--skip-gradrf", action="store_true", help="skip grad")
    parser.add_argument("--n-clusters", type=int, default=4, help="")
    args = parser.parse_args()

    main(args)

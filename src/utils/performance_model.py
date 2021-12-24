import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from make_vis_data import get_dataset, get_model, sampling_index


class MySampler(Sampler):
    def __init__(self, data_source, indeces):
        self.data_source = data_source
        self.indeces = indeces

    def __iter__(self):
        return iter(self.indeces)

    def __len__(self):
        return len(self.indeces)


def accuracy(output, target, topk=(1,), with_incorrectindex=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        incorrect_imgindex = []
        for k in topk:
            index = correct[:k].sum(0).eq(0)
            incorrect_imgindex.append(index)
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if with_incorrectindex:
            return res, incorrect_imgindex
        else:
            return res


def measure_performance(
    model, dataset, sampler=None, num_workers=0, batch_size=200, gpu=-1, **kwargs
):
    if "with_acts" in kwargs:
        with_acts = kwargs["with_acts"]
    else:
        with_acts = False

    if gpu > -1:
        model.cuda(gpu)

    acc1_list = []
    acc5_list = []

    acc1_indexlist = []
    acc5_indexlist = []
    loss_list = []
    out_list = []

    loss_func = nn.CrossEntropyLoss()
    if gpu > -1:
        loss_func.cuda(gpu)

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if gpu > -1:
                images = images.cuda(gpu, non_blocking=True)
                targets = targets.cuda(gpu, non_blocking=True)

            output = model(images)
            loss = loss_func(output, targets)
            (acc1, acc5), (index1, index5) = accuracy(output, targets, topk=(1, 5))
            acc1_list.append(acc1[0].item())
            acc5_list.append(acc5[0].item())
            if gpu > -1:
                index1 = index1.cpu()
                index5 = index5.cpu()
            acc1_indexlist.append(index1.numpy())
            acc5_indexlist.append(index5.numpy())
            loss_list.append(loss.item())
            if with_acts:
                out_list.append(output.to("cpu").detach().numpy())
    np_data = (np.asarray(loss_list), np.asarray(acc1_list), np.asarray(acc5_list))
    np_index = (np.asarray(acc1_indexlist), np.asarray(acc5_indexlist))
    if with_acts:
        return np_data, np_index, out_list
    else:
        return np_data, np_index


def main(out_dir, **kwargs):
    save_dir = os.path.join(out_dir, "data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = get_model(kwargs["arch"])
    dataset, _ = get_dataset(kwargs["dataset_name"])
    index = sampling_index(dataset, kwargs["dataset_name"])

    print("measure perfromance")
    data, index = measure_performance(
        model, dataset, sampler=MySampler(dataset, index), **kwargs
    )

    fnames = ["acc1", "acc5", "loss", "acc1_incorrectindex", "acc5_incorrectindex"]
    for i, d in enumerate((data + index)):
        path = os.path.join(save_dir, fnames[i])
        print("save data to {}".format(path))
        np.save(path, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-g", "--gpu", type=int, default=-1, help="gpu id. if id = -1 then cpu"
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="output directory name"
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default="imagenet", help="dataset name"
    )
    parser.add_argument("-a", "--arch", type=str, default="resnet34", help="model name")
    parser.add_argument(
        "-p", "--first_pad", action="store_false", help="disable initial padding"
    )
    args = parser.parse_args()
    main(args.out, dataset_name=args.dataset_name, arch=args.arch)

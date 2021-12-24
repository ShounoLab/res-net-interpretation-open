import os

import git
import six
import yaml


def save_config_file_from_args(config_filename, args, add_dict=None):
    d = dict()
    d.update({"args": vars(args)})
    d.update({"git": get_current_git_config()})
    if add_dict is not None:
        d.update(add_dict)

    save_config_file_from_dict(config_filename, d)


def get_current_git_config():
    repo = git.Repo(search_parent_directories=True)

    url = list(repo.remotes.origin.urls)[0]
    hexsha = repo.head.object.hexsha
    d = {
        "url": url,
        "hexsha": hexsha,
    }
    return d


def mkdir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


# ref: https://github.com/wandb/client/blob/e9b843e0e2da88b5edf9e4d54a347eceff0489ff/wandb/sdk/lib/config_util.py
def save_config_file_from_dict(config_filename, config_dict):
    """
    config_filename:
        path and file name of config

    config_dict:
        dictionary
    """
    s = b"config version: 1"
    s += b"\n\n" + yaml.dump(
        config_dict,
        Dumper=yaml.SafeDumper,
        default_flow_style=False,
        allow_unicode=True,
        encoding="utf-8",
    )
    data = s.decode("utf-8")
    mkdir(os.path.dirname(config_filename))
    with open(config_filename, "w") as conf_file:
        conf_file.write(data)


def dict_from_config_file(filename):
    assert os.path.exists(filename)

    conf_file = open(filename)
    loaded = yaml.load(conf_file, Loader=yaml.FullLoader)
    data = dict()
    for k, v in six.iteritems(loaded):
        data[k] = v

    return data


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
        default="analysis/receptive_field/",
        help="output directory name",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=128, help="mini batch size"
    )
    parser.add_argument(
        "-m", "--max-iter", type=int, default=None, help="max itertion number"
    )
    parser.add_argument(
        "-a", "--arch", type=str, default="resnet34", help="model architecture"
    )
    parser.add_argument("--device", type=str, default="cpu", help="model architecture")
    parser.add_argument(
        "-w", "--workers", type=int, default=6, help="worker number of using dataloader"
    )
    parser.add_argument("--max-ch-cnt", type=int, default=None, help="max channel size")
    parser.add_argument("--skip-counting", action="store_true", help="skip coutning")
    parser.add_argument("--skip-mean-rf", action="store_true", help="skip mean rf")
    args = parser.parse_args()
    print(type(vars(args)))

    save_config_file_from_args("test_confg.yaml", args)
    d = dict_from_config_file("test_confg.yaml")

    print(d["args"])
    print(d["git"])

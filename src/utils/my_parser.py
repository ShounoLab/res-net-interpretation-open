import argparse
import multiprocessing as mp


def get_argparse(name=None):
    """
    get my argparse.
    Args:
        name; name of script
    Return:
        args
    """
    parser = argparse.ArgumentParser()
    if name is None:
        msg = "please set name ot the script"
        raise ValueError(msg)
    elif "kmeans-cluster" in name:
        parser.add_argument("--rf-dir", type=str, required=True, help="rf_data path")
        parser.add_argument("--arch", type=str, required=True, help="arch path")
        parser.add_argument("--layer-name", type=str, required=True, help="layer name")
        parser.add_argument("--n-sample", type=int, default=64, help="")
        parser.add_argument("--n-jobs", type=int, default=4, help="")
        parser.add_argument("--n-clusters", type=int, default=64, help="")
    elif "rf-images" in name:
        parser.add_argument("--rf-dir", type=str, required=True, help="rf_data path")
        parser.add_argument("--arch", type=str, required=True, help="arch path")
        parser.add_argument("--layer-name", type=str, required=True, help="layer name")
        parser.add_argument("--n-jobs", type=int, default=mp.cpu_count() - 4, help="")
        parser.add_argument("--batch-size", type=int, default=256, help="")
    else:
        raise ValueError(name)

    args = parser.parse_args()

    return args

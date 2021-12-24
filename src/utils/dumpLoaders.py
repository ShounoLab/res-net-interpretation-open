import os
import pickle
from glob import glob

import numpy as np

from my_model import get_ilsvrc2012

from .config import dict_from_config_file


def glob_analysis_root(analysis_root, model_name, key_layer, block_id):
    if block_id is None:
        return glob(os.path.join(analysis_root, model_name + "*", key_layer + "*"))
    else:
        return glob(
            os.path.join(
                analysis_root, model_name + "*", key_layer + "*" + str(block_id)
            )
        )


def glob_rf_root(rf_root, model_name, key_layer, block_id):
    if block_id is None:
        return glob(os.path.join(rf_root, "*" + model_name + "*" + key_layer + "*"))
    else:
        return glob(
            os.path.join(
                rf_root, "*" + model_name + "*" + key_layer + "." + str(block_id) + "*"
            )
        )


class RFdataLoader(object):
    def __init__(self, analysis_root, rf_root, keywords=None):
        self.analysis_root = analysis_root
        self.rf_root = rf_root
        if keywords is None or keywords == "all":
            self.keywords = {
                "rfimgs",
                "rfgrads",
                "channel",
                "config",
                "act_preact",
                "config-script",
            }
        elif isinstance(keywords, (set, list, tuple)):
            self.keywords = keywords
        else:
            raise TypeError("keywords" + type(keywords))

        self._fmap_shape = None
        self._dataset = None

    def set_vis_layer(self, model_name, key_layer, block_id):
        if "act_preact" in self.keywords:
            cond_path = glob_analysis_root(
                self.analysis_root, model_name, key_layer, block_id
            )
            assert len(cond_path) != 0, "Not found"
            assert (
                len(cond_path) == 1
            ), "ambiguous; model name or key_layer or block_id. you can check paths to use _show_glob_paths"
            self.act_dir_path = cond_path[0]
        else:
            self.act_dir_path = None

        cond_path = glob_rf_root(self.rf_root, model_name, key_layer, block_id)
        assert len(cond_path) != 0, "Not found"
        assert (
            len(cond_path) == 1
        ), "ambiguous; model name or key_layer or block_id. you can check paths to use _show_glob_paths"
        self.rf_dir_path = cond_path[0]

    def set_ch_data(self, ch):
        if self.rf_dir_path is None:
            msg = "do set_vis_layer"
            raise ValueError(msg)

        if "act_preact" in self.keywords:
            path = os.path.join(self.act_dir_path, "act_preact_{:03}.pkl".format(ch))
            with open(path, "rb") as f:
                self._act_preact = pickle.load(f)

        if "rfimgs" in self.keywords:
            path = os.path.join(
                self.rf_dir_path, "top_rf_datas", "rfimgs-{:03}.npy".format(ch)
            )
            self._rfimgs = np.load(path)
        if "rfgrads" in self.keywords:
            path = os.path.join(
                self.rf_dir_path, "top_rf_datas", "rfgrads-{:03}.npy".format(ch)
            )
            self._rfgrads = np.load(path)

        if "channel" in self.keywords:
            try:
                path = os.path.join(
                    self.rf_dir_path, "top_activation", "channel-{:03}.pkl".format(ch)
                )
                with open(path, "rb") as f:
                    self._channel = pickle.load(f)
            except EOFError:
                self._channel = None

        if "config" in self.keywords:
            path = os.path.join(self.rf_dir_path, "config.pkl")
            with open(path, "rb") as f:
                self._config = pickle.load(f)

        if "config-script" in self.keywords:
            path = os.path.join(self.rf_dir_path, "config.yaml")
            self._config_script = dict_from_config_file(path)

    @property
    def dataset(self):
        if self._dataset is not None:
            return self._dataset
        # NOTE: only Imagenet
        self._dataset = get_ilsvrc2012(
            mode="test", val_txt=self._config_script["args"]["val_list"]
        )
        return self._dataset

    @property
    def fmap_spatial_shape(self):
        h = self.config["layer_info"][0]
        w = h
        return (h, w)

    @property
    def activation_indeces(self):
        if self.channel is None:
            return None

        if "fmap_shape" in self.config:
            return np.unravel_index(
                self.channel["top_act_index"],
                (self.config["len_dataset"],) + self.config["fmap_shape"][-2:],
            )
        else:
            len_dataset = len(self.dataset)
            return np.unravel_index(
                self.channel["top_act_index"], (len_dataset,) + self.fmap_spatial_shape
            )

    @property
    def rfimgs(self):
        if "rfimgs" in self.keywords:
            return self._rfimgs
        else:
            return None

    @property
    def rfgrads(self):
        if "rfgrads" in self.keywords:
            return self._rfgrads
        else:
            return None

    @property
    def act_preact(self):
        if "_act_preact" in self.keywords:
            return self._act_preact
        else:
            return None

    @property
    def channel(self):
        if "channel" in self.keywords:
            return self._channel
        else:
            return None

    @property
    def config(self):
        if "config" in self.keywords:
            return self._config
        else:
            return None

    def _show_glob_paths(self, model_name, key_layer, block_id):
        print("analysis path")
        cond_path = glob_analysis_root(
            self.analysis_root, model_name, key_layer, block_id
        )
        print(cond_path)
        cond_path = glob_rf_root(self.rf_root, model_name, key_layer, block_id)
        print("rf path")
        print(cond_path)

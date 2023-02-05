import os
from copy import deepcopy
from pathlib import Path

import munch

import unetgeo.utils.general_utils as gu


class ConfigManager(object):
    def __init__(self, config_dir, config_file_name):
        self.config_dir = config_dir
        self.config_file_name = config_file_name
        self.cfg = gu.load_yaml(Path(config_dir) / config_file_name)

        self.cfg = munch.munchify(self.cfg)
        self.cfg.path_data_folder = self._get_path_data_folder()

        self.cfg.config_file_name = config_file_name
        self.cfg.train_param.path_save_model_dir = (
            self.cfg.path_data_folder / "model" / "saved" / config_file_name
        )
  
    @staticmethod
    def _get_path_data_folder():
        path_data_repo = Path("./data")
        return path_data_repo

    @staticmethod
    def _get_path_trained_model(path_data_folder, weight_path):
        path_trained_model = Path(path_data_folder) / Path(weight_path)

        return path_trained_model

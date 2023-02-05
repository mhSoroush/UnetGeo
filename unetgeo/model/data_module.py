
from pathlib import Path
from typing import List, Union, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

import unetgeo.utils.general_utils as gu
from unetgeo.model.data_model import DataModel 


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.task = cfg.model_param.task
        self.path_data_folder = Path(cfg.path_data_folder)
        self.data_paths = cfg.data_paths
        self.s_set_batch_size = cfg.train_param.s_set_batch_size
        self.q_set_batch_size = cfg.train_param.q_set_batch_size
        self.batch_size_for_test = cfg.train_param.batch_size_for_test
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            self.s_set_data, self.q_set_data = self._prepare_train_data()
        # if stage == "test":
        #     self.test_q_set_data = self._prepare_test_datas()
            # prepare test

    def _prepare_train_data(self):
        s_set_data = get_data(
            self.path_data_folder / self.data_paths["s_set"],
            "s_set",
            self.cfg
        )
        q_set_data = get_data(
            self.path_data_folder / self.data_paths["q_set"],
            "q_set",
            self.cfg
        )

        return s_set_data, q_set_data

    def _prepare_test_datas(self):
        q_set_data = get_data(
            self.path_data_folder / self.data_paths["q_set"],
            "q_set",
            self.cfg
        )
        return q_set_data

    def _prepare_predict_data(self, path_predict_input_json):
        s_set_data = get_data(
            self.path_data_folder / self.data_paths["s_set"],
            "s_set",
            self.cfg
        )
        pr_q_set_data = get_data(
            path_predict_input_json, 
            "infer", 
            self.cfg,
            is_json=True
        )
        return s_set_data, pr_q_set_data

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        support_set_loader = DataLoader(
            batch_size= self.s_set_batch_size,
            dataset= self.s_set_data,
            shuffle= True,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        query_set_loader = DataLoader(
            batch_size= self.q_set_batch_size,
            dataset= self.q_set_data,
            shuffle= True,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        return {"s_set": support_set_loader, "q_set": query_set_loader}

    # def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     pass

    # def test_dataloader(self):
    #     query_set_loader = DataLoader(
    #         batch_size= self.q_set_batch_size,
    #         dataset= self.test_q_set_data,
    #         shuffle= False,
    #         num_workers=self.cfg.train_param.n_cpus,
    #         collate_fn=lambda x: x,
    #     )
    #     return query_set_loader

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        s_set_data, pr_q_set_data = self._prepare_predict_data(self.path_predict_input_json)
        support_set_loader = DataLoader(
            batch_size= self.s_set_batch_size,
            dataset= s_set_data,
            shuffle= False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        query_set_loader = DataLoader(
            batch_size= self.batch_size_for_test,
            dataset= pr_q_set_data,
            shuffle= False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )

        #combined_loader = CombinedLoader({"s_set":support_set_loader, "q_set": query_set_loader})
        return [support_set_loader, query_set_loader]

def get_data(fpath, mode, cfg, is_json=False):
    if is_json:
        raw_data = [gu.load_json(fpath)]
    else:
        raw_data = gu.load_jsonl(fpath, cfg["toy_data"], cfg["toy_size"])
    data = DataModel(raw_data, mode, cfg, fpath)
    return data

import os
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import torch

import unetgeo.utils.general_utils as gu
from unetgeo.model.data_module import DataModule
from unetgeo.model.model import RelationTagger

def prepare_data_model_trainer(cfg):
    path_logs = Path("./logs") / cfg.config_file_name
    os.makedirs(path_logs, exist_ok=True)
    tb_logger = pl.loggers.TensorBoardLogger(path_logs)

    data_module = DataModule(cfg)
    model = get_model(
        cfg.model_param,
        cfg.train_param,
        cfg.infer_param,
        cfg.path_data_folder,
        cfg.verbose,
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        #log_every_n_steps=cfg.train_param.get("log_every_n_steps", 50),
        gpus=torch.cuda.device_count(),
        max_epochs=cfg.train_param.max_epochs,
        #val_check_interval=cfg.train_param.val_check_interval,
        limit_train_batches=cfg.train_param.limit_train_batches,
        #limit_val_batches=cfg.train_param.limit_val_batches,
        #num_sanity_val_steps=1,
        progress_bar_refresh_rate=100,
        accumulate_grad_batches=cfg.train_param.accumulate_grad_batches,
        #accelerator=cfg.train_param.accelerator,
        precision=cfg.model_param.precision,
        gradient_clip_val=cfg.train_param.gradient_clip_val,
        gradient_clip_algorithm=cfg.train_param.gradient_clip_algorithm,
    )

    return data_module, model, trainer


def get_model(hparam, tparam, iparam, path_data_folder, verbose=False):
    if hparam.model_name == "RelationTagging":
        model = RelationTagger(
            hparam, tparam, iparam, path_data_folder, verbose=verbose
        )
    else:
        raise NotImplementedError

    return model


def do_training(cfg):
    data_module, model, trainer = prepare_data_model_trainer(cfg)
    trainer.fit(model, data_module)


# def do_testing(cfg):
#     data_module, model, trainer = prepare_data_model_trainer(cfg)
#     trainer.test(model, datamodule=data_module)


def do_prediction(cfg, path_predict_input_json):
    data_module, model, trainer = prepare_data_model_trainer(cfg)
    assert cfg.raw_data_input_type == "type1"

    data_module.path_predict_input_json = path_predict_input_json
    out = trainer.predict(model, datamodule=data_module)
    s_features = out[0][0]
    q_features = out[1][0]
    results = model.get_output_of_qSet_vs_sSet(s_features, q_features, "predict")

    path_to_save = path_predict_input_json.__str__() + ".out.json"
    gu.write_json(path_to_save, results["pr_parses"][0])
    print("\n \n")
    pprint(results["pr_parses"][0])

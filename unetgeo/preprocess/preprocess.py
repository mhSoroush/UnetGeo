
from unetgeo.model.data_module import DataModule
from unetgeo.preprocess.preprocess_utils import run_preprocess


def do_preprocess(cfg):
    if cfg.model_param.task in ["funsd", "lehrvertrag"]:
        # download and combine the original data preprocess
        run_preprocess(cfg)
    
    data_module = DataModule(cfg)
    s_set_data, q_set_data = data_module._prepare_train_data()
    datas = [s_set_data, q_set_data]
    for data in datas:
        data.gen_type1_data()

    pass

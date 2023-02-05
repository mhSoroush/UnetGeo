
from unetgeo.model.run_model import do_prediction, do_training
from unetgeo.preprocess.preprocess import do_preprocess
from unetgeo.utils.config_manager import ConfigManager


class Agent(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def do_preprocess(self):
        do_preprocess(self.cfg)

    def do_training(self):
        do_training(self.cfg)

    def do_prediction(self, path_predict_input_json):
        do_prediction(self.cfg, path_predict_input_json)

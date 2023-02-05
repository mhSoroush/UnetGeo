import argparse
import torch
torch.cuda.empty_cache()
import os
from unetgeo import Agent, ConfigManager
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    args = get_args()
    cfg = ConfigManager(args.config_dir, args.config_file_name).cfg
    agent = Agent(cfg)
    if args.mode == "preprocess":
        agent.do_preprocess()
    elif args.mode == "train":
        # Geometric-based solution has no training process, 
        # so it does actually testing 
        agent.do_training()
    elif args.mode == "predict":
        agent.do_prediction(args.path_predict_input_json)
    else:
        raise ValueError


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file_name", default="funsd.preprocess.yaml")
    parser.add_argument(
        "-m", "--mode", help="preprocess|train|eval|make_resource|release|serve", default="preprocess"
    )  # list
    parser.add_argument("-d", "--config_dir", default="./configs")
    parser.add_argument("-p", "--path_predict_input_json", default="")
    args = parser.parse_args()
    if args.path_predict_input_json:
        assert args.mode == "predict"
    return args

if __name__ == "__main__":
    main()
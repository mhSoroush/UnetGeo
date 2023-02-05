#!/bin/bash
config_file=funsd.train.yaml
file=main.py
arg1="-c $config_file -m predict -p ./data/results/funsd_predict.json "
cuda_n=0
export CUDA_VISIBLE_DEVICES=$cuda_n
python3 $file $arg1
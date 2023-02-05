import json
import multiprocessing
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import torch
import yaml
import numpy as np
import cv2


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_json(path, out1):
    with open(path, "wt", encoding="utf-8") as f:
        json_str = json.dumps(out1, cls=NumpyEncoder, ensure_ascii=False)
        json_str += "\n"
        f.writelines(json_str)


def write_jsonl(path, out):
    with open(path, "wt", encoding="utf-8") as f:
        for out1 in out:
            json_str = json.dumps(out1, cls=NumpyEncoder, ensure_ascii=False)
            json_str += "\n"
            f.writelines(json_str)

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
        data = json.loads(data.strip())

    return data

def load_yaml(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.full_load(f)

    return data

def load_jsonl(filepath, toy_data=False, toy_size=4, shuffle=False):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size:
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print("The data shuffled.")
        seed = 1
        random.Random(seed).shuffle(data)  # fixed

    return data

# def gen_slices(dim, i, j):
#     _s = slice(i, j)
#     _ss = [_s] * dim
#     return _ss

def get_local_rank():
    """
    Pytorch lightning save local rank to environment variable "LOCAL_RANK".
    From rank_zero_only
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank

def timeit(fun):
    def timed(*args, **kw):
        st = time.time()
        result = fun(*args, **kw)
        ed = time.time()
        print(f"Execution time of {fun.__name__} = {ed - st}s")
        return result

    return timed

def get_img_size(img_path):
    img = cv2.imread(img_path)
    if len(img.shape) == 3:
        (img_height, img_width, img_channel) = img.shape
    elif len(img.shape == 2):
        (img_height, img_width) = img.shape
        img_channel = None
    else:
        raise NotImplementedError
    
    return img_height, img_width, img_channel

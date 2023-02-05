import os
from collections import Iterable
from copy import deepcopy
from functools import reduce
from pathlib import Path
from tracemalloc import take_snapshot

import numpy
import numpy as np
import torch
import unetgeo.utils.general_utils as gu

def get_dim_of_id(id):
    if isinstance(id[0], Iterable):
        # id = np.array(id)
        dim = len(id.shape)
        for d1 in id.shape:
            assert (
                d1 == id.shape[0]
            )  # assume the length in each dimension is identical.
    else:
        dim = 1

    return dim

def collect_features_batchwise(features):
    data_ids = [x["data_id"] for x in features]
    img_paths = [x["img_path"] for x in features]
    img_512_paths = [x["img_512_path"] for x in features]
    mask_512_paths = [x["mask_512_path"] for x in features]
    image_urls = [x["image_url"] for x in features]
    texts = [x["text"] for x in features]
    labels = [x["label"] for x in features]
    coords = [x["coord"] for x in features]
    verticals = [x["vertical"] for x in features]
    coord_centers = [x["coord_center"] for x in features]

    return (
        data_ids,
        img_paths, 
        img_512_paths,
        mask_512_paths,
        image_urls,
        texts,
        labels,
        coords,
        verticals,
        coord_centers
    )


class RelationTaggerUtils:
    @classmethod
    def load_trained_model(cls, model, batch_data_in):
        (
            data_ids,
            img_paths,
            img_512_paths,
            mask_512_paths
        ) = batch_data_in

        model(
            data_ids,
            img_paths,
            img_512_paths,
            mask_512_paths
        )
    
    @classmethod
    def save_pr_json(cls,path_analysis_dir,file_name, file):
        os.makedirs(path_analysis_dir, exist_ok=True)
        gu.write_json(
            Path(path_analysis_dir / file_name), file
        )
        
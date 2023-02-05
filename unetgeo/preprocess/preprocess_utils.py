import os
from pathlib import Path

import cv2

import unetgeo.utils.general_utils as gu


def get_filepaths(load_dir, file_extension):
    fnames = os.listdir(load_dir)
    filepaths = [os.path.join(load_dir, f) for f in fnames]
    # files = filter(os.path.isfile, files)
    filepaths = [x for x in filepaths if x.endswith(file_extension)]
    return fnames, filepaths


def gen_data(mode,cfg):
    assert mode in ["s_set", "q_set"]
    data_dir = Path("./data") / cfg.dataset_path

    if mode in ["s_set", "q_set"]:
        path_ori0 = data_dir / "originalAnd512" / f"{mode}"

    path_ori_json = path_ori0 / "annotations"
    path_ori_img = path_ori0 / "images"
    path_512_img = path_ori0 / "images_512"
    path_512_mask = path_ori0 / "masks_512"

    fnames_json, filepaths_json = get_filepaths(path_ori_json, ".json")

    ##
    new_data = []
    for fname, fpath in zip(fnames_json, filepaths_json):
        t1 = gu.load_json(fpath)
        t1["meta"] = {}
        t1["meta"]["fname"] = fname
        image_id = fname.split(".")[0]
        t1["meta"]["image_id"] = image_id

        fpath_img = path_ori_img / f"{image_id}.png"
        fpath_512_img = path_512_img / f"{image_id}.png"
        fpath_512_mask = path_512_mask / f"{image_id}.png"
        t1["meta"]["image_path"] = fpath_img.__str__()
        t1["meta"]["image_512_path"] = fpath_512_img.__str__()
        t1["meta"]["mask_512_path"] = fpath_512_mask.__str__()


        img = cv2.imread(fpath_img.__str__())
        width, height = img.shape[1], img.shape[0]
        t1["meta"]["image_size"] = {"width": width, "height": height}
        new_data.append(t1)

    path_save = data_dir / mode / f"{mode}_type0.jsonl"
    os.makedirs(os.path.dirname(path_save), exist_ok=True)

    gu.write_jsonl(path_save, new_data)
    return None


def run_preprocess(cfg):
    for mode in ["s_set", "q_set"]:
        gen_data(mode, cfg)

import numpy as np
import torch
import unetgeo.model.data_utils as du
import unetgeo.utils.general_utils as gu


class DataModel(torch.utils.data.Dataset):
    def __init__(self, raw_data, mode, cfg, fpath):
        self.task = cfg.model_param.task
        self.mode = mode

        self.fields = cfg.model_param.fields
        self.field_rs = cfg.model_param.field_representers
        self.n_fields = len(self.fields)
        self.augment_data = cfg.train_param.augment_data
        self.augment_coord = cfg.train_param.augment_coord
        self.coord_aug_params = cfg.train_param.initial_coord_aug_params

        self.raw_data_input_type = cfg.raw_data_input_type

        self.data = self._normalize_raw_data(raw_data)
        
        self.fpath = fpath

    def gen_type1_data(self):
        def _normalize_label(label):
            if label is not None:
                if isinstance(label[0], np.ndarray):
                    label = [x.tolist() for x in label]
            return label

        assert self.raw_data_input_type == "type0"
        # 1. Generate type1 data
        new_data = []
        for data1 in self.data:
            (
                data_id,
                text,
                coord,
                vertical,
                label,
                img_sz,
                img_feature,
                img_path,
                img_512_path,
                mask_512_path,
                img_url,
            ) = self._get_each_field_from_raw_data(data1)

            label = _normalize_label(label)
            new_data1 = {
                "data_id": data_id,
                "fields": self.fields,
                "field_rs": self.field_rs,
                "text": text,
                "label": list(label) if label is not None else None,
                "coord": coord,
                "vertical": vertical,
                "img_sz": img_sz,
                "img_feature": img_feature,
                "img_path": img_path,
                "img_512_path": img_512_path,
                "mask_512_path": mask_512_path,
                "img_url": img_url,
            }
            new_data.append(new_data1)

        # 2. save
        fpath_str = self.fpath.__str__()
        print(f"Working on {fpath_str}")
        assert fpath_str.endswith(".jsonl")
        if fpath_str.endswith("_type0.jsonl"):
            path_save = fpath_str[:-12] + "_type1.jsonl"
        else:
            path_save = fpath_str[:-6] + "_type1.jsonl"
        gu.write_jsonl(path_save, new_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "s_set":
            feature = self.gen_feature(
                self.data[idx],
                self.augment_data,
                self.augment_coord,
                self.coord_aug_params,
            )
        else:
            # q_set/test
            feature = self.gen_feature(
                self.data[idx],
                augment_data=False,
                augment_coord=False,
                coord_aug_params=None,
            )
        return feature

    def _normalize_raw_data(self, raw_data):
        if self.raw_data_input_type == "type0":
            data = self._normalize_raw_data_type0(raw_data)

        elif self.raw_data_input_type == "type1":
            # optimized data format
            data = raw_data
        else:
            raise NotImplementedError

        return data

    def _normalize_raw_data_type0(self, raw_data):
        data = []
        for raw_data1 in raw_data:
            t1 = {}
            if self.mode != "infer":
                label, feature = du.get_label_and_feature(
                    raw_data1, self.task, self.fields, self.field_rs
                )

                img_sz, confidence, data_id, img_path, img_512_path, mask_512_path = du.get_meta_feature(
                    self.task, raw_data1, feature
                )
            else:  # infer
                (
                    label,
                    feature,
                    confidence,
                    img_sz,
                    data_id,
                    img_path,
                    img_512_path, 
                    mask_512_path
                ) = du.get_label_and_feature_infer_mode(self.task, raw_data1)

            text = [x[0] for x in feature]
            coord = [x[1] for x in feature]
            is_vertical = [x[2] for x in feature]

            t1["data_id"] = data_id
            t1["label"] = label
            t1["ocr_feature"] = {
                "text": text,
                "coord": coord,
                "is_vertical": is_vertical,
                "confidence": confidence,
            }
            t1["img_sz"] = img_sz
            t1["img_feature"] = None  # shall be used later
            t1["img_path"] = img_path
            t1["img_512_path"] = img_512_path
            t1["mask_512_path"]= mask_512_path
            if "meta" in raw_data1:
                t1["image_url"] = raw_data1["meta"].get("image_url")
            else:
                t1["image_url"] = None
            data.append(t1)

        return data

    def gen_feature(
        self,
        data,
        augment_data,
        augment_coord,
        coord_aug_params,
    ):
        if augment_data:
            assert self.mode == "train"

        (
            data_id,
            text,
            coord,
            vertical,
            label,
            img_sz,
            img_feature,
            img_path,
            img_512_path,
            mask_512_path,
            image_url,
        ) = self._get_each_field_from_raw_data(data)

        if self.mode == "infer":
            (text, coord, vertical) = du.remove_blank_box(text, coord, vertical)

        if augment_coord:
            img = None
            clip_box_coord = True
            _, coord = du.gen_augmented_coord(
                img, coord, img_sz, coord_aug_params, clip_box_coord
            )

        coord_center = self.get_center(np.array(coord))

        feature = {
            "data_id": data_id,
            "image_url": image_url,
            "img_path": img_path,
            "img_512_path": img_512_path,
            "mask_512_path": mask_512_path,
            "text": text,
            "label": torch.as_tensor(label) if self.mode != "infer" else label,
            "coord": torch.as_tensor(coord), 
            "vertical": torch.as_tensor(vertical), 
            "coord_center": torch.as_tensor(np.array(coord_center)) 
        }
        return feature

    def _get_each_field_from_raw_data(self, t1):
        if self.raw_data_input_type == "type1":
            data_id = t1["data_id"]
            text = t1["text"]
            coord = t1["coord"]
            vertical = t1["vertical"]
            label = t1["label"]
            img_sz = t1["img_sz"]
            img_feature = t1["img_feature"]
            img_path = t1["img_path"]
            img_512_path = t1["img_512_path"]
            mask_512_path = t1["mask_512_path"]
            image_url = t1["img_url"]
        elif self.raw_data_input_type == "type0":
            data_id = t1["data_id"]
            text = t1["ocr_feature"]["text"]
            coord = t1["ocr_feature"]["coord"]
            vertical = t1["ocr_feature"]["is_vertical"]
            label = t1["label"]
            if self.mode == "infer":
                assert label is None
            img_sz = t1["img_sz"]
            img_feature = t1["img_feature"]
            img_path = t1["img_path"]
            img_512_path = t1["img_512_path"]
            mask_512_path = t1["mask_512_path"]
            image_url = t1.get("image_url")
        else:
            raise NotImplementedError

        return (
            data_id, 
            text, 
            coord, 
            vertical, 
            label, 
            img_sz, 
            img_feature, 
            img_path, 
            img_512_path, 
            mask_512_path, 
            image_url
        )
  
    def get_center(self, coord):
        center = []
        for coord1 in coord:
            center1 = np.sum(coord1, axis=0) / 4
            center.append(center1)
        return center
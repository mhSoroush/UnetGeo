from copy import deepcopy
from pathlib import Path
from pprint import pprint
from sys import exit
from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import unetgeo.model.model_utils as mu
import unetgeo.utils.analysis_utils as au
import unetgeo.utils.general_utils as gu
from unetgeo.model.metric import UnetGeoMetric
from unetgeo.model.model_analysis_utils import ModelAnalysisUtils as mau
from unetgeo.model.model_optimizer import get_optimizer
from unetgeo.model.model_data_parsing import (gen_parses)
from unetgeo.model.model_utils import RelationTaggerUtils as rtu

from UNET.model_unet import UNetModel


class RelationTagger(pl.LightningModule):
    def __init__(self, hparam, tparam, iparam, path_data_folder, verbose=False):
        """ """
        super().__init__()

        self.hparam = hparam
        self.tparam = tparam
        self.iparam = iparam
        self.verbose = verbose

        self.task = hparam.task
        self.task_lan = hparam.task_lan
        self.fields = hparam.fields
        self.field_rs = hparam.field_representers
        self.n_fields = len(hparam.fields)
        self.name = hparam.model_name
        self.n_classes = hparam.n_classes
        self.weights = hparam.weights
        self.angle = hparam.angle

        self.optimizer_type = tparam.optimizer_type
        self.loss_type = tparam.loss_type

        self.spade_metric = UnetGeoMetric(1, dist_sync_on_step=False)

    def forward(
        self,
        data_ids,
        img_paths,
        img_512_paths,
        mask_512_paths
    ):

        def get_trained_UNet_model():
            img_height, img_width, img_channel = gu.get_img_size(img_512_paths[0])

            if img_channel != None:
                new_channel = img_channel
            else:
                new_channel = 1

            if self.weights.trained:
                model1 = UNetModel(self.n_classes, img_height, img_width, img_channel)
                model1.compile(optimizer=self.optimizer_type, loss=self.loss_type, metrics=['accuracy'])
                trained_model_path = self.weights.path
                model1.load_weights(trained_model_path)
                return model1
            else:
                raise NotImplementedError
        
        self.UNet_model = get_trained_UNet_model()
        # return UNet_model

    def meta_run(self, mode, batch):
        # 1. Batchwise collection of features
        (
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
        ) = mu.collect_features_batchwise(batch)

        batch_data_in = (
            data_ids,
            img_paths,
            img_512_paths,
            mask_512_paths
        )

        rtu.load_trained_model(self, batch_data_in) 

        if labels[0] is not None:
            # Generate gt parse
            f_parses, f_parse_box_ids, f_parse_head_ids = gen_parses(
                self.task,
                self.fields,
                texts,
                labels,
                l_max_gen=self.hparam.l_max_gen_of_each_parse
            )
        else:
            parses = [None] * len(texts)
            f_parses = [None] * len(texts)
            text_unit_field_labels = [None] * len(texts)
            f_parse_box_ids = [None] * len(texts)
        
        features = {
            "data_ids": data_ids,
            "img_paths": img_paths,
            "img_512_paths": img_512_paths,
            "mask_512_paths": mask_512_paths,
            "texts": texts,
            "labels": labels, 
            "f_parse_box_ids": f_parse_box_ids,
            "coords": coords,
            "coord_centers": coord_centers
        }
        return features

    def training_step(self, batch, batch_idx):
        s_set_batch = batch["s_set"]
        q_set_batch = batch["q_set"] 
        
        s_features = self.meta_run("s_set", s_set_batch)
        q_features = self.meta_run("q_set", q_set_batch)
        results = self.get_output_of_qSet_vs_sSet(s_features, q_features, "train")

        for b in range(len(results["pr_parses"])): 
            dir_path = Path("./data/results")
            file_name = f"{self.task}_{b}_file.json"
            rtu.save_pr_json(dir_path,file_name, results["pr_parses"][b])
            print("\n")
            pprint(results["pr_parses"][b])
            # cat json.file | jq -cr '.[]'
    
        (
            precision_edge_avg,
            recall_edge_avg,
            f1_edge_avg,
            precision_parse,
            recall_parse,
            f1_parse,
            acc,
        ) = self.spade_metric.compute()
        self.spade_metric.reset()

        print("\n")
        print("Entity Recognition & Entity labeling; Acc:", "{:.1f}".format(acc), "\n")
        print("Link pred. between key-value Entities; F1:", "{:.1f}".format(f1_parse), "\n")

        exit()

    def get_output_of_qSet_vs_sSet(self, s_features, q_features, mode):
        results = {}
        # Support set features
        s_data_ids = s_features["data_ids"]
        s_img_paths = s_features["img_paths"]
        s_img_512_paths = s_features["img_512_paths"] 
        s_mask_512_paths = s_features["mask_512_paths"] 
        s_texts = s_features["texts"]
        s_labels = s_features["labels"]
        s_f_parse_box_ids = s_features["f_parse_box_ids"]
        s_coords = s_features["coords"]
        s_coord_centers = s_features["coord_centers"]
        
        # Query set features
        q_data_ids = q_features["data_ids"]
        q_img_paths = q_features["img_paths"]
        q_img_512_paths = q_features["img_512_paths"]
        q_mask_512_paths = q_features["mask_512_paths"]
        q_texts = q_features["texts"]
        q_labels = q_features["labels"]
        q_f_parse_box_ids = q_features["f_parse_box_ids"]
        q_coords = q_features["coords"]
        q_coord_centers = q_features["coord_centers"]     
 
        (
            imgs, 
            masks, 
            pred_masks
        )= mau.get_imgs_masks_and_predict_masks(self.UNet_model, q_img_512_paths, q_mask_512_paths)

        (
            h_bboxes, 
            q_bboxes, 
            o_bboxes
        )= mau.get_pred_bboxes_for_each_classes(imgs, pred_masks)

        (
            h_seg_bbox_idxs, 
            q_seg_bbox_idxs, 
            o_seg_bbox_idxs, 
            v_bbox_idxs, 
        )= mau.get_OCR_bbox_idxs_matched_to_UNet_bboxes(q_coords, h_bboxes, q_bboxes, o_bboxes, q_img_paths)

        (   header_segments,
            key_segments,
            value_segments,
            other_segments,
            pr_link_TkeyIdxs_to_TvalIdxs,
            v_list_not_seen_in_angle
        )= mau.pr_v_seg_and_link_using_Geometric(
            q_coord_centers,
            h_seg_bbox_idxs,
            q_seg_bbox_idxs,
            o_seg_bbox_idxs,
            v_bbox_idxs,
            q_texts, 
            self.angle
        )

        link_direction = mau.get_direction_of_key_to_value_by_s_set(
                len(self.fields),
                s_labels, 
                s_coord_centers
            )

        if mode == "train":
            pr_key_to_value_mat = mau.pr_key_value_link(
                pr_link_TkeyIdxs_to_TvalIdxs,
                link_direction,
                q_coord_centers,
                len(self.fields),
                q_labels
            )

            pr_seg_idx_to_its_t_id = mau.map_segments_to_target_idx(
                header_segments, 
                key_segments, 
                value_segments, 
                other_segments,
                v_list_not_seen_in_angle
            )

            (
                gt_header_segment, 
                gt_key_segment, 
                gt_value_segment, 
                gt_other_segment
            ) = mau.get_gt_key_value_segments(q_f_parse_box_ids)

            gt_seg_idx_to_its_t_id = mau.map_segments_to_target_idx(
                gt_header_segment, 
                gt_key_segment, 
                gt_value_segment, 
                gt_other_segment,
                []
            )

            accs = mau.drawImage_and_print_acc(
                pr_seg_idx_to_its_t_id,
                gt_seg_idx_to_its_t_id, 
                q_coords, self.task, 
                q_data_ids, 
                q_img_paths, 
                q_texts
            )
            
            # This segment prediction mat is not accurate. Hence, accs is used
            pr_segment_mat = mau.predict_segment_mat(
                header_segments, 
                key_segments, 
                value_segments, 
                other_segments, 
                q_texts,
                self.fields,
                q_labels
            ) 
            for batch in range(len(q_texts)):
                # compute F1 for segmentation 
                tp_edge_s, fn_edge_s, fp_edge_s = au.my_cal_tp_fn_fp_of_edges(q_labels[batch][0], pr_segment_mat[batch])
                # F1 for link between key-value
                tp_edge_l, fn_edge_l, fp_edge_l = au.my_cal_tp_fn_fp_of_edges(q_labels[batch][1], pr_key_to_value_mat[batch])

                self.spade_metric.update(tp_edge_s, fp_edge_s, fn_edge_s, tp_edge_l, fp_edge_l, fn_edge_l, accs[batch])
            
            results["test"] = None 
        elif mode == "predict":
                results["test"] = None  
        else: 
            raise NotImplementedError 
        
        # Parsing 
        parse_list = mau.parse_segments_to_json(pr_link_TkeyIdxs_to_TvalIdxs, key_segments, value_segments, q_texts)
        results["pr_parses"] = parse_list

        return results

    def training_epoch_end(self, outputs) -> None:
        if gu.get_local_rank() == 0:
            print(f"Epoch {self.current_epoch} \n")

    def configure_optimizers(self):
        optimizer = get_optimizer(self.tparam, self)
        return {"optimizer": optimizer}

    @rank_zero_only
    @gu.timeit
    def predict_step(self, batch, batch_idx, dataloader_idx):
        # dataloader_idx == 0 is s_set, dataloader_idx == 1 is q_set
        if dataloader_idx == 0:
            features = self.meta_run("s_set", batch)
        elif dataloader_idx == 1:
            features = self.meta_run("q_set", batch)
        assert len(features["data_ids"]) == 1

        return features

    # TODO geometric-based is not a learned model
    # def validation_step(self, batch, batch_idx):
    #     pass

    # def validation_epoch_end(self, outputs):
    #     pass

    # def test_step(self, batch, batch_idx):
    #     q_features = self.meta_run("q_set", batch)
    #     return q_features

    # @rank_zero_only
    # def test_epoch_end(self, outputs: List[Any]) -> None:
    #     pass
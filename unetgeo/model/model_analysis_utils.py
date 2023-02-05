import numpy as np
import torch
from torch.linalg import norm

from configs.label_enum import Label
from pathlib import Path
from PIL import Image, ImageDraw

import cv2
from matplotlib import pyplot as plt
from keras.utils import normalize

from skimage.measure import label, regionprops

from shapely import Polygon


class KMeans:
    def __init__(self, max_iter = 1):
        self.max_iter = max_iter

    def initialize(self, best_key_ids_as_Kmeans, n_center_word, angle):
        self.angle = angle
        self.target_ids = []
        self.fixed_t_ids = sorted(best_key_ids_as_Kmeans)
        self.id_to_t_key_id_dic = {f"{k_id}": k_id for k_id in self.fixed_t_ids}
        # self.id_to_t_key_id_dic = {'any_token_id': target_key_id}
        self.cluster_idx_dic = {f"{k_id}": [] for k_id in self.fixed_t_ids}

        self.n_center_word = n_center_word
        self.v_idx_not_visible_in_angle = []
    
    def fit(self, ids_not_in_target, mode, text):
        for iter in range(self.max_iter):
            for not_t_id in ids_not_in_target:
                target_ids = list(set(self.target_ids + self.fixed_t_ids))

                X_dist = self.get_euc_dist_on_x(self.n_center_word[not_t_id], self.n_center_word[target_ids])
                Y_dist = self.get_euc_dist_on_y(self.n_center_word[not_t_id], self.n_center_word[target_ids])
                k_v_angle = self.get_angle_vectorization(self.n_center_word[target_ids], self.n_center_word[not_t_id])
                
                # Reducing degree will cause that could not see a word in which is very close to the based word
                angle_scope = self.angle  # 20 degree   
                #words_in_angle_scope = [[i, (x + 4*y)] for i, (a, x, y) in enumerate(zip(k_v_angle, X_dist, Y_dist)) if a < angle_scope]
                words_in_angle_scope = [[i, (2*a + x + 4*y)] for i, (a, x, y) in enumerate(zip(k_v_angle, X_dist, Y_dist)) if a < angle_scope]
                if len(words_in_angle_scope) > 0:
                    # sort by Euclidean distance
                    words_in_angle_scope.sort(key=lambda x: x[1])
                    min_idx_in_list = words_in_angle_scope[0][0] # First index is the minimum
                    min_id = target_ids[min_idx_in_list]

                    if min_id in self.fixed_t_ids:
                        self.cluster_idx_dic[f"{min_id}"].append([not_t_id])
                        self.id_to_t_key_id_dic[f"{not_t_id}"] = min_id
                        self.target_ids.append(not_t_id)
                    else:
                        fixed_id = self.id_to_t_key_id_dic[f"{min_id}"]
                        for ele_list in self.cluster_idx_dic[f"{fixed_id}"]:
                            if min_id in ele_list:
                                ele_list.append(not_t_id)
                        
                        self.id_to_t_key_id_dic[f"{not_t_id}"] = fixed_id
                        self.target_ids[self.target_ids.index(min_id)] = not_t_id

                else:
                    self.v_idx_not_visible_in_angle.append(not_t_id)
                    min_idx = 0
                    continue

            if mode == "key":
                for k in self.cluster_idx_dic:
                    if len(self.cluster_idx_dic[k]) == 0:
                        self.cluster_idx_dic[k].insert(0, [int(k)])
                    else:
                        self.cluster_idx_dic[k][0].insert(0, int(k))

        return self.cluster_idx_dic, self.v_idx_not_visible_in_angle
    
    def convert_dic_to_segments(self, cluster_dic):
        clusters = []
        for k in cluster_dic:
            clusters += cluster_dic[k]
        return clusters

    def get_euc_dist_on_y(self, vec1, vec2):
        # compute euclidean distance based on Y (not x) values
        if isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
            return torch.sqrt((vec1[1]- vec2[:, 1])**2)
        return np.sqrt((vec1[1]- vec2[:, 1])**2)
    
    def get_euc_dist_on_x(self, vec1, vec2):
        # compute euclidean distance based on x (not y) values
        if isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
            return torch.sqrt((vec1[0]- vec2[:,0])**2)
        return np.sqrt((vec1[0]- vec2[:,0])**2)

    def get_angle(self, vec1, vec2):
        # Important vec1 must be the left side of vec2 
        base_vec = vec1
        y1 , y2 = base_vec[1], vec2[1]
        x1, x2 = base_vec[0], vec2[0]
        tangent = ((y2 - y1)/(x2 - x1)) if (x2 - x1) != 0.0 else torch.tensor(float("Inf"))
        angle = torch.arctan(torch.abs(tangent))  # Take the absulute value of Radian
        return angle
    
    def get_angle_vectorization(self, vec1, vec2):
        # Important vec1 must be the left side of vec2 
        base_vec = vec1
        Y1 , y2 = base_vec[:, 1], vec2[1]
        X1, x2 = base_vec[:, 0], vec2[0]
        # finding slope/tangent
        t_numerator = y2 - Y1
        t_denuminator = x2 - X1
    
        angles = []
        for (t_n, t_dn) in zip(t_numerator, t_denuminator):
            if t_dn == 0.0:
                tangent = torch.tensor(float("Inf"))
            else:
                tangent = (t_n)/(t_dn)
            angle = torch.arctan(torch.abs(tangent))  # Take the absulute value of Radian
            angles.append(angle)
        return angles


class ModelAnalysisUtils:

    @classmethod
    def get_imgs_masks_and_predict_masks(cls, UNet_model, q_img_512_paths, q_mask_512_paths):
        imgs = []
        masks = []
        for b in range(len(q_img_512_paths)):
            img1 = cv2.imread(q_img_512_paths[b])
            img2 = cv2.imread(q_img_512_paths[b])
            imgs.append(img1)

            mask1 = cv2.imread(q_mask_512_paths[b])
            mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY) # remove channel axis
            mask2 = cv2.imread(q_mask_512_paths[b])
            mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
            masks.append(mask1)

        imgs = np.asarray(imgs)
        imgs = normalize(imgs, axis=1)
        preds = UNet_model.predict(imgs)      # (b_s, height, width, n_classes) 
        pred_masks = np.argmax(preds, axis=3) # (b_s, height, width)

        test_image = imgs[0]
        test_mask = masks[0]
        pred_mask = pred_masks[0]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))

        ax1.set_title("Test image")
        ax2.set_title("Test label")
        ax3.set_title("Pred test label")
        ax1.imshow(test_image[:,:,0], cmap='gray')
        ax2.imshow(test_mask, cmap='jet')
        ax3.imshow(pred_mask, cmap='jet')
        plt.show()

        return imgs, masks, pred_masks

    @classmethod
    def get_pred_bboxes_for_each_classes(cls, imgs, pred_masks):
        def get_bboxes(img1, props1, color):
            bboxes = []
            for prop in props1:
                # [xmin, ymin, xmax, ymax]
                box = [prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]]
                bboxes.append(box)
                cv2.rectangle(img1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), color, 2)
            return bboxes
        
        def plot_bboxes(img1, img_cp1, pred_mask_cp1):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
            ax1.set_title('Image')
            ax2.set_title('Pred Mask')
            ax3.set_title('Derived bounding box from Pred Mask')
            ax1.imshow(img1[:,:,0], cmap='gray')
            ax2.imshow(pred_mask_cp1, cmap='jet')
            ax3.imshow(img_cp1, cmap='jet')
            plt.show()


        classes = {"BG": 0, "header": 1, "question": 2, "other": 3}
        b_length = imgs.shape[0]

        batchwise_h_bboxes = []
        batchwise_q_bboxes = []
        batchwise_o_bboxes = []

        for b in range(b_length):
            img_copy1 = imgs[b].copy()
            for class1 in classes:
                # We dont create bboxes for background
                if classes[class1] == classes["BG"]:
                    continue
                
                pred_mask_class1 = pred_masks[b].copy()
                # Make the pred_mask_class1 as binary of integer, either a class or background
                pred_mask_class1 = np.where(pred_mask_class1 ==  classes[class1], classes[class1], classes["BG"])
                
                lbl_1 = label(pred_mask_class1)
                props1 = regionprops(lbl_1)

                if class1 == "header":
                    h_bboxes = get_bboxes(img_copy1, props1, (0,0, 255))
                elif class1 == "question":
                    q_bboxes = get_bboxes(img_copy1, props1, (255,255,0))
                elif class1 == "other":
                    o_bboxes = get_bboxes(img_copy1, props1, (255,0,0))
                else:
                    # This type of class is not implmented
                    raise NotImplementedError
            plot_bboxes(imgs[b], img_copy1, pred_masks[b])
            cls.clean_pred_bboxes([h_bboxes, q_bboxes, o_bboxes])
            #cls.plot_bboxes(imgs[b], pred_masks[b], {"header": h_bboxes, "question": q_bboxes, "other": o_bboxes})

            cls.clean_pred_bboxes([h_bboxes, q_bboxes, o_bboxes])
            #cls.plot_bboxes(imgs[b], pred_masks[b], {"header": h_bboxes, "question": q_bboxes, "other": o_bboxes})

            cls.clean_pred_bboxes([h_bboxes, q_bboxes, o_bboxes])
            cls.plot_bboxes(imgs[b], pred_masks[b], {"header": h_bboxes, "question": q_bboxes, "other": o_bboxes})

            batchwise_h_bboxes.append(h_bboxes)
            batchwise_q_bboxes.append(q_bboxes)
            batchwise_o_bboxes.append(o_bboxes)
        return batchwise_h_bboxes, batchwise_q_bboxes, batchwise_o_bboxes        
        
    @classmethod
    def clean_pred_bboxes(cls, list_bboxes):
        # Remove if a bbox is too small
        # Correct duplicate bboxes and change to the one with more pixels (area)
        # Note: the lists are passed by reference, no need to return 
        list1 = list_bboxes[0]
        for list2 in list_bboxes[1:]:
            cls.box_cleaning_process(list1, list2)

        if len(list_bboxes[1:]) >= 2:
            cls.clean_pred_bboxes(list_bboxes[1:])
        
    @classmethod
    def box_cleaning_process(cls, boxes1, boxes2):
        for i, box1 in enumerate(boxes1):
            poly1 = cls.get_polygon(box1)
            if poly1.area <= 10.0:
                boxes1.pop(i)
                continue
            #print(poly1.area)
            for j, box2 in enumerate(boxes2):
                poly2 = cls.get_polygon(box2)

                if poly2.area <= 10.0:
                    boxes2.pop(j)
                    continue
                
                if poly2.intersects(poly1):
                    boxes2.pop(j)
                    continue

                #print(poly2.area)
                if poly1.covers(poly2):
                    #if poly1.area < poly2.area:
                    boxes2.pop(j)

                    #print("header is poped")
                    #else:
                    #boxes2.pop(j)
                    #print("question is poped")
                    print(f"cover: {j}")
                    continue
                    
                
                # if poly2.covers(poly1):
        
                #     #if poly1.area < poly2.area:
                #     boxes2.pop(j)
                #     #print("header is poped")
                #     #else:
                #     #print("question is poped")

                if poly1.overlaps(poly2):
                    print(f"overlaps: {i}")
            

                if poly1.touches(poly2):
                    boxes2.pop(j)
                    print(f"touches: {j}")
                    continue

    @classmethod
    def get_polygon(cls, bbox):
        (xmin, ymin, xmax, ymax) = bbox
        return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
                
    @classmethod
    def plot_bboxes(cls, img, pred_mask_cp1, dict_bboxes):
        img_cp1 = img.copy()
        for dict_key in (dict_bboxes):
            if dict_key == "header":
                color = (0,0, 255)
            elif dict_key == "question":
                color = (255,255,0)
            else:
                color = (255,0,0)

            for box1 in dict_bboxes[dict_key]:
                cv2.rectangle(img_cp1, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
        ax1.set_title('Image')
        ax2.set_title('Pred Mask')
        ax3.set_title('Clean bounding box from Pred Mask')
        ax1.imshow(img[:,:,0], cmap='gray')
        ax2.imshow(pred_mask_cp1, cmap='jet')
        ax3.imshow(img_cp1, cmap='jet')
        plt.show()

    @classmethod
    def get_OCR_bbox_idxs_matched_to_UNet_bboxes(cls, q_coords, h_bboxes, q_bboxes, o_bboxes, q_img_paths):
        def scale_ocr_coord(coord1, height, width, resized_height=512, resized_width=512):
            (xmin, ymin, xmax, ymax) = coord1
            h_scale = resized_height / height
            w_scale = resized_width / width
            new_xmin = xmin * w_scale
            new_ymin = ymin * h_scale
            new_xmax = xmax * w_scale
            new_ymax = ymax * h_scale

            return [new_xmin, new_ymin, new_xmax, new_ymax]
        def is_ocr_and_UNet_bboxes_intersect(scaled_q_bbox1, UNet_bbox1):
            scaled_poly1 = cls.get_polygon(scaled_q_bbox1)
            UNet_poly1 = cls.get_polygon(UNet_bbox1)
            if scaled_poly1.intersects(UNet_poly1):
                return True
            return False

        batch_h_seg_bbox_idxs = []
        batch_q_seg_bbox_idxs = []
        batch_o_seg_bbox_idxs = []
        batch_v_bbox_idxs = []

        for b in range(len(q_coords)):
            h_seg_bbox_idxs = []
            q_seg_bbox_idxs = []
            o_seg_bbox_idxs = []
            v_bbox_idxs = []
            # TODO this code should be imporoved, in order to handle many/less lebels
            dict_bboxes = {"header": h_bboxes[b], "question": q_bboxes[b], "other": o_bboxes[b]}
            for dict_key in dict_bboxes:
                for counter in dict_bboxes[dict_key]:
                    if dict_key == "header":
                        h_seg_bbox_idxs.append([])
                    elif dict_key == "question":
                        q_seg_bbox_idxs.append([])
                    elif dict_key == "other":
                        o_seg_bbox_idxs.append([])
                    else:
                        raise NotImplementedError
                    
            # Get the image dimension to scale bboxes, in order to match the resized (512) image bboxes
            orig_img = cv2.imread(q_img_paths[b])
            img_height, img_width = orig_img.shape[0], orig_img.shape[1]

            for ocr_coord_idx, q_coord1 in enumerate(q_coords[b]):
                q_coord1 = cls.convert_coordinate_to_xmin_ymin_xmax_ymax(q_coord1)
                scaled_q_bbox1 = scale_ocr_coord(q_coord1, img_height, img_width)
                is_intersect = False
                for dict_key in dict_bboxes:
                    for class1_seg_idx, UNet_bbox1 in enumerate(dict_bboxes[dict_key]):
                        
                        if is_ocr_and_UNet_bboxes_intersect(scaled_q_bbox1, UNet_bbox1):

                            if dict_key == "header":
                                h_seg_bbox_idxs[class1_seg_idx].append(ocr_coord_idx)
                                is_intersect = True
                                break
                            elif dict_key == "question":
                                q_seg_bbox_idxs[class1_seg_idx].append(ocr_coord_idx)
                                is_intersect = True
                                break
                            elif dict_key == "other":                            
                                o_seg_bbox_idxs[class1_seg_idx].append(ocr_coord_idx)
                                is_intersect = True
                                break
                            else: 
                                raise NotImplementedError

                        continue
                    
                    if is_intersect:
                        break
                
                if is_intersect == False:
                    v_bbox_idxs.append(ocr_coord_idx)             
            
            # Remove empty sublist for each list, and make the first token as reprentitive/target of that segment
            batch_h_seg_bbox_idxs.append({f"{seg_idxs[0]}": [seg_idxs] for seg_idxs in h_seg_bbox_idxs if seg_idxs })
            batch_q_seg_bbox_idxs.append({f"{seg_idxs[0]}": [seg_idxs] for seg_idxs in q_seg_bbox_idxs if seg_idxs })
            batch_o_seg_bbox_idxs.append({f"{seg_idxs[0]}": [seg_idxs] for seg_idxs in o_seg_bbox_idxs if seg_idxs })
            batch_v_bbox_idxs.append(v_bbox_idxs)
        return batch_h_seg_bbox_idxs, batch_q_seg_bbox_idxs, batch_o_seg_bbox_idxs, batch_v_bbox_idxs,

    @classmethod
    def pr_v_seg_and_link_using_Geometric(
        cls, 
        q_coord_centers,
        h_seg_bbox_idxs,
        q_seg_bbox_idxs,
        o_seg_bbox_idxs,
        v_bbox_idxs,
        q_texts,
        angle

        ):
        header_list = []
        key_list = []
        other_list = []
        value_list = []
        pr_link_list = []
        v_list_not_seen_in_angle = []

        for b in range(len(q_texts)):
            pr_link_TkeyIdxs_to_TvalIdxs = {} # {"key_id": [val_id1, val_id2]} 

            # Take the first token as representive/target of that segment
            t_key_idx = [int(idx) for idx in list(q_seg_bbox_idxs[b].keys())]
            # remove t_value_idx in value_idxs
            #key_idxs_not_in_target = [k_seg[0] for k_seg in q_bbox_idxs[b]]
            # value_idxs_not_in_target = [v_id for v_id in value_idxs[b] if v_id not in t_value_idx[b]]
            #value_idxs_not_in_target = value_idxs[b]

            num_iterations = 1     
            # For key
            k_kmean = KMeans(num_iterations)
            #k_kmean.initialize(t_key_idx[b], n_center_word[b])
            #k_cluster_ids_dic = k_kmean.fit(key_idxs_not_in_target,"key", q_text[b])

            # For Value
            v_kmean = KMeans(num_iterations)
            v_kmean.initialize(t_key_idx, q_coord_centers[b], angle)
            v_cluster_ids_dic, v_idx_not_visible_in_angle = v_kmean.fit(v_bbox_idxs[b],"value", q_texts[b])

            for t_k_idx in v_cluster_ids_dic:
                val_clusters = v_cluster_ids_dic[t_k_idx]
                if len(val_clusters) > 0:
                    t_val_idxs = []
                    for cluster in val_clusters:
                        t_val_idxs.append(cluster[0])
                    pr_link_TkeyIdxs_to_TvalIdxs[f"{t_k_idx}"] = t_val_idxs
                else:
                    pr_link_TkeyIdxs_to_TvalIdxs[f"{t_k_idx}"] = []

            # append each batch
            pr_link_list.append(pr_link_TkeyIdxs_to_TvalIdxs)
            key_list.append(k_kmean.convert_dic_to_segments(q_seg_bbox_idxs[b]))
            header_list.append(k_kmean.convert_dic_to_segments(h_seg_bbox_idxs[b]))
            other_list.append(k_kmean.convert_dic_to_segments(o_seg_bbox_idxs[b])) 
            value_list.append(v_kmean.convert_dic_to_segments(v_cluster_ids_dic))

            v_list_not_seen_in_angle.append(v_idx_not_visible_in_angle)

        return header_list, key_list, value_list, other_list, pr_link_list, v_list_not_seen_in_angle    

    @classmethod
    def get_euc_dist_on_x(cls, vec1, vec2):
        # compute euclidean distance based on x (not y) values
        if isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
            return torch.sqrt((vec1[0]- vec2[0])**2)
        return np.sqrt((vec1[0]- vec2[0])**2)
    
    @classmethod
    def get_direction_of_key_to_value_by_s_set(cls, row_offset, s_label, s_n_center_word):
        is_horizontal = True
        direction_list = []
        for batch in range(len(s_n_center_word)):
            s_label[batch][1] # 1 is Entity linkin labels

            # Just randomly take a row and find whether link between key and value is horizontal or vertical
            # Ex: First row
            key_idx = row_offset - row_offset
            first_row = s_label[batch][1][row_offset, :]
            value_idx = int(torch.where(first_row == 1)[0])

            key_x, key_y = s_n_center_word[batch][key_idx][0], s_n_center_word[batch][key_idx][1] 
            value_x, value_y = s_n_center_word[batch][value_idx][0], s_n_center_word[batch][value_idx][1]

            if (value_x - key_x) > (value_y - key_y):
                is_horizontal = True
            else:
                is_horizontal = False
            direction_list.append({
                "is_horizontal": is_horizontal
            })
        return direction_list

    @classmethod
    def pr_key_value_link(cls, pr_TkeyIdxs_to_TvalIdxs, link_direction, q_n_center_word, row_offset, q_labels):
        label_list = []
        for b in range(len(q_n_center_word)):
            num_rows = len(q_n_center_word[b]) + row_offset
            num_cols = len(q_n_center_word[b])
            pr_entity_linking_mat = torch.zeros(num_rows , num_cols, dtype=int).type_as(q_labels[b][1])
            
            if link_direction[0]["is_horizontal"]:
                # Compute Euclidean distance of each value id with all key ids
                for k_id in pr_TkeyIdxs_to_TvalIdxs[b]:
                    v_ids = pr_TkeyIdxs_to_TvalIdxs[b][k_id]
                    for v_id in v_ids:
                        if isinstance(int(k_id), int) and isinstance(v_id, int):
                            pr_entity_linking_mat[int(k_id) + row_offset, v_id] = 1
            else: 
                raise NotImplementedError

            label_list.append(pr_entity_linking_mat)
        return label_list
    
    @classmethod
    def predict_segment_mat(cls, header_segments, key_segments, value_segments, other_segments, q_text, fields, q_labels):
        row_offset = len(fields)
        pr_segment_mat_list = []
        for batch in range(len(q_text)):
            num_rows = len(q_text[batch]) + row_offset
            num_cols = len(q_text[batch])
            pr_segment_mat = torch.zeros(num_rows, num_cols, dtype=int).type_as(q_labels[batch][0])

            # Insert header indicies to pr_segment_mat 
            for header1 in  header_segments[batch]:
                for i_idx, h_id in enumerate(header1):
                    if i_idx == 0:
                        pr_segment_mat[fields.index(Label.HEADER.value), h_id] = 1
                    else: 
                        pr_segment_mat[h_id + row_offset - 1 , h_id] = 1

            # Insert key indicies to pr_segment_mat 
            for keys1 in  key_segments[batch]:
                for i_idx, k_id in enumerate(keys1):
                    if i_idx == 0:
                        pr_segment_mat[fields.index(Label.QUESTION.value), k_id] = 1
                    else: 
                        pr_segment_mat[k_id + row_offset - 1 , k_id] = 1

            # Insert value indicies to pr_segment_mat
            for values1 in value_segments[batch]:
                for i_idx, val_id in enumerate(values1):
                    if i_idx == 0:
                        pr_segment_mat[fields.index(Label.ANSWER.value), val_id] = 1
                    else:
                        pr_segment_mat[val_id + row_offset -1, val_id] = 1

            # Insert other indicies to pr_segment_mat
            for other1 in other_segments[batch]:
                for i_idx, other_id in enumerate(other1):
                    if i_idx == 0:
                        pr_segment_mat[fields.index(Label.OTHER.value), other_id] = 1
                    else:
                        pr_segment_mat[other_id + row_offset -1, other_id] = 1
            
            pr_segment_mat_list.append(pr_segment_mat)
        return pr_segment_mat_list

    @classmethod
    def map_segments_to_target_idx(
            cls, 
            header_segments, 
            key_segments, 
            value_segments, 
            other_segments, 
            v_list_not_seen_in_angle = []
        ):
        # map segment idxs to its first segment idx, since the first segment idx
        # is the target idx of that segment
        # [5,6] => {"5": "qa.question_5", "6": "qa.question_5"}

        mapped_list = []
        
        for batch in range(len(key_segments)):
            mag_seg_idx_to_its_t_id = {} # {"idx": "qa.question_t_id"}

            for segs in header_segments[batch]:
                seg_dic = cls.map_segment_to_t_idx(segs, Label.HEADER.value)
                mag_seg_idx_to_its_t_id.update(seg_dic)

            for segs in key_segments[batch]: 
                seg_dic = cls.map_segment_to_t_idx(segs, Label.QUESTION.value)
                mag_seg_idx_to_its_t_id.update(seg_dic)
            
            for segs in value_segments[batch]:
                seg_dic = cls.map_segment_to_t_idx(segs, Label.ANSWER.value)
                mag_seg_idx_to_its_t_id.update(seg_dic)

            for segs in other_segments[batch]:
                seg_dic = cls.map_segment_to_t_idx(segs, Label.OTHER.value)
                mag_seg_idx_to_its_t_id.update(seg_dic)

            if len(v_list_not_seen_in_angle) > 0:
                # list of value idx that are not visible by key angle
                v_segs = v_list_not_seen_in_angle[batch]
                map_v_seg_idx_to_v_idx = {}
                for v_idx in v_segs:
                    value = f"{Label.ANSWER.value}_{v_idx}"
                    map_v_seg_idx_to_v_idx[f"{v_idx}"] = value
                mag_seg_idx_to_its_t_id.update(map_v_seg_idx_to_v_idx)

            mapped_list.append(mag_seg_idx_to_its_t_id)
        return mapped_list
            
    @classmethod
    def map_segment_to_t_idx(cls, segments, label):
        t_idx = 0
        map_seg_idx_to_t_idx = {}
        for counter, idx in enumerate(segments):
            if counter == 0:
                t_idx = idx 

            if Label.QUESTION.value == label:
                value = f"{Label.QUESTION.value}_{t_idx}"
            elif Label.ANSWER.value == label:
                value = f"{Label.ANSWER.value}_{t_idx}"
            elif Label.HEADER.value == label:
                value = f"{Label.HEADER.value}_{t_idx}"
            elif Label.OTHER.value == label:
                value = f"{Label.OTHER.value}_{t_idx}"
            else:
                raise NotImplementedError

            map_seg_idx_to_t_idx[f"{idx}"] = value
        return map_seg_idx_to_t_idx
    
    @classmethod
    def get_gt_key_value_segments(cls, q_f_parse_box_ids):
        header_seg_list = []
        key_seg_list = []
        value_seg_list = []
        other_seg_list = []
        for batch in range(len(q_f_parse_box_ids)):
            header_seg = []
            key_seg = []
            value_seg = []
            other_seg = []
            for ele in q_f_parse_box_ids[batch]:
                label = list(ele.keys())[0]
                if label == Label.HEADER.value:
                    header_seg.append(list(ele.values())[0])
                elif label == Label.QUESTION.value:
                    key_seg.append(list(ele.values())[0])
                elif label == Label.ANSWER.value:
                    value_seg.append(list(ele.values())[0])
                elif label == Label.OTHER.value:
                    other_seg.append(list(ele.values())[0])
                else: 
                    raise NotImplementedError
            
            header_seg_list.append(header_seg)
            key_seg_list.append(key_seg)
            value_seg_list.append(value_seg)
            other_seg_list.append(other_seg)
        return header_seg_list, key_seg_list, value_seg_list, other_seg_list

    @classmethod
    def drawImage_and_print_acc(cls, pr_mapped_dict, gt_mapped_dict, coord, task, q_data_ids, q_img_path, q_text):
        acc_list = []

        for b in range(len(gt_mapped_dict)):
            if task == "funsd":
                path = Path(q_img_path[b])
                image = Image.open(path).convert("RGB")
                width_scale  = image.width/image.width
                height_scale = image.height/image.height
                draw = ImageDraw.Draw(image)

            tp = 0
            gt_idxs = list(gt_mapped_dict[b].keys())
            for idx in gt_idxs:
                gt_t_idx, gt_label = cls.get_t_idx_and_label(gt_mapped_dict[b][idx])
                pr_t_idx, pr_label = cls.get_t_idx_and_label(pr_mapped_dict[b][idx])

                if (gt_label == Label.HEADER.value) and (gt_label == pr_label) and (gt_t_idx ==  pr_t_idx):
                    tp += 1
                    outline = (0,0, 255)
                elif (gt_label == Label.QUESTION.value) and (gt_label == pr_label) and (gt_t_idx ==  pr_t_idx):
                    tp += 1
                    outline = (255,255,0)
                elif (gt_label == Label.ANSWER.value) and (gt_label == pr_label) and (gt_t_idx == pr_t_idx):
                    tp += 1
                    outline = (124,252,0)
                elif (gt_label == Label.OTHER.value) and (gt_label == pr_label) and (gt_t_idx ==  pr_t_idx):
                    outline = (255,0,0)
                else:
                    outline = (255,165,0)

                if task == "funsd":
                    coord1 = cls.convert_coordinate_to_xmin_ymin_xmax_ymax(coord[b][int(idx)])
                    draw.rectangle(
                        [
                            coord1[0] * width_scale, 
                            coord1[1] * height_scale,
                            coord1[2] * width_scale, 
                            coord1[3] * height_scale
                        ],
                        outline= outline, 
                        width=2
                    )
            if task == "funsd":
                save_path = Path(f"./data/results/{q_data_ids[b]}.png")
                image.save(save_path) 

            # Accuracy
            total =  len(pr_mapped_dict[b])
            acc_list.append((tp / total) * 100)
        return acc_list
                           
    @classmethod
    def get_t_idx_and_label(cls, label_and_idx):
        val_list = str(label_and_idx).split("_")
        label = val_list[0]
        t_idx = val_list[1]
        return t_idx, label

    @classmethod
    def convert_coordinate_to_xmin_ymin_xmax_ymax(cls, coord1):
        coord1 = coord1.cpu().numpy()
        all_x = coord1[:, 0]
        all_y = coord1[:, 1]
        xmin = np.min(all_x)
        xmax = np.max(all_x)
        ymin = np.min(all_y)
        ymax = np.max(all_y)
        return [xmin, ymin, xmax, ymax]
    
    @classmethod
    def parse_segments_to_json(cls, pr_link_TkeyIdxs_to_TvalIdxs, key_segments, value_segments, text):
        parse_list = []
        for b in range(len(pr_link_TkeyIdxs_to_TvalIdxs)):
            parse = []
            for t_k in pr_link_TkeyIdxs_to_TvalIdxs[b]:
                t_v_list = pr_link_TkeyIdxs_to_TvalIdxs[b][t_k]

                # Search in key segments
                for k_seg in key_segments[b]:
                    if int(t_k) in k_seg:
                        k_words = ""
                        for kidx in k_seg:
                            k_words += text[b][kidx] + " "
                        break
                           
                sib_v_words = []
                if len(t_v_list) > 0:
                    for t_v in t_v_list:
                        # Search in value segments
                        for v_seg in value_segments[b]:
                            if t_v in v_seg:
                                v_words = ""
                                for vidx in v_seg:
                                    v_words += text[b][vidx] + " " 
                                break
                        sib_v_words.append(v_words.strip())
                else: 
                    sib_v_words.append("None")
                parse.append({k_words.strip(): sib_v_words})

            parse_list.append(parse)  
        return parse_list 
                
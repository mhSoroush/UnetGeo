import random as python_random
import numpy as np
from unetgeo.utils.data_augmentation_utils import image_rotation, image_warping
import torch


def gen_augmented_coord(
    img, coord, img_sz, coord_aug_params, clip_box_coord, normalize_amp=False
):
    if img is None:
        width_and_height = img_sz["width"], img_sz["height"]
    else:
        width_and_height = img.shape[1], img.shape[0]

    if type(coord_aug_params[0]) == list:
        param_w1, param_w2, param_r = coord_aug_params
        n_min1, n_max1, amp_min1, amp_max1 = param_w1
        n_min2, n_max2, amp_min2, amp_max2 = param_w2
        angle_min, angle_max = param_r
    else:
        n_min, n_max, amp_min, amp_max, angle_min, angle_max = coord_aug_params
        n_min1, n_max1, amp_min1, amp_max1 = n_min, n_max, amp_min, amp_max
        n_min2, n_max2, amp_min2, amp_max2 = n_min, n_max, amp_min, amp_max

    n = python_random.uniform(n_min1, n_max1)
    n2 = python_random.uniform(n_min2, n_max2)

    amp = python_random.uniform(amp_min1, amp_max1)
    amp2 = python_random.uniform(amp_min2, amp_max2)

    angle = python_random.uniform(angle_min, angle_max)

    # random switching of amp (when min max both are positive
    r = python_random.random()
    amp = amp if r > 0.5 else -amp
    r2 = python_random.random()
    amp2 = amp2 if r2 > 0.5 else -amp2

    img_w, nboxes_w = image_warping(
        img,
        width_and_height,
        coord,
        clip_box_coord,
        n,
        amp,
        direction=0,
        normalize_amp=normalize_amp,
    )

    img_w2, nboxes_w2 = image_warping(
        img_w,
        width_and_height,
        nboxes_w,
        clip_box_coord,
        n2,
        amp2,
        direction=1,
        normalize_amp=normalize_amp,
    )

    img_r, nboxes_r = image_rotation(
        img_w2, width_and_height, nboxes_w2, clip_box_coord, angle
    )

    return img_r, nboxes_r


def gen_augmented_text_tok1(token_pool, text_tok1, token_aug_params):
    """
    Example
    text_tok1 = ['App', "##le", "Juice"]
    """
    p_del, p_subs, p_insert, p_tail_insert, n_max_insert = token_aug_params
    cum_ps = np.cumsum([p_del, p_subs, p_insert])
    new_text_tok1 = []
    for token in text_tok1:
        r_middle = python_random.random()
        if r_middle < cum_ps[0]:
            # delete
            pass
        elif r_middle < cum_ps[1]:
            # replace (substitute)
            new_token = python_random.choice(token_pool)
            new_text_tok1.append(new_token)
        elif r_middle < cum_ps[2]:
            # insert in front of token
            n_target = python_random.randint(1, n_max_insert)
            new_tokens = python_random.choices(token_pool, k=n_target)
            new_text_tok1.extend(new_tokens)
        else:
            new_text_tok1.append(token)

        r_tail = python_random.random()
        # if new_text_tok1 is empty, insert new token always
        if r_tail < p_tail_insert or len(new_text_tok1) == 0:
            # insert new tokens.
            n_target = python_random.randint(1, n_max_insert)
            new_tokens = python_random.choices(token_pool, k=n_target)
            new_text_tok1.extend(new_tokens)

    return new_text_tok1
    
def get_label_and_feature(raw_data1, task, fields, field_rs):
    if task in  ["funsd", "lehrvertrag"]:
        label, feature = get_adj_mat_funsd(fields, field_rs, raw_data1)
    else:
        raise NotImplementedError

    return label, feature

def get_label_and_feature_infer_mode(task, raw_data1):
    label = None
    if task in ["funsd", "lehrvertrag"]:
        feature = [
            (x["text"], x["boundingBox"], x["isVertical"], x["confidence"])
            for x in raw_data1["words"]
        ]
        confidence = [x[3] for x in feature]
        img_sz = raw_data1["meta"]["img_size"][0]
        data_id = raw_data1["meta"]["image_id"]
        img_path = raw_data1["meta"]["image_path"]
        img_512_path = raw_data1["meta"]["image_512_path"]
        mask_512_path = raw_data1["meta"]["mask_512_path"]
    else:
        raise NotImplementedError

    return label, feature, confidence, img_sz, data_id, img_path, img_512_path, mask_512_path

def get_meta_feature(task, raw_data1, feature):
    if task in ["funsd", "lehrvertrag"]:
        img_sz = raw_data1["meta"]["image_size"]
        confidence = [1] * len(feature)
        data_id = raw_data1["meta"]["image_id"]
        img_path = raw_data1["meta"]["image_path"]
        img_512_path = raw_data1["meta"]["image_512_path"]
        mask_512_path = raw_data1["meta"]["mask_512_path"]
    else:
        raise NotImplementedError

    return img_sz, confidence, data_id, img_path, img_512_path, mask_512_path

def funsd_box_to_coord(box):
    x1, y1, x3, y3 = box
    x2, y2 = x3, y1
    x4, y4 = x1, y3

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

def get_some_stat_from_form_checking_data_structure(form):
    l_box_at_each_form = [len(x["words"]) for x in form]
    l_word = sum(l_box_at_each_form)

    id_to_label_ori = {}
    form_id_to_first_col_id = {}
    i_col = -1
    for i_form, form1 in enumerate(form):
        id = form1["id"]
        # assert i_form == id  # assume id is well-sorted.
        label_ori = form1["label"]

        id_to_label_ori[id] = label_ori

        for i_word1, word1 in enumerate(form1["words"]):
            i_col += 1
            if i_word1 == 0:
                x1_before = word1["box"][0]
                y1_before = word1["box"][1]
                form_id_to_first_col_id[id] = i_col
            else:
                x1_now = word1["box"][0]
                y1_now = word1["box"][1]
                # if x1_now < x1_before:
                #     assert y1_now > y1_before

    return l_word, id_to_label_ori, form_id_to_first_col_id

def get_adj_mat_funsd(fields, field_rs, raw_data1):
    """ """

    # cols = (text, coord, is_vertical (0 or 1))

    map_label_ori_to_field = {
        "answer": "qa.answer",
        "question": "qa.question",
        "header": "header.header",
        "other": "other.other",
    }

    form = raw_data1["form"]

    (
        l_word,
        id_to_label_ori,
        form_id_to_first_col_id,
    ) = get_some_stat_from_form_checking_data_structure(form)

    n_field = len(fields)
    row_offset = n_field

    cols = []
    adj_mat_fg = np.zeros([2, n_field + l_word, l_word], dtype=np.int)
    i_col = -1
    for i_form, form1 in enumerate(form):
        word = form1["words"]
        linking = form1["linking"]  # [ [], [], ...]
        label_ori_target = form1["label"]
        field_target = map_label_ori_to_field[label_ori_target]

        fid_target = fields.index(field_target)
        id_target = form1["id"]

        # check_assumption_on_linking(linking, id_target, id_to_label_ori)

        # rel-g construction
        for linking1 in linking:
            hid, tid = linking1
            hcid = form_id_to_first_col_id[hid]
            tcid = form_id_to_first_col_id[tid]
            adj_mat_fg[1, row_offset + hcid, tcid] = 1

        # rel-s construction
        # assume words are sorted along x coordinate
        for i_word1, word1 in enumerate(word):
            i_col += 1
            is_vertical = 0
            coord1 = funsd_box_to_coord(word1["box"])

            if i_word1 == 0:
                adj_mat_fg[0, fid_target, i_col] = 1
            else:
                adj_mat_fg[0, row_offset + i_col - 1, i_col] = 1

            l_word1 = len(word1["text"])
            text1 = word1["text"] if l_word1 > 0 else "[UNK]"

            col = (text1, coord1, is_vertical, id_target)
            cols.append(col)

    return adj_mat_fg, cols

def get_direction_vec(coord1, vertical1):
    c1, c2, c3, c4 = coord1
    # x1, y1 = c1
    # x2, y2 = c2
    # x3, y3 = c3
    # x4, y4 = c4

    if vertical1:
        direction_vec = (c3 + c4) / 2 - (c1 + c2) / 2
    else:
        direction_vec = (c2 + c3) / 2 - (c1 + c4) / 2

    return direction_vec

def augment_vertical(vertical1, l_tok1):
    return [vertical1] * l_tok1

def augment_char_size(csz, l_tok1):
    return [csz] * l_tok1

def get_coord1_first_char(coord1, dvec, vertical1, n_char11_offset, n_char11):
    c1, c2, c3, c4 = coord1

    if vertical1:
        new_c1 = c1 + dvec * n_char11_offset
        new_c2 = c2 + dvec * n_char11_offset
        new_c3 = c2 + dvec * n_char11
        new_c4 = c1 + dvec * n_char11
    else:
        new_c1 = c1 + dvec * n_char11_offset
        new_c4 = c4 + dvec * n_char11_offset
        new_c2 = c1 + dvec * n_char11
        new_c3 = c4 + dvec * n_char11
        # new_c3 = c1 + dvec * n_char11

    return [new_c1, new_c2, new_c3, new_c4]

def augment_coord(coord1, vertical1, l_tok1, method, text_tok1):
    """

    Args:
        coord1: numpy ndarray
        vertical1: bool
        l_tok1: numeric
        bag_of_words: bool

    Returns:

    """
    direction_vec = get_direction_vec(coord1, vertical1)
    direction_vecs = [direction_vec] * l_tok1

    if method == "bag_of_words":
        coord_tok1 = [coord1] * l_tok1
    elif method == "equal_division":
        """each token as if has single char width"""
        coord_tok1 = []
        dvec_tok = direction_vec / l_tok1
        coord1_fc = get_coord1_first_char(
            coord1, dvec_tok, vertical1, n_char11_offset=0, n_char11=1
        )

        for i in range(l_tok1):
            tok_pos = coord1_fc + i * dvec_tok
            coord_tok1.append(tok_pos.tolist())

    elif method == "char_lv_equal_division":
        coord_tok1 = []
        text_tok1_no_sharp = [xx.replace("#", "") for xx in text_tok1]
        n_char_text_tok1_no_sharp = [len(xx) for xx in text_tok1_no_sharp]

        # original #### token cause the problem, thus add single length for it
        for i_nchar, n_char11 in enumerate(n_char_text_tok1_no_sharp):
            if n_char11 == 0:
                n_char_text_tok1_no_sharp[i_nchar] = 1

        n_char_text_tok1_no_sharp_cumsum = np.cumsum(n_char_text_tok1_no_sharp).tolist()
        l_char1 = n_char_text_tok1_no_sharp_cumsum[-1]
        dvec = direction_vec / l_char1
        n_char11_before = 0
        for i, n_char11 in enumerate(n_char_text_tok1_no_sharp_cumsum):
            tok_pos = np.array(
                get_coord1_first_char(
                    coord1,
                    dvec,
                    vertical1,
                    n_char11_offset=n_char11_before,
                    n_char11=n_char11,
                )
            )
            n_char11_before = n_char11
            coord_tok1.append(tok_pos.tolist())
    else:
        raise NotImplementedError

    return coord_tok1, direction_vecs

def get_char_size1(coord1, v):
    """
    if not vetical
    c1          c2
    c4          c3


    if vetical
    c1  c2



    c4  c3
    """
    c1, c2, c3, c4 = coord1
    if v:
        csz = (np.linalg.norm(c1 - c2) + np.linalg.norm(c3 - c4)) / 2
    else:
        csz = (np.linalg.norm(c1 - c4) + np.linalg.norm(c2 - c3)) / 2

    return csz

def remove_target(list_in, l_str):
    tf_arr = np.array(l_str) != 0
    list_out = np.array(list_in)[tf_arr].tolist()
    return list_out

def remove_blank_box(text, coord, vertical):
    l_str = [len(x.strip()) for x in text]

    text = remove_target(text, l_str)
    coord = remove_target(coord, l_str)
    vertical = remove_target(vertical, l_str)

    return text, coord, vertical

def char_height_normalization(n_char_unit, char_height):
    unit_len = np.min([x for x in char_height if x > 0])
    lb = 0
    ub = n_char_unit
    new_arr = np.clip(char_height / unit_len, lb, ub)
    return new_arr.astype(np.int)

def feature_normalization(token_features):
    token_features =  token_features.squeeze(0) # remove the batch dim
    nrmzd_feature = []
    for f100 in token_features:
        nrmzd_f100 = [torch.exp(f) for f in f100]
        t_f100 = torch.stack(nrmzd_f100, dim=0)
        total = torch.sum(t_f100)
        t_f100 = t_f100/total
        nrmzd_feature.append(t_f100)

    nrmz_token_f = torch.stack(nrmzd_feature, dim=0)
    return nrmz_token_f.unsqueeze(0)
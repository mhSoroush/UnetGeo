import numpy as np
import torch
import torch.nn.functional as F

import unetgeo.utils.general_utils as gu

def gen_parses(
    task,
    fields,
    text_units,
    label_units,
    l_max_gen,
):
    """
    f_parses: a list of serialized fields
    g_parses: a list of grouped serialized fields
    """
    f_label_idx = 0
    g_label_idx = 1
    
    # 1. Generate serialized individual fields
    label_fs = [label[f_label_idx] for label in label_units]
    f_parses, f_parse_box_ids, f_parse_head_ids = gen_f_parses(
        task, fields, text_units, label_fs, l_max_gen
    )

    return f_parses, f_parse_box_ids, f_parse_head_ids

def gen_f_parses(task, fields, text_units, label_fs, l_max_gen):
    parses = []
    parse_box_ids = []
    parse_head_ids = []

    label_2_json = {}

    target_relation = 1  # Currently only single type in rel-s. 0 for no relation.
    row_offset = len(fields)

    for b, (text_unit, label_f) in enumerate(zip(text_units, label_fs)):
        # for each batch
        parse = []
        parse_box_id = []
        parse_head_id = []
        rel_mat = (
            label_f.cpu() if isinstance(label_f, torch.Tensor) else label_f
        )  # edge
        for i, field in enumerate(fields):
            row = np.array(rel_mat[i])
            # 1. Seeding: Find seed nodes for each field type.
            idx_nz = np.where(row == target_relation)[0]  # non-zero
            for i_nz in idx_nz:
                # 2. Serialization: from seed nodes, generate boxes recursively.
                boxes = [text_unit[i_nz]]
                box_ids = [i_nz]
                boxes, box_ids = gen_boxes_single_path(
                    boxes,
                    box_ids,
                    i_nz,
                    text_unit,
                    rel_mat,
                    row_offset,
                    target_relation,
                    l_max_gen,
                )

                parse.append({field: boxes})
                box_ids = np.array(box_ids).tolist()
                parse_box_id.append({field: box_ids})
                parse_head_id.append(i_nz)

        parses.append(refine_parse(task, parse))
        parse_box_ids.append(parse_box_id)
        parse_head_ids.append(parse_head_id)

    return parses, parse_box_ids, parse_head_ids

def gen_boxes_single_path(
    boxes, box_ids, col_idx, text_unit, rel_mat, row_offset, target_relation, l_max_gen
):
    row = np.array(rel_mat[col_idx + row_offset])
    next_col_idxs = np.where(row == target_relation)[0]
    if next_col_idxs.size > 0 and len(boxes) < l_max_gen:
        assert next_col_idxs.size == 1
        next_col_idx = next_col_idxs[0]
        boxes += [text_unit[next_col_idx]]
        box_ids += [next_col_idx]
        return gen_boxes_single_path(
            boxes,
            box_ids,
            next_col_idx,
            text_unit,
            rel_mat,
            row_offset,
            target_relation,
            l_max_gen,
        )
    else:
        return boxes, box_ids

def refine_parse(task, parse: list):
    if task == "namecard" or "receipt_v1":
        new_parse = []
        for parse1 in parse:
            assert len(parse1) == 1
            for k, v in parse1.items():
                # new_parse1 = {k: ''.join(v)}
                new_parse1 = {k: " ".join(v)}

            new_parse.append(new_parse1)
    else:
        raise NotImplementedError
    return new_parse

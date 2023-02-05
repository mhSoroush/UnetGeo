from copy import deepcopy

import numpy as np
import torch

def get_tp_fn_fp_edge(gt_mat, pr_mat):
    # calculate
    # tp: true positive
    # fn: false negative
    # fp: false positive
    idx_gtt = gt_mat == 1  # gt true
    idx_gtf = gt_mat == 0  # gt false
    tp1 = sum(pr_mat[idx_gtt] == 1)
    fn1 = sum(pr_mat[idx_gtt] == 0)
    fp1 = sum(pr_mat[idx_gtf] == 1)

    return tp1, fn1, fp1

def my_cal_tp_fn_fp_of_edges(labels, pr_labels):
    """
    Args:
        labels: [row, col]
        pr_labels: [row, col]

    # tp: true positive
    # fn: false negative
    # fp: false positive

    """
    gt_mat = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    pr_mat = pr_labels.cpu().numpy() if isinstance(pr_labels, torch.Tensor) else pr_labels

    tp1, fn1, fp1 = get_tp_fn_fp_edge(gt_mat, pr_mat)

    return tp1, fn1, fp1

def my_cal_p_r_f1(tp, fn, fp):
    p = tp / ((tp + fp) if (tp + fp) != 0.0 else np.Inf ) 
    r = tp / ((tp + fn) if (tp + fn) != 0.0 else np.Inf )

    f1_denominator = (p + r) if (p + r) != 0.0 else np.Inf
    f1 = ((2 * p * r) / f1_denominator)
    return p, r, f1

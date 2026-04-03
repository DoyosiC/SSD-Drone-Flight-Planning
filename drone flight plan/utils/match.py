"""
https://github.com/amdegroot/ssd.pytorch
のbox_utils.pyより使用
関数matchを行うファイル

本章の実装はGitHub：amdegroot/ssd.pytorch [4] を参考にしています。
MIT License
Copyright (c) 2017 Max deGroot, Ellis Brown

"""

import torch

def point_form(boxes):
    """
    Convert prior_boxes to (xmin, ymin, xmax, ymax) representation
    for comparison to point form ground truth data.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax

def center_size(boxes):
    """
    Convert boxes to (cx, cy, w, h) representation.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,   # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)      # w, h

def intersect(box_a, box_b):
    """
    Compute the area of intersection between each pair of box_a and box_b.
    box_a: [A,4], box_b: [B,4]
    Returns: [A,B]
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = (max_xy - min_xy).clamp(min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap (IoU) of two sets of boxes.
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    """
    # 1. Jaccard overlap (IoU) [num_objects, num_priors]
    overlaps = jaccard(truths, point_form(priors))

    # 2. Bipartite Matching
    # Best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # Best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    # 3. Squeeze unnecessary dimensions
    best_truth_idx = best_truth_idx.squeeze(0)
    best_truth_overlap = best_truth_overlap.squeeze(0)
    best_prior_idx = best_prior_idx.squeeze(1)
    best_prior_overlap = best_prior_overlap.squeeze(1)

    # 4. Ensure best prior
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # 5. Ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors], background=0
    conf[best_truth_overlap < threshold] = 0  # label as background

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf

def encode(matched, priors, variances):
    """
    Encode the variances from the priorbox layers into the ground truth boxes.
    matched: [num_priors, 4]
    priors: [num_priors, 4]
    variances: [2] or list
    Returns: [num_priors, 4]
    """
    # (cx, cy, w, h)
    priors_cx = priors[:, :2]
    priors_wh = priors[:, 2:]
    matched_cx = (matched[:, :2] + matched[:, 2:]) / 2
    matched_wh = matched[:, 2:] - matched[:, :2]

    g_cxcy = (matched_cx - priors_cx) / (variances[0] * priors_wh)
    g_wh = torch.log(matched_wh / priors_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)

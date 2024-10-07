'''
@file: eval_utils.py
@breif: the evaluation tools functions
@author: Winter
@update: 2023.10.10
'''
import numpy as np
from collections import defaultdict
from typing import Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc

from .scan_utils import local2global

def evaluate(gt_annos: list, dt_annos: list, iou: float=0.5) -> dict:
    '''
    Do evaluation for detection result.

    Parameters
    ----------
    gt_annos: np.ndarray
        ground truth annotations for many frames.
    dt_annos: np.ndarray
        detection annotations for many frames.
    iou: float
        IoU threshold
    
    Return
    ----------
    result: dict
        evaluation result
    '''
    assert len(gt_annos) == len(dt_annos), "The number of frames between \
        Ground-Truth and Detection must be equal."

    frames_num = len(gt_annos)
    frames_dt_xy, frames_dt_cls, frames_dt_idx = [], [], []
    frames_gt_xy, frames_gt_idx = [], []

    # prepare data
    counter = 0
    for i in range(frames_num):
        if len(dt_annos[i]["xy"]) > 0:
            assert len(dt_annos[i]["xy"]) == len(dt_annos[i]["scores"]), \
                "The number of scores and x-y coordinate must match."
            frames_dt_xy  += dt_annos[i]["xy"]
            frames_dt_cls += dt_annos[i]["scores"]
            frames_dt_idx += [counter] * len(dt_annos[i]["xy"])
        
        if len(gt_annos[i]["xy"]) > 0:
            frames_gt_xy  += gt_annos[i]["xy"]
            frames_gt_idx += [counter] * len(gt_annos[i]["xy"])
        
        counter += 1

    frames_dt_xy  = np.array(frames_dt_xy, dtype=np.float32)    # [D, 2]
    frames_dt_cls = np.array(frames_dt_cls, dtype=np.float32)   # [D, 1]
    frames_dt_idx = np.array(frames_dt_idx, dtype=np.int32)     # [D, 1]
    frames_gt_xy  = np.array(frames_gt_xy, dtype=np.float32)    # [G, 2]
    frames_gt_idx = np.array(frames_gt_idx, dtype=np.int32)     # [G, 1]

    result = calPrecisionRecall(frames_dt_xy, frames_dt_cls, frames_dt_idx,
        frames_gt_xy, frames_gt_idx, iou)

    print(
        f"AP_{iou} {result['AP']:5f}\n"
        f"peak-F1_{iou} {result['peak_F1']:5f}\n"
        f"EER_{iou} {result['EER']:5f}\n"
    )

    return result

def calPrecisionRecall(dt_xys: np.ndarray, dt_scores: np.ndarray, dt_idxs: np.ndarray,
    gt_xys: np.ndarray, gt_idxs: np.ndarray, iou: float, criterion: str="radius") -> dict:
    '''
    Calculate precision and recall as well as some other measurements.

    Parameters
    ----------
    dt_xys: np.ndarray
        detection (x, y) for many frames
    dt_scores: np.ndarray
        detection scores for many frames
    dt_idxs: np.ndarray
        detection frame id for many frames
    gt_xys: np.ndarray
        ground truth (x, y) for many frames
    gt_idxs: np.ndarray
        ground truth frame id for many frames
    iou: float
        IoU threshold
    criterion: str
        the method for IoU matrix calculation
    
    Return
    ----------
    result: dict
        evaluation result
    '''
    iou = iou * np.ones(len(gt_idxs), dtype=np.float32)
    frames_idx = np.unique(np.r_[dt_idxs, gt_idxs])

    dt_accepted_idxs = defaultdict(list)
    tps = np.zeros(len(frames_idx), dtype=np.uint32)
    fps = np.zeros(len(frames_idx), dtype=np.uint32)
    fns = np.array([np.sum(gt_idxs == f) for f in frames_idx], dtype=np.uint32)

    precs = np.full_like(dt_scores, np.nan)
    recs = np.full_like(dt_scores, np.nan)
    threshs = np.full_like(dt_scores, np.nan)

    # loop threshold
    indices = np.argsort(dt_scores, kind="mergesort")
    for i, idx in enumerate(reversed(indices)):
        frame_idx = dt_idxs[idx]
        frame_idx_ = np.where(frames_idx == frame_idx)[0][0]

        # accept detection
        dt_idx = dt_accepted_idxs[frame_idx]
        dt_idx.append(idx)
        threshs[i] = dt_scores[idx]
        dt_xy = dt_xys[dt_idx]

        # ground-truth respectively
        gts_mask = gt_idxs == frame_idx
        gt_xy = gt_xys[gts_mask]

        # No ground-truth, but there is a detection.
        if len(gt_xy) == 0:
            fps[frame_idx_] += 1
        # There is ground-truth and detection in this frame.
        else:
            # [N_gt, N_dt], each entry stands for the iou of gt_i and dt_j
            if criterion == "radius":
                # True if too far and False if may match 
                iou_matrix = iou[gts_mask, None] < cdist(gt_xy, dt_xy)
            else:
                raise NotImplementedError
            
            gt_i, dt_j = linear_sum_assignment(iou_matrix)
            tps[frame_idx_] = np.sum(np.logical_not(iou_matrix[gt_i, dt_j]))
            fps[frame_idx_] = len(dt_xy) - tps[frame_idx_]
            fns[frame_idx_] = len(gt_xy) - tps[frame_idx_]
        
        tp, fp, fn = np.sum(tps), np.sum(fps), np.sum(fns)
        precs[i] = tp / (fp + tp) if fp + tp > 0 else np.nan
        recs[i] = tp / (fn + tp) if fn + tp > 0 else np.nan

    # measurement calculation
    def peakF1(precs, recs):
        '''
        F1-number
        '''
        return np.max(2 * precs * recs / np.clip(precs + recs, 1e-16, 2 + 1e-16))

    def eer(precs, recs):
        '''
        Equal Error Rate
        '''
        p1 = np.where(precs != 0)[0][0]
        r1 = np.where(recs != 0)[0][0]
        idx = np.argmin(np.abs(precs[p1:] - recs[r1:]))
        return (precs[p1 + idx] + recs[r1 + idx]) / 2 

    def ap(precs, recs):
        '''
        Average precision (area under PR-curve)
        '''
        return auc(recs, precs)

    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(recs) >= 0) == len(recs) - 1
    return {
        "precisions": precs,
        "recalls": recs,
        "thresholds": threshs,
        "AP": ap(precs, recs),
        "peak_F1": peakF1(precs, recs),
        "EER": eer(precs, recs)
    }
    

def nms(scans: np.ndarray, phi: np.ndarray, pred_cls: np.ndarray, pred_reg: np.ndarray,
    min_dist: float=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Performing non-maximum suppression(NMS) on the predicted results.

    Parameters
    ----------
    scans: np.ndarray
        laser distance information
    phi: np.ndarray
        laser angle information
    pred_cls: np.ndarray
        prediction scores
    pred_reg: np.ndarray
        prediction regression
    min_dist: float
        the minimum threshold to NMS
    
    Return
    ----------
    det_xys: np.ndarray
        detection (x, y) for `scans` in global coordinate
    det_cls: np.ndarray
        detection confidence scores for `det_xys`
    instance_mask: np.ndarray
        NMS information
    '''
    assert len(pred_cls.shape) == 1
    _, num_pts = scans.shape

    pred_xs, pred_ys = local2global(scans, phi, pred_reg[:, 0], pred_reg[:, 1], cartesian=True)
    # sort prediction with descending confidence
    sort_inds = np.argsort(pred_cls)[::-1]
    pred_xs, pred_ys = pred_xs.T[sort_inds], pred_ys.T[sort_inds]
    pred_cls = pred_cls[sort_inds]

    # compute pair-wise distance
    xdiff = pred_xs.reshape(num_pts, 1) - pred_xs.reshape(1, num_pts)
    ydiff = pred_ys.reshape(num_pts, 1) - pred_ys.reshape(1, num_pts)
    p_dist = np.sqrt(np.square(xdiff) + np.square(ydiff))

    # nms
    keep = np.ones(num_pts, dtype=np.bool_)
    instance_mask = np.zeros(num_pts, dtype=np.int32)
    instance_id = 1
    for i in range(num_pts):
        if not keep[i]:
            continue

        dup_inds = p_dist[i] < min_dist
        keep[dup_inds] = False
        keep[i] = True
        instance_mask[sort_inds[dup_inds]] = instance_id
        instance_id += 1

    det_xys = np.stack((pred_xs, pred_ys), axis=1)[keep]
    det_cls = pred_cls[keep]
    
    return np.squeeze(det_xys), det_cls, instance_mask
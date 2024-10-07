'''
@file: matcher.py
@breif: Modules to compute the matching cost and solve the corresponding LSAP.
@author: Winter
@update: 2023.10.6
'''
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops.boxes import box_area

def boxIou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def gBoxIou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results, so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = boxIou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class HungarianMatcher(nn.Module):
    '''
    This class computes an assignment between the targets and the predictions of the network.
    @NOTE: For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Parameters
    ----------
    cost_labels: float
        This is the relative weight of the classification error in the matching cost
    cost_boxes: float
        This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
    cost_giou: float
        This is the relative weight of the giou loss of the bounding box in the matching cost
    '''
    def __init__(self, cost_labels: float = 1, cost_boxes: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_labels = cost_labels
        self.cost_boxes = cost_boxes
        self.cost_giou = cost_giou
        assert cost_labels != 0 or cost_boxes != 0 or cost_giou != 0, "all costs can not be zero"

    @torch.no_grad()
    def forward(self, outputs: dict, targets: dict) -> list:
        '''
        Performs the matching

        Parameters
        ----------
        outputs: dict
            pred_cls: Tensor of dim [batch_size, num_queries, 1] with the classification logits
            pred_reg: Tensor of dim [batch_size, num_queries, 2] with the predicted regression

        targets: dict
            target_cls: Tensor of dim [batch_size, scan_pts] (where 1 means the target and 0 means no-object)
            target_reg: Tensor of dim [num_obj, 2] containing the regression for target objects cross all batches
            target_boxes: Tensor of dim [num_obj, 4] containing the box coordinate for target objects cross all batches
            tgt_size: list of length batch_size, each element stands for the number of targets in one frame

        Return
        ----------
        index_list: list
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        '''
        batch_size, num_queries = outputs["pred_cls"].shape[:2]

        # scan box
        out_boxes = outputs["pred_boxes"].flatten(0, 1)
        tgt_boxes = targets["target_boxes"]
        
        # We flatten to compute the cost matrices in a batch
        out_cls = outputs["pred_cls"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, 1]
        out_reg = outputs["pred_reg"].flatten(0, 1)              # [batch_size * num_queries, 2]

        # annotation
        tgt_size  = targets["target_size"]
        tgt_reg   = targets["target_reg"]
        tgt_boxes = targets["target_boxes"]
        assert sum(tgt_size) == tgt_reg.shape[0] == tgt_boxes.shape[0], \
            "The number of targets must be aligned with the regression."

        # Compute the classification cost
        cost_labels = -out_cls

        # Compute the L1 cost between boxes
        cost_boxes = torch.cdist(out_reg, tgt_reg, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -gBoxIou(out_boxes, tgt_boxes)

        # Final cost matrix
        C = self.cost_boxes * cost_boxes + self.cost_labels * cost_labels + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(tgt_size, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


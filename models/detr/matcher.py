# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
Modified from: https://github.com/facebookresearch/detr
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcherAVA(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, binary_loss: bool = False, before: bool = False, clip_len: int = 32):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.binary_loss = binary_loss
        self.before = before
        self.clip_len = clip_len
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        # Also concat the target labels
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_bbox = tgt_bbox[:,1:]
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=False):
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            out_prob = outputs["pred_logits_b"].flatten(0, 1).softmax(-1)
            cost_class = -out_prob[:, 1:2].repeat(1, len(tgt_bbox))
            # Final cost matrix
            C = self.cost_bbox * cost_bbox+ self.cost_giou * cost_giou + self.cost_class * cost_class 
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class HungarianMatcherUCF(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, binary_loss: bool = False, before: bool = False, clip_len: int = 32):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.binary_loss = binary_loss
        self.before = before
        self.clip_len = clip_len
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, t, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, t, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        l = self.clip_len
        
        # for ease, assume a single batch case
        try:
            front_pad = targets[0]["front_pad"]
            end_pad = -targets[0]["end_pad"]
            if end_pad == 0:
                end_pad = None
        except: # in case of JHMDB
            front_pad = 0
            end_pad = None

        bs, t, num_queries, num_classes = outputs["pred_logits"].shape
        out_bbox = outputs["pred_boxes"][:,front_pad:end_pad,:,:].flatten(0,2) # bs*t*nq, 4

        # Also concat the target labels
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # bs*num_actors*t, 5
        tgt_bbox = tgt_bbox[:,1:].view(bs, -1, t, 4)[:,:,front_pad:end_pad,:]
        num_actors = tgt_bbox.size(1)
        num_valid_frames = tgt_bbox.size(2)
        tgt_bbox = tgt_bbox.transpose(1,2).contiguous()
        sizes = []
        tgt_bbox = tgt_bbox.flatten(0,2) # bs*t*num_actors, 4
        valid_tgt_bbox = []
        for i, box in enumerate(tgt_bbox):
            if i%num_actors == 0:
                sizes.append(0)
            if not ((box[1:] == 0.).all()):
                sizes[-1] += 1
                valid_tgt_bbox.append(box)
            else:
                pass
        try:
            valid_tgt_bbox = torch.stack(valid_tgt_bbox)
        except: # when there is no valid box
            return None
        
        num_valid_boxes = sum(sizes)
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, valid_tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(valid_tgt_bbox))
        out_prob = outputs["pred_logits_b"][:,front_pad:end_pad,:,:].flatten(0, 2).softmax(-1)
        cost_class = -out_prob[:, 1:2].repeat(1, num_valid_boxes)

        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class 

        C = C.view(bs*(num_valid_frames), num_queries, -1).cpu()
        # sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]    

class HungarianMatcherJHMDB(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, binary_loss: bool = False, before: bool = False, clip_len: int = 32):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.binary_loss = binary_loss
        self.before = before
        self.clip_len = clip_len
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, t, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, t, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        l = self.clip_len
        
        # for ease, assume a single batch case
        front_pad = targets[0]["front_pad"]
        end_pad = -targets[0]["end_pad"]
        if end_pad == 0:
            end_pad = None

        bs, t, num_queries, num_classes = outputs["pred_logits"].shape
        out_bbox = outputs["pred_boxes"][:,front_pad:end_pad,:,:].flatten(0,2) # bs*t*nq, 4

        # Also concat the target labels
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # bs*num_actors*t, 5
        tgt_bbox = tgt_bbox[:,1:].view(bs, -1, t, 4)[:,:,front_pad:end_pad,:]
        num_actors = tgt_bbox.size(1)
        num_valid_frames = tgt_bbox.size(2)
        tgt_bbox = tgt_bbox.transpose(1,2).contiguous()
        sizes = [1]*num_valid_frames
        tgt_bbox = tgt_bbox.flatten(0,2) # bs*t*num_actors, 4
        num_valid_boxes = sum(sizes)
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        out_prob = outputs["pred_logits_b"][:,front_pad:end_pad,:,:].flatten(0, 2).softmax(-1)
        cost_class = -out_prob[:, 1:2].repeat(1, num_valid_boxes)

        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class 

        C = C.view(bs*(num_valid_frames), num_queries, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]            


def build_matcher(cfg):
    if cfg.CONFIG.DATA.DATASET_NAME == "ava":
        return HungarianMatcherAVA(cost_class=cfg.CONFIG.MATCHER.COST_CLASS, cost_bbox=cfg.CONFIG.MATCHER.COST_BBOX, cost_giou=cfg.CONFIG.MATCHER.COST_GIOU, binary_loss=cfg.CONFIG.MATCHER.BNY_LOSS, before=cfg.CONFIG.MATCHER.BEFORE, clip_len=cfg.CONFIG.DATA.TEMP_LEN)
    elif cfg.CONFIG.DATA.DATASET_NAME == "ucf":
        return HungarianMatcherUCF(cost_class=cfg.CONFIG.MATCHER.COST_CLASS, cost_bbox=cfg.CONFIG.MATCHER.COST_BBOX, cost_giou=cfg.CONFIG.MATCHER.COST_GIOU, binary_loss=cfg.CONFIG.MATCHER.BNY_LOSS, before=cfg.CONFIG.MATCHER.BEFORE, clip_len=cfg.CONFIG.DATA.TEMP_LEN)
    elif cfg.CONFIG.DATA.DATASET_NAME == "jhmdb":
        return HungarianMatcherJHMDB(cost_class=cfg.CONFIG.MATCHER.COST_CLASS, cost_bbox=cfg.CONFIG.MATCHER.COST_BBOX, cost_giou=cfg.CONFIG.MATCHER.COST_GIOU, binary_loss=cfg.CONFIG.MATCHER.BNY_LOSS, before=cfg.CONFIG.MATCHER.BEFORE, clip_len=cfg.CONFIG.DATA.TEMP_LEN)
"""
Reference: https://github.com/MCG-NJU/STMixer
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# from models.transformer.util import box_ops
from .util import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_sigmoid, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from models.detr.segmentation import sigmoid_focal_loss, dice_loss


class SetCriterionAVA(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight, num_classes, num_queries, matcher, weight_dict, eos_coef, losses, data_file,
                 evaluation=False, label_smoothing_alpha=0.1):
        """ Create the criterion.
        Parameters
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight = weight
        self.evaluation = evaluation
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.data_file = data_file
        empty_weight = torch.ones(3)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.label_smoothing_alpha = 0.1

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        try:
            src_logits_b = outputs['pred_logits_b'] # bs nq 3
            target_classes_b = torch.full(src_logits_b.shape[:2], 2,
                                dtype=torch.int64, device=src_logits.device)
            target_classes_b[idx] = 1
            loss_ce_b = F.cross_entropy(src_logits_b.transpose(1, 2), target_classes_b, self.empty_weight.to(src_logits_b.device).to(src_logits_b.dtype))
        except Exception as e:
            breakpoint()
        src_logits_sig = src_logits.sigmoid()
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        n_p = max(target_classes_o.sum(), 1)
        target_classes_o_ = target_classes_o
        true_label = 1
        false_label = 0
        if self.label_smoothing_alpha:
            alpha = self.label_smoothing_alpha
            true_label = (1-alpha)*true_label + alpha/2
            false_label = (1-alpha)*false_label + alpha/2
            target_classes_o[target_classes_o == 0] = false_label
            target_classes_o[target_classes_o == 1] = true_label
        
        target_classes = torch.full(src_logits.shape, false_label,
                                    dtype=torch.float32, device=src_logits.device)
        # rebalance way 1:
        weights = torch.full(src_logits.shape[:2], 1,
                             dtype=torch.float32, device=src_logits.device)
        weights[idx] = self.weight

        weights = weights.view(weights.shape[0], weights.shape[1], 1)  # [:,:,None]
        target_classes[idx] = target_classes_o
        if self.evaluation:
            loss_ce = F.binary_cross_entropy(src_logits_sig, target_classes)
        else:
            loss_ce = sigmoid_focal_loss(src_logits, target_classes, weights) / n_p

        losses = {'loss_ce': loss_ce}
        try:
            losses['loss_ce_b'] = loss_ce_b
        except:
            pass
        if log:
            # docs this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy_sigmoid(src_logits[idx], target_classes_o_)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes[:, 1:]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # docs use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
             The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class SetCriterionUCF(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight, num_classes, num_queries, matcher, weight_dict, eos_coef, losses, data_file,
                 evaluation=False, label_smoothing_alpha=0.1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight = weight
        self.evaluation = evaluation
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.data_file = data_file
        self.label_smoothing_alpha = label_smoothing_alpha
        empty_weight_b = torch.ones(3)
        empty_weight_b[-1] = self.eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.register_buffer('empty_weight_b', empty_weight_b)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # bs, t, n, 24
        T = outputs['pred_logits'].size(1)
        empty_frame = False

        # for ease, assume a single batch case
        front_pad = targets[0]["front_pad"]
        end_pad = -targets[0]["end_pad"]
        if end_pad == 0:
            end_pad = None

        try:
            idx = self._get_src_permutation_idx(indices)
        except:
            empty_frame = True
            idx = T + 1

        src_logits_b = outputs['pred_logits_b'].flatten(0,1)# bs, t, n, 3
        target_classes_b = torch.full(src_logits_b.shape[:2], 2,
                            dtype=torch.int64, device=src_logits_b.device)   
        try:
            target_classes_b[front_pad:end_pad,:][idx] = 1
        except:
            pass

        loss_ce_b = F.cross_entropy(src_logits_b.transpose(1,2), target_classes_b, self.empty_weight_b.to(src_logits.device))
        
        if not empty_frame:
            target_classes_o = torch.cat([t["labels"] for t in targets])
            if target_classes_o.ndim == 1:
                target_classes_o = target_classes_o[None]
            target_classes_o = target_classes_o[:, front_pad:end_pad].transpose(0,1).contiguous().flatten() # t*num_actors
            target_classes_o = target_classes_o[target_classes_o != self.num_classes]
        src_logits = src_logits.flatten(0,1) # bs*t, n, 24
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # bs*t, n_a 
        if not empty_frame:
            target_classes[front_pad:end_pad,:][idx] = target_classes_o

        target_classes_onehot = F.one_hot(target_classes, self.num_classes+1).float()
        true_label = 1
        false_label = 0
        if self.label_smoothing_alpha:
            alpha = self.label_smoothing_alpha
            true_label = (1-alpha)*true_label + alpha/self.num_classes
            false_label = (1-alpha)*false_label + alpha/self.num_classes
            target_classes_onehot[target_classes_onehot == 0] = false_label
            target_classes_onehot[target_classes_onehot == 1] = true_label

        weights = torch.full(src_logits.shape[:2], 1,
                             dtype=torch.float32, device=src_logits.device)
        if not empty_frame:
            weights[idx] = self.weight
        weights = weights[..., None]
        new_src_logits = inverse_sigmoid(src_logits_b.softmax(-1)[..., 1:2] * src_logits.sigmoid())
        loss_ce = sigmoid_focal_loss(new_src_logits, target_classes_onehot[..., :-1], weights) / len(src_logits)

        losses = {'loss_ce': loss_ce}
        losses['loss_ce_b'] = loss_ce_b

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            if not empty_frame:
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            else:
                losses['class_error'] = 100 - accuracy(src_logits.flatten(0,1), target_classes.flatten(0,1))[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        try:
            idx = self._get_src_permutation_idx(indices)
        except:
            device = outputs["pred_boxes"].device
            loss = (0*outputs["pred_boxes"]).sum()
            return {'loss_bbox': loss,
                    'loss_giou': loss}

        front_pad = targets[0]["front_pad"]
        end_pad = -targets[0]["end_pad"]
        if end_pad == 0:
            end_pad = None        
        
        # idx[0]: range(bs*t)
        # idx[1]: the matched idx corresponds to idx[0]
        bs, T, num_queries = outputs['pred_boxes'].shape[:3]
        src_boxes = outputs['pred_boxes'][:,front_pad:end_pad].flatten(0,1)[idx]
        
        # for ease, assume a single batch case
        front_pad = targets[0]["front_pad"]
        end_pad = -targets[0]["end_pad"]
        if end_pad == 0:
            end_pad = None            

        target_boxes = torch.cat([t["boxes"] for t in targets])
        target_boxes = target_boxes[:,1:].view(bs, -1, T, 4)[:,:,front_pad:end_pad,:]
        
        num_actors = target_boxes.size(1)
        num_valid_frames = target_boxes.size(2)        
        target_boxes = target_boxes.transpose(1,2).contiguous()
        sizes = []
        valid_target_boxes = []
        target_boxes = target_boxes.flatten(0,2) # bs*t*num_actors, 4
        for i, box in enumerate(target_boxes):
            if i%num_actors == 0:
                sizes.append(0)
            if not (box[1:] == 0.).all():
                sizes[-1] += 1
                valid_target_boxes.append(box)
            else:
                pass        
        valid_target_boxes = torch.stack(valid_target_boxes)
        num_valid_boxes = sum(sizes)   

        loss_bbox = F.l1_loss(src_boxes, valid_target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_valid_boxes        

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(valid_target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_valid_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # docs use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # indices: list of length 40

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs_ in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs_, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs_, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class SetCriterionJHMDB(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight, num_classes, num_queries, matcher, weight_dict, eos_coef, losses, data_file,
                 evaluation=False, label_smoothing_alpha=0.1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight = weight
        self.evaluation = evaluation
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.data_file = data_file
        self.label_smoothing_alpha = label_smoothing_alpha
        empty_weight_b = torch.ones(3)
        empty_weight_b[-1] = self.eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.register_buffer('empty_weight_b', empty_weight_b)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # bs, t, n, 24
        T = outputs['pred_logits'].size(1)

        front_pad = targets[0]["front_pad"]
        end_pad = -targets[0]["end_pad"]
        valid_len = src_logits.size(1) - front_pad + end_pad
        if end_pad == 0:
            end_pad = None

        idx = self._get_src_permutation_idx(indices)
        src_logits_b = outputs['pred_logits_b'].flatten(0,1)# bs, t, n, 3
        target_classes_b = torch.full(src_logits_b.shape[:2], 2,
                            dtype=torch.int64, device=src_logits_b.device)   
        target_classes_b[front_pad:end_pad,:][idx] = 1
        loss_ce_b = F.cross_entropy(src_logits_b.transpose(1,2), target_classes_b, self.empty_weight_b.to(src_logits.device))
        
        target_classes_o = torch.cat([t["labels"] for t in targets])
        if target_classes_o.ndim == 1:
            target_classes_o = target_classes_o[None]
        target_classes_o = target_classes_o[:, front_pad:end_pad].transpose(0,1).contiguous().flatten() # t*num_actors
        target_classes_o = target_classes_o[target_classes_o != self.num_classes]
        src_logits = src_logits.flatten(0,1) # bs*t, n, 21
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # bs*t, n_a
        target_classes[front_pad:end_pad,:][idx] = target_classes_o

        target_classes_onehot = F.one_hot(target_classes, self.num_classes+1).float()
        true_label = 1
        false_label = 0
        if self.label_smoothing_alpha:
            alpha = self.label_smoothing_alpha
            true_label = (1-alpha)*true_label + alpha/self.num_classes
            false_label = (1-alpha)*false_label + alpha/self.num_classes
            target_classes_onehot[target_classes_onehot == 0] = false_label
            target_classes_onehot[target_classes_onehot == 1] = true_label

        weights = torch.full(src_logits.shape[:2], 1,
                             dtype=torch.float32, device=src_logits.device)
        weights[idx] = self.weight
        weights = weights[..., None][front_pad:end_pad]
        new_src_logits = inverse_sigmoid(src_logits_b.softmax(-1)[..., 1:2] * src_logits.sigmoid())
        loss_ce = sigmoid_focal_loss(new_src_logits[front_pad:end_pad], target_classes_onehot[..., :-1][front_pad:end_pad], weights) / valid_len

        losses = {'loss_ce': loss_ce}
        losses['loss_ce_b'] = loss_ce_b

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        
        front_pad = targets[0]["front_pad"]
        end_pad = -targets[0]["end_pad"]
        if end_pad == 0:
            end_pad = None        
        
        # idx[0]: range(bs*t)
        # idx[1]: the matched idx corresponds to idx[0]
        bs, T, num_queries = outputs['pred_boxes'].shape[:3]
        src_boxes = outputs['pred_boxes'][:,front_pad:end_pad,:].flatten(0,1)[idx]
        
        # for ease, assume a single batch case
        target_boxes = torch.cat([t["boxes"] for t in targets])
        target_boxes = target_boxes[:,1:].view(bs, -1, T, 4)[:,:,front_pad:end_pad,:]

        num_actors = target_boxes.size(1)
        num_valid_frames = target_boxes.size(2)        
        target_boxes = target_boxes.transpose(1,2).contiguous()
        sizes = [1]*num_valid_frames

        target_boxes = target_boxes.flatten(0,2) # bs*t*num_actors, 4
        num_valid_boxes = sum(sizes)
    
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_valid_boxes        

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_valid_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # docs use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # indices: list of length 40

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs_ in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs_, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs_, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class PostProcessAVA(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        try:
            out_logits_b, out_logits, out_bbox = outputs['pred_logits_b'], outputs['pred_logits'], outputs['pred_boxes']
        except:
            out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']


        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        scores = prob.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        try:
            output_b = out_logits_b.softmax(-1).detach().cpu().numpy()[..., 1:2]
            return scores, boxes, output_b
        except:
            return scores, boxes

class PostProcessUCF(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        try:
            out_logits, out_bbox, out_logits_b = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_logits_b']
        except:
            out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob_b = out_logits_b.softmax(-1)[..., 1:2]
        prob = inverse_sigmoid(out_logits.sigmoid() * prob_b).sigmoid()

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        scores = prob.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        try:
            output_b = out_logits_b.softmax(-1).detach().cpu().numpy()[..., 1:2]
            return scores, boxes, output_b
        except:
            return scores, boxes

class PostProcessJHMDB(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        try:
            out_logits, out_bbox, out_logits_b = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_logits_b']
        except:
            out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob_b = out_logits_b.softmax(-1)[..., 1:2]
        prob = inverse_sigmoid(out_logits.sigmoid() * prob_b).sigmoid()

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        scores = prob.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        try:
            output_b = out_logits_b.softmax(-1).detach().cpu().numpy()[..., 1:2]
            return scores, boxes, output_b
        except:
            return scores, boxes

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def build_criterion_and_postprocessor(cfg, matcher):
    weight_dict = {'loss_ce': cfg.CONFIG.LOSS_COFS.DICE_COF, 'loss_bbox': cfg.CONFIG.LOSS_COFS.BBOX_COF}
    weight_dict['loss_giou'] = cfg.CONFIG.LOSS_COFS.GIOU_COF
    weight_dict['loss_ce_b'] = cfg.CONFIG.LOSS_COFS.PERSON_COF
    losses = ['labels', 'boxes'] #, 'cardinality'
    if cfg.CONFIG.DATA.DATASET_NAME == 'ava':
        criterion = SetCriterionAVA(cfg.CONFIG.LOSS_COFS.WEIGHT,
                                    cfg.CONFIG.DATA.NUM_CLASSES,
                                    num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                    matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,
                                    losses=losses,
                                    data_file=cfg.CONFIG.DATA.DATASET_NAME,
                                    evaluation=cfg.CONFIG.EVAL_ONLY,
                                    label_smoothing_alpha=cfg.CONFIG.MODEL.LABEL_SMOOTHING_ALPHA,)
        postprocessors = {'bbox': PostProcessAVA()}
    elif cfg.CONFIG.DATA.DATASET_NAME == 'jhmdb':
        criterion = SetCriterionJHMDB(cfg.CONFIG.LOSS_COFS.WEIGHT,
                                    cfg.CONFIG.DATA.NUM_CLASSES,
                                    num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                    matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,                                    
                                    losses=losses,
                                    data_file=cfg.CONFIG.DATA.DATASET_NAME,
                                    evaluation=cfg.CONFIG.EVAL_ONLY,
                                    label_smoothing_alpha=cfg.CONFIG.MODEL.LABEL_SMOOTHING_ALPHA,)
        postprocessors = {'bbox': PostProcessJHMDB()}
    else:
        criterion = SetCriterionUCF(cfg.CONFIG.LOSS_COFS.WEIGHT,
                                    cfg.CONFIG.DATA.NUM_CLASSES,
                                    num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                                    matcher=matcher, weight_dict=weight_dict,
                                    eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,                                    
                                    losses=losses,
                                    data_file=cfg.CONFIG.DATA.DATASET_NAME,
                                    evaluation=cfg.CONFIG.EVAL_ONLY,
                                    label_smoothing_alpha=cfg.CONFIG.MODEL.LABEL_SMOOTHING_ALPHA,)        
        postprocessors = {'bbox': PostProcessUCF()}
    
    return criterion, postprocessors

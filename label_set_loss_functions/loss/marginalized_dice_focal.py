"""
@brief  Pytorch implementation of the mean marginalized Dice loss + marginalized Focal loss.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch
from torch.nn.modules.loss import _WeightedLoss
from label_set_loss_functions.convertor.marginalization import log_softmax_marginalize
from label_set_loss_functions.utils import check_label_set_map
from label_set_loss_functions.loss.mean_dice_loss import EPSILON


class MarginalizedDiceFocalLoss(_WeightedLoss):
    def __init__(self, gamma=2., weight=None, squared=True, reduction='mean', labels_superset_map=None):
        """
        :param gamma: (float) value of the exponent gamma in the definition
            of the Focal loss.
        :param weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
        :param squared: bool. If True, squared the terms in the denominator of the Dice loss.
        :param labels_superset_map: dict to handle superclasses
        :param reduction: str.
        """
        super(MarginalizedDiceFocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma
        self.squared = squared
        check_label_set_map(labels_superset_map)
        self.labels_superset_map = labels_superset_map

    def _prepare_data(self, input_batch, target):
        # Flatten the predicted class score map and the segmentation ground-truth
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))
        # Compute the marginalization
        log_pred_proba, target_proba = log_softmax_marginalize(
            flat_input=flat_input,
            flat_target=flat_target,
            labels_superset_map=self.labels_superset_map,
        )
        return log_pred_proba, target_proba

    def forward(self, input, target):
        logpt, target_proba = self._prepare_data(input, target) # b,c,s

        # Get the proba
        pt = torch.exp(logpt)  # b,c,s

        # Compute the batch of marginalized Focal loss values
        if self.weight is not None:
            if self.weight.type() != logpt.data.type():
                self.weight = self.weight.type_as(logpt.data)
            # Expand the weight to a map of shape b,c,s
            w_class = self.weight[None, :, None] # c => 1,c,1
            w_class = w_class.expand((logpt.size(0), -1, logpt.size(2)))  # 1,c,1 => b,c,s
            logpt = logpt * w_class
        weight = torch.pow(-pt + 1., self.gamma)
        # Sum over classes and mean over voxels
        per_voxel_focal_loss = torch.sum(-weight * target_proba * logpt, dim=1)  # b,s
        focal_loss = torch.mean(per_voxel_focal_loss, dim=1)  # b,

        # Compute the batch marginalized mean Dice loss
        num = pt * target_proba  # b,c,s --p*g
        num = torch.sum(num, dim=2)  # b,c
        if self.squared:
            den1 = torch.sum(pt * pt, dim=2)  # b,c
            den2 = torch.sum(target_proba * target_proba, dim=2)  # b,c
        else:
            den1 = torch.sum(pt, dim=2)  # b,c
            den2 = torch.sum(target_proba, dim=2)  # b,c
        dice = (2. * num) / (den1 + den2 + EPSILON)
        # Get the mean of the dices over all classes
        dice_loss = 1. - torch.mean(dice, dim=1)  # b,

        loss = dice_loss + focal_loss

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        # Default is mean reduction
        else:
            return loss.mean()


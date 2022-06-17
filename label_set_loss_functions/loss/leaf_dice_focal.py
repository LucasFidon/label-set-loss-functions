"""
@brief  Pytorch implementation of the Leaf Dice loss + marginalized Focal loss.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch.nn as nn
from label_set_loss_functions.loss import MarginalizedFocalLoss, LeafDiceLoss
from label_set_loss_functions.utils import check_label_set_map


class LeafDiceFocalLoss(nn.Module):
    def __init__(self, labels_superset_map, gamma=2., weight=None, squared=True, reduction='mean'):
        """
        :param gamma: (float) value of the exponent gamma in the definition
            of the Focal loss.
        :param weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
        :param squared: bool. If True, squared the terms in the denominator of the Dice loss.
        :param labels_superset_map: dict to handle superclasses
        :param reduction: str.
        """
        super(LeafDiceFocalLoss, self).__init__()
        self.focal_loss = MarginalizedFocalLoss(
            gamma=gamma,
            weight=weight,
            reduction=reduction,
            labels_superset_map=labels_superset_map,
        )
        self.leaf_dice = LeafDiceLoss(
            reduction=reduction,
            labels_superset_map=labels_superset_map,
            squared=squared,
        )
        self.gamma = gamma
        self.squared = squared
        check_label_set_map(labels_superset_map)
        self.labels_superset_map = labels_superset_map


    def forward(self, input, target):
        focal_loss = self.focal_loss(input, target)  # reduction already done
        leaf_dice = self.leaf_dice(input, target)
        loss = leaf_dice + focal_loss
        return loss

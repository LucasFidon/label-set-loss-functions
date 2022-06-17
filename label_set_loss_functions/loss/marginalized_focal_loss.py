"""
@brief  Pytorch implementation of the marginalized Focal loss.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch
from label_set_loss_functions.loss.focal_loss import FocalLoss
from label_set_loss_functions.convertor.marginalization import log_softmax_marginalize
from label_set_loss_functions.utils import check_label_set_map


class MarginalizedFocalLoss(FocalLoss):
    def __init__(self, labels_superset_map, gamma=2., weight=None, reduction='mean'):
        """
        :param gamma: (float) value of the exponent gamma in the definition
            of the Focal loss.
        :param weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
        :param labels_superset_map: dict to handle superclasses
        :param reduction: str.
        """
        super(MarginalizedFocalLoss, self).__init__(
            gamma=gamma,
            weight=weight,
            reduction=reduction,
        )
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

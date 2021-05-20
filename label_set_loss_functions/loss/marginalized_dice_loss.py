"""
@brief  Pytorch implementation of the mean marginalized Dice loss

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch
from label_set_loss_functions.loss.mean_dice_loss import MeanDiceLoss
from label_set_loss_functions.convertor.marginalization import softmax_marginalize


class MarginalizedDiceLoss(MeanDiceLoss):
    def __init__(self, labels_superset_map, reduction='mean', squared=True):
        """
        :param labels_superset_map: dict to handle superclasses
        :param reduction: str.
        :param squared: bool.
        """
        super(MarginalizedDiceLoss, self).__init__(
            reduction=reduction,
            squared=squared,
        )
        self.labels_superset_map = labels_superset_map

    def _prepare_data(self, input_batch, target):
        # Flatten the predicted class score map and the segmentation ground-truth
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))
        # Compute the marginalization
        pred_proba, target_proba = softmax_marginalize(
            flat_input=flat_input,
            flat_target=flat_target,
            labels_superset_map=self.labels_superset_map,
        )
        return pred_proba, target_proba

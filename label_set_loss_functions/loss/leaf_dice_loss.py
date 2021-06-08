"""
@brief  Pytorch implementation of the mean Leaf-Dice loss

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch
import torch.nn.functional as F
from label_set_loss_functions.loss.mean_dice_loss import MeanDiceLoss
from label_set_loss_functions.utils import check_label_set_map


class LeafDiceLoss(MeanDiceLoss):
    def __init__(self, labels_superset_map, reduction='mean', squared=True):
        """
        :param labels_superset_map: dict to handle superclasses
        :param reduction: str.
        :param squared: bool.
        """
        super(LeafDiceLoss, self).__init__(
            reduction=reduction,
            squared=squared,
        )
        check_label_set_map(labels_superset_map)
        self.labels_superset_map = labels_superset_map

    def _prepare_data(self, input_batch, target):
        num_out_classes = input_batch.size(1)
        # Prepare the batch prediction
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))
        pred_proba = F.softmax(flat_input, dim=1)  # b,c,s
        flat_target = flat_target.long()  # make sure that the target is a long before using one_hot
        target_proba = F.one_hot(
            flat_target, num_classes=-1).permute(0, 2, 1).float()
        # Remove the supersets channels from the target proba.
        # As a consequence they will be masked in the loss
        # no need to do anything else.
        target_proba = target_proba[:, :num_out_classes, :]
        return pred_proba, target_proba

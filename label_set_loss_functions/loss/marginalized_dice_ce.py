"""
@brief  Pytorch implementation of the mean marginalized Dice loss + marginalized Cross Entropy loss.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

from label_set_loss_functions.loss.marginalized_dice_focal import MarginalizedDiceFocalLoss


class MarginalizedDiceCE(MarginalizedDiceFocalLoss):
    def __init__(self, weight=None, squared=True, reduction='mean', labels_superset_map=None):
        """
        :param weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
        :param squared: bool. If True, squared the terms in the denominator of the Dice loss.
        :param labels_superset_map: dict to handle superclasses
        :param reduction: str.
        """
        super(MarginalizedDiceCE, self).__init__(
            gamma=0,
            weight=weight,
            squared=squared,
            reduction=reduction,
            labels_superset_map=labels_superset_map,
        )

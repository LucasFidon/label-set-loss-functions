"""
@brief  Pytorch implementation of the marginalized Cross Entropy loss.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

from label_set_loss_functions.loss import MarginalizedFocalLoss


class MarginalizedCrossEntropyLoss(MarginalizedFocalLoss):
    def __init__(self, weight=None, reduction='mean'):
        """
        :param weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
        :param reduction: str.
        """
        super(MarginalizedCrossEntropyLoss, self).__init__(
            gamma=0,
            weight=weight,
            reduction=reduction,
        )

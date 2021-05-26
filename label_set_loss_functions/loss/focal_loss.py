"""
@brief  Pytoch implementation of the Focal Loss
        [1] "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017

        This is a weighted variant of the Cross Entropy (CE).
        In contrast to the CE, the Focal Loss is negligeable
        for well classified units.
        As a result, the Focal Loss focuses on not yet well classified units.

        When gamma=0, the focal loss is the cross entropy loss.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class FocalLoss(_WeightedLoss):
    """
    PyTorch implementation of the Focal Loss.
    [1] "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017
    """
    def __init__(self, gamma=2., weight=None, reduction='mean'):
        """
        :param gamma: (float) value of the exponent gamma in the definition
            of the Focal loss.
        :param weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
            This corresponds to the weights \alpha in [1].
        :param reduction: (string) Reduction operation to apply on the loss batch.
            It can be 'mean', 'sum' or 'none' as in the standard PyTorch API
            for loss functions.
        """
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma

    def _prepare_data(self, input_batch, target):
        num_out_classes = input_batch.size(1)
        # Prepare the batch prediction
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))  # b,s
        log_pred_proba = F.log_softmax(flat_input, dim=1)  # b,c,s
        target_proba = F.one_hot(flat_target, num_classes=num_out_classes)
        target_proba = target_proba.permute(0, 2, 1).float()  # b,c,s
        return log_pred_proba, target_proba

    def forward(self, input, target):
        logpt, target_proba = self._prepare_data(input, target) # b,c,s

        # Get the proba
        pt = torch.exp(logpt)  # b,c,s

        if self.weight is not None:
            if self.weight.type() != logpt.data.type():
                self.weight = self.weight.type_as(logpt.data)
            # Expand the weight to a map of shape b,c,s
            w_class = self.weight[None, :, None] # c => 1,c,1
            w_class = w_class.expand((logpt.size(0), -1, logpt.size(2)))  # 1,c,1 => b,c,s
            logpt = logpt * w_class

        # Compute the loss mini-batch
        weight = torch.pow(-pt + 1., self.gamma)
        # sum over classes and mean over voxels
        per_voxel_loss = torch.sum(-weight * target_proba * logpt, dim=1)  # b,s
        loss = torch.mean(per_voxel_loss, dim=1)  # b,

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        # Default is mean reduction
        else:
            return loss.mean()

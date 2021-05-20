"""
@brief  Marginalization functions.
        Plug this between the output of your neural network and your loss function.
        This should replace either the softmax or log_softmax function.
        Please see src/loss/marginalized_dice_loss.py for an example with the mean Dice loss.
@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   May 2021
"""

import torch
import torch.nn.functional as F

EPS = 1e-10


def softmax_marginalize(flat_input, flat_target, labels_superset_map):
    assert flat_input.dim() == 1 + flat_target.dim(), "Only compatible with label map ground-truth segmentation"

    # Get the nb of classes that are not supersets.
    # This should correspond to the number of output channels.
    num_out_classes = flat_input.size(1)  # nb of classes that are not supersets

    # Initialize the batch probability map prediction
    pred_proba = F.softmax(flat_input, dim=1).permute(0, 2, 1)  # b,s,c

    # Initialize the target probability map
    # Warning: here we assume that the superset label numbers are higher
    # than the (singleton) label number
    target_proba = F.one_hot(flat_target, num_classes=-1).float()  # b,s,c+
    # Remove the supersets
    target_proba = target_proba[:, :, :num_out_classes]  # b,s,c

    # Marginalize the predicted and target probability maps
    for super_class in list(labels_superset_map.keys()):
        with torch.no_grad():
            # Compute the mask to use for the marginalization wrt super class super_class
            super_class_size = len(labels_superset_map[super_class])
            super_class_mask = (flat_target == super_class)  # b,s
            w = torch.zeros(num_out_classes, device=flat_input.device)  # c,
            w_bool = torch.zeros(num_out_classes, device=flat_input.device)  # c,
            mask = torch.zeros_like(pred_proba, device=flat_input.device, requires_grad=False)  # b,s,c
            for c in labels_superset_map[super_class]:
                w[c] = 1. / super_class_size
                w_bool[c] = 1
            mask[super_class_mask, :] = w_bool[None, :]

            # Marginalize the target probability for the super class super_class
            target_proba[super_class_mask, :] = w[None, :]

        # Marginalize the predicted proba for the super class super_class
        marginal_map_full = torch.sum(w[None, None, :] * pred_proba, dim=2)  # b,s
        # This uses a lot of memory...
        pred_proba = (1 - mask) * pred_proba + mask * marginal_map_full[:, :, None]

    # Transpose to match PyTorch convention
    pred_proba = pred_proba.permute(0, 2, 1)  # b,c,s
    target_proba = target_proba.permute(0, 2, 1)  # b,c,s

    return pred_proba, target_proba


def log_softmax_marginalize(flat_input, flat_target, labels_superset_map):
    """
    Compute
    log(1/Omega x \sum_{gt classes set} exp(out_c)) - log(1/Omega x \sum_{all classes} exp(out_c))

    :param flat_input:
    :param flat_target:
    :param labels_superset_map:
    :return:
    """
    assert flat_input.dim() == 1 + flat_target.dim(), "Only compatible with label map ground-truth segmentation"

    # Get the nb of classes that are not supersets.
    # This should correspond to the number of output channels.
    num_out_classes = flat_input.size(1)  # nb of classes that are not supersets

    # Normalize flat_input for stability
    max_flat_input, _ = torch.max(flat_input, dim=1, keepdim=True)
    flat_input = flat_input - max_flat_input  # substract the max per-voxel activation

    # Initialize the batch log probability map prediction
    pred_logproba = F.log_softmax(flat_input, dim=1).permute(0, 2, 1)  # b,s,c
    # We need the exponential of the output map
    pred_explogit = torch.exp(flat_input).permute(0, 2, 1)  # b,s,c
    pred_explogit = pred_explogit + EPS  # add epsilon to avoid to get -inf out of torch.log later

    # Initialize the target probability map
    # Warning: here we assume that the superset label numbers are higher
    # than the (singleton) label number
    target_proba = F.one_hot(flat_target, num_classes=-1).float()  # b,s,c+
    # Remove the supersets
    target_proba = target_proba[:, :, :num_out_classes]  # b,s,c

    # Marginalize the predicted and target probability / log-probability maps
    for super_class in list(labels_superset_map.keys()):
        super_class_size = len(labels_superset_map[super_class])
        super_class_mask = (flat_target == super_class)  # b,s
        w = torch.zeros(num_out_classes, device=flat_input.device)  # c,
        w_bool = torch.zeros(num_out_classes, device=flat_input.device)  # c,
        mask = torch.zeros_like(pred_logproba, device=flat_input.device)  # b,s,c
        for c in labels_superset_map[super_class]:
            w[c] = 1. / super_class_size
            w_bool[c] = 1
        mask[super_class_mask, :] = w_bool[None, :]  # 1 if part of the ground-truth superset else 0
        target_proba[super_class_mask, :] = w[None, :]
        # Most important lines of code; Compute for all voxels:
        # log(1/Omega x \sum_{super_class} exp(out_c)) - log(\sum_{all classes} exp(out_c))
        superclass_logproba = torch.log(torch.sum(w[None, None, :] * pred_explogit, dim=2))  # b,s
        superclass_logproba -= torch.log(torch.sum(pred_explogit, dim=2))
        # This uses a lot of memory...
        # Copy the logproba to all the class proba of the ambiguous voxels
        pred_logproba = (1 - mask) * pred_logproba + mask * superclass_logproba[:, :, None]

    # Transpose to match PyTorch convention
    pred_logproba = pred_logproba.permute(0, 2, 1)  # b,c,s
    target_proba = target_proba.permute(0, 2, 1)  # b,c,s

    return pred_logproba, target_proba

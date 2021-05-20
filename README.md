# Label-set Loss Functions for Partial Supervision
Label-set loss functions are loss functions 
that can handle partially segmented images,
i.e. images for which some, but not all, regions of interest
are segmented.


## Installation
```bash
pip install git+https://github.com/LucasFidon/label-set-loss-functions.git
```

## Example
```python
import torch
from label_set_loss_functions.loss import LeafDiceLoss, MarginalizedDiceLoss

# Example with 4 classes (labels 0 to 3) and one super class (label 4) that contains the labels 2 and 3
labels_superset_map = {4: [2, 3]}

# partial segmentation in 1D with 3 voxels (ground truth).
# Dimensions: num batch x num leaf labels x num voxels = 1 x 3
partial_seg = torch.tensor([[0, 1, 4]], dtype=torch.int64).cuda()

# We define two predicted score maps that are equivalent wrt the partial annotation
# Dimensions: num batch x num leaf labels x num voxels = 1 x 4 x 3
score_pred1 = torch.tensor([[[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]], dtype=torch.float32).cuda()
score_pred2 = torch.tensor([[[100, 0, 0], [0, 100, 0], [0, 0, 0], [0, 0, 100]]], dtype=torch.float32).cuda()

# Marginalized Dice loss
marg_dice = MarginalizedDiceLoss(labels_superset_map)
marg_dice(score_pred1, partial_seg)  # approximately 0
marg_dice(score_pred2, partial_seg)  # approximately 0

# Leaf-Dice loss
leaf_dice = LeafDiceLoss(labels_superset_map)
leaf_dice(score_pred1, partial_seg)  # 0.5
leaf_dice(score_pred2, partial_seg)  # 0.5
```

## How to cite
TO DO
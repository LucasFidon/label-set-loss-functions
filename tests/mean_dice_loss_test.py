import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from label_set_loss_functions.loss.mean_dice_loss import MeanDiceLoss


class TestMeanDiceLoss(unittest.TestCase):
    def test_bin_seg_2d(self):
        # define 2d examples
        target = torch.tensor(
            [[0,0,0,0],
             [0,1,1,0],
             [0,1,1,0],
             [0,0,0,0]]
        )
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(
            target, num_classes=2).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(
            1 - target, num_classes=2).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = MeanDiceLoss()

        # mean dice loss for pred_very_good should be close to 0
        dice_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(dice_loss_good, 0., places=3)

        # same test, but with target with a class dimension
        target_4dim = target.unsqueeze(1)  # shape (1, 1, H, W)
        dice_loss_good = float(loss.forward(pred_very_good, target_4dim).cpu())
        self.assertAlmostEqual(dice_loss_good, 0., places=3)

        # mean dice loss for pred_very_poor should be close to 1
        dice_loss_poor = float(loss.forward(pred_very_poor, target).cpu())
        self.assertAlmostEqual(dice_loss_poor, 1., places=3)

    def test_empty_class_2d(self):
        num_classes = 2
        # define 2d examples
        target = torch.tensor(
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(
            target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(
            1 - target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = MeanDiceLoss()

        # mean dice loss for pred_very_good should be close to 0.5 (and not 0)
        # because for empty prediction and empty target segmentation
        # the Dice loss value is 0 and not 1
        dice_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(dice_loss_good, 0.5, places=3)

        # mean dice loss for pred_very_poor should be close to 1
        dice_loss_poor = float(loss.forward(pred_very_poor, target).cpu())
        self.assertAlmostEqual(dice_loss_poor, 1., places=3)

    def test_multi_class_seg_2d(self):
        num_classes = 6  # labels 0 to 5
        # define 2d examples
        target = torch.tensor(
            [[0,0,0,0],
             [0,1,2,0],
             [0,3,4,0],
             [0,0,0,0]]
        )
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(
            target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(
            target + 1, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = MeanDiceLoss()

        # mean dice loss for pred_very_good should be close to 1/6 and not 0
        # because we do not have an epsilon in the numerator of the Dice loss.
        # As a result, for empty prediction and empty taregt the Dice loss value is 1 and not 0.
        dice_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(dice_loss_good, 1./6., places=3)

        # mean dice loss for pred_very_poor should be close to 1
        dice_loss_poor = float(loss.forward(pred_very_poor, target).cpu())
        self.assertAlmostEqual(dice_loss_poor, 1., places=3)

    def test_bin_seg_3d(self):
        # define 3d examples
        target = torch.tensor(
            [
            # raw 0
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            # raw 1
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]],
            # raw 2
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]]
             ]
        )
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W, D)
        pred_very_good = 1000 * F.one_hot(
            target, num_classes=2).permute(0, 4, 1, 2, 3).float()
        pred_very_poor = 1000 * F.one_hot(
            1 - target, num_classes=2).permute(0, 4, 1, 2, 3).float()

        # initialize the mean dice loss
        loss = MeanDiceLoss()

        # mean dice loss for pred_very_good should be close to 0
        dice_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(dice_loss_good, 0., places=3)

        # mean dice loss for pred_very_poor should be close to 1
        dice_loss_poor = float(loss.forward(pred_very_poor, target).cpu())
        self.assertAlmostEqual(dice_loss_poor, 1., places=3)

    def test_convergence(self):
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        learning_rate = 0.01
        max_iter = 20

        # define a simple 3d example
        target_seg = torch.tensor(
            [
            # raw 0
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            # raw 1
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]],
            # raw 2
             [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]]
             ]
        )
        target_seg = torch.unsqueeze(target_seg, dim=0)
        image = 12 * target_seg + 27
        image = image.float()
        num_classes = 2
        num_voxels = 3 * 4 * 4
        # define a one layer model
        class OnelayerNet(nn.Module):
            def __init__(self):
                super(OnelayerNet, self).__init__()
                self.layer = nn.Linear(num_voxels, num_voxels * num_classes)
            def forward(self, x):
                x = x.view(-1, num_voxels)
                x = self.layer(x)
                x = x.view(-1, num_classes, 3, 4, 4)
                return x

        # initialise the network
        net = OnelayerNet()

        # initialize the loss
        loss = MeanDiceLoss()

        # initialize an SGD
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

        loss_history = []
        # train the network
        for _ in range(max_iter):
            # set the gradient to zero
            optimizer.zero_grad()

            # forward pass
            output = net(image)
            loss_val = loss(output, target_seg)

            # backward pass
            loss_val.backward()
            optimizer.step()

            # stats
            loss_history.append(loss_val.item())

        # count the number of SGD steps in which the loss decreases
        num_decreasing_steps = 0
        for i in range(len(loss_history) - 1):
            if loss_history[i] > loss_history[i+1]:
                num_decreasing_steps += 1
        decreasing_steps_ratio = float(num_decreasing_steps) / (len(loss_history) - 1)

        # verify that the loss is decreasing for sufficiently many SGD steps
        self.assertTrue(decreasing_steps_ratio > 0.9)


if __name__ == '__main__':
    unittest.main()

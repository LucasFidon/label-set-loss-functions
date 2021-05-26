import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from label_set_loss_functions.loss import MarginalizedDiceLoss


class TestMarginalizedDiceLoss(unittest.TestCase):
    def test_partial_label_2d(self):
        num_classes = 3  # labels 0 to 2
        labels_superset_map = {
            3: [1, 2],
            4: [0, 1, 2],
        }
        # Define a 2d example
        target = torch.tensor(
            [[0,0,0,0],
             [0,1,2,0],
             [0,3,4,0],
             [0,0,0,0]]
        )
        # Add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        target_2 = target.clone()
        target_2[target_2 > 2] = 2
        pred_very_good = 1000 * F.one_hot(
            target_2, num_classes=num_classes).permute(0, 3, 1, 2).float()
        target_1 = target.clone()
        target_1[target_1 > 1] = 1
        pred_1_error = 1000 * F.one_hot(
            target_1, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred_3_errors = 1000 * F.one_hot(
            torch.zeros_like(target), num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Initialize the mean dice loss
        loss = MarginalizedDiceLoss(labels_superset_map=labels_superset_map)

        # Mean dice loss for pred_very_good should be close to 0
        true_res = 0.
        dice_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(dice_loss_good, true_res, places=3)

        true_res = 0.2831006
        dice_loss_1_error = float(loss.forward(pred_1_error, target).cpu())
        self.assertAlmostEqual(dice_loss_1_error, true_res, places=3)

        true_res = 0.6027724297266077
        dice_loss_3_error = float(loss.forward(pred_3_errors, target).cpu())
        self.assertAlmostEqual(dice_loss_3_error, true_res, places=3)

    def test_convergence_partial_label(self):
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        learning_rate = 0.001
        max_iter = 50
        num_classes = 2  # labels 0 and 1
        labels_superset_map = {
            2: [0, 1],
        }

        # Define a simple 3d example
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
              [0, 1, 2, 2],
              [0, 0, 2, 2]],
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
        # Define a one layer model
        class OnelayerNet(nn.Module):
            def __init__(self):
                super(OnelayerNet, self).__init__()
                self.layer = nn.Linear(num_voxels, num_voxels * num_classes)
            def forward(self, x):
                x = x.view(-1, num_voxels)
                x = self.layer(x)
                x = x.view(-1, num_classes, 3, 4, 4)
                return x

        # Initialise the network
        net = OnelayerNet()

        # Initialize the loss
        loss = MarginalizedDiceLoss(labels_superset_map=labels_superset_map)

        # Initialize an SGD
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

        loss_history = []
        # Train the network
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

        # Count the number of SGD steps in which the loss decreases
        num_decreasing_steps = 0
        for i in range(len(loss_history) - 1):
            if loss_history[i] > loss_history[i+1]:
                num_decreasing_steps += 1
        decreasing_steps_ratio = float(num_decreasing_steps) / (len(loss_history) - 1)

        # Verify that the loss is decreasing for sufficiently many SGD steps
        self.assertTrue(decreasing_steps_ratio > 0.5)


if __name__ == '__main__':
    unittest.main()

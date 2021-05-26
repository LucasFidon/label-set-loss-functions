import unittest
import torch
import numpy as np
from label_set_loss_functions.convertor import softmax_marginalize, log_softmax_marginalize


class TestMarginalizationFunctions(unittest.TestCase):
    def test_softmax_marginalize(self):
        num_classes = 3  # labels 0 to 2
        labels_superset_map = {
            3: [1, 2],
            4: [0, 1, 2],
        }
        # Define 1d example
        target = torch.tensor(  # shape: (batch size, num voxels)
            [[0, 0, 0, 1, 2, 0, 0, 3, 4, 0]]
        )
        # First score map for a very good segmentation
        score_map1 = 100. * torch.tensor(  # shape: (batch size, num classes, num voxels)
            [[
                [1., 1., 1., 0., 0., 1., 1., 0., 1., 1.],
                [0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
            ]]
        )
        # Second score map for a very good segmentation (equivalent to the first one)
        score_map2 = 100. * torch.tensor(  # shape: (batch size, num classes, num voxels)
            [[
                [1., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1., 0., 0., 1., 0.01, 0.]
            ]]
        )
        # Expected output probabilities (same for the two previous examples)
        expected_out = np.array(  # shape: (batch size, num classes, num voxels)
            [
                [1., 1., 1., 0., 0., 1., 1., 0., 1. / len(labels_superset_map[4]), 1.],
                [0., 0., 0., 1., 0., 0., 0., 1. / len(labels_superset_map[3]), 1. / len(labels_superset_map[4]), 0.],
                [0., 0., 0., 0., 1., 0., 0., 1. / len(labels_superset_map[3]), 1. / len(labels_superset_map[4]), 0.]
            ]
        )
        # Compute the softmax + marginalization
        marg_proba1, marg_target = softmax_marginalize(
            flat_input=score_map1, flat_target=target, labels_superset_map=labels_superset_map)
        self.assertAlmostEqual(np.linalg.norm(marg_target.cpu().numpy() - expected_out), 0.)
        self.assertAlmostEqual(np.linalg.norm(marg_proba1.cpu().numpy() - expected_out), 0.)
        marg_proba2, _ = softmax_marginalize(
            flat_input=score_map2, flat_target=target, labels_superset_map=labels_superset_map)
        self.assertAlmostEqual(np.linalg.norm(marg_proba2.cpu().numpy() - expected_out), 0.)


    def test_log_softmax_marginalize(self):
        num_classes = 3  # labels 0 to 2
        labels_superset_map = {
            3: [1, 2],
            4: [0, 1, 2],
        }
        # Define 1d example
        target = torch.tensor(  # shape: (batch size, num voxels)
            [[0, 0, 0, 1, 2, 0, 0, 3, 4, 0]]
        )
        # First score map for a very good segmentation
        score_map1 = 100. * torch.tensor(  # shape: (batch size, num classes, num voxels)
            [[
                [1., 1., 1., 0., 0., 1., 1., 0., 1., 1.],
                [0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
            ]]
        )
        # Second score map for a very good segmentation (equivalent to the first one)
        score_map2 = 100. * torch.tensor(  # shape: (batch size, num classes, num voxels)
            [[
                [1., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1., 0., 0., 1., 0.01, 0.]
            ]]
        )
        # Expected output probabilities (same for the two previous examples)
        expected_out = np.array(  # shape: (batch size, num classes, num voxels)
            [
                [1., 1., 1., 0., 0., 1., 1., 0., 1. / len(labels_superset_map[4]), 1.],
                [0., 0., 0., 1., 0., 0., 0., 1. / len(labels_superset_map[3]), 1. / len(labels_superset_map[4]), 0.],
                [0., 0., 0., 0., 1., 0., 0., 1. / len(labels_superset_map[3]), 1. / len(labels_superset_map[4]), 0.]
            ]
        )
        # Compute the softmax + marginalization
        log_marg_proba1, marg_target = log_softmax_marginalize(
                flat_input=score_map1, flat_target=target, labels_superset_map=labels_superset_map)
        marg_proba1 = torch.exp(log_marg_proba1)
        self.assertAlmostEqual(np.linalg.norm(marg_target.cpu().numpy() - expected_out), 0.)
        self.assertAlmostEqual(np.linalg.norm(marg_proba1.cpu().numpy() - expected_out), 0.)
        log_marg_proba2, _ = log_softmax_marginalize(
                flat_input=score_map2, flat_target=target, labels_superset_map=labels_superset_map)
        marg_proba2 = torch.exp(log_marg_proba2)
        self.assertAlmostEqual(np.linalg.norm(marg_proba2.cpu().numpy() - expected_out), 0.)


if __name__ == '__main__':
    unittest.main()

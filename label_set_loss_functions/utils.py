import numpy as np


def check_label_set_map(labels_superset_map):
    # Check that the super class labels are higher than for the class labels
    for labelset in list(labels_superset_map):
        set = labels_superset_map[labelset]
        err_msg = 'Superset label %d is lower than at least one class label in %s' % (labelset, str(set))
        np.testing.assert_array_less(set, labelset, err_msg=err_msg)

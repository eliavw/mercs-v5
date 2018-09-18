# Standard imports
import os
import numpy as np
import sys
from os.path import dirname


# Custom imports
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

from mercs.algo.inference import init_predictions, update_X


def test_init_predictions():
    nb_rows, nb_cols = 10,10

    obs = init_predictions(nb_rows, nb_cols, type=np.float64)

    assert obs.shape == (nb_rows, nb_cols)
    assert isinstance(obs[0, 0], np.float64)


def test_update_X():
    X = np.zeros((100, 4), dtype=np.float64)
    Y = np.random.rand(100, 2)
    act_att_idx = [1, 2]

    obs = update_X(X, Y, act_att_idx)

    test_col_0 = obs[:, act_att_idx[0]] == Y.T[0]
    test_col_1 = obs[:, act_att_idx[1]] == Y.T[1]

    assert obs.shape == X.shape
    assert test_col_0.all()
    assert test_col_1.all()

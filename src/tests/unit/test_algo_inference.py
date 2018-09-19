# Standard imports
import os
import numpy as np
import sys
from os.path import dirname


# Custom imports
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

from mercs.algo.inference import (init_predictions,
                                  update_X,
                                  predict_values_from_numer,
                                  predict_values_from_proba)


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


def test_predict_values_from_numer():
    # Init
    nb_atts = 4
    numer_res = [None] * nb_atts

    for i in range(nb_atts):
        numer_res[i] = np.random.rand(100, 1)

    counts = np.random.randint(1, 10, size=(nb_atts))

    # Actual Test
    obs = predict_values_from_numer(numer_res, counts)

    assert obs.shape[0] == numer_res[0].shape[0]
    assert obs.shape[1] == nb_atts
    assert isinstance(obs, np.ndarray)


def test_predict_values_from_proba():
    # Init
    nb_atts = 5
    nb_classes = 3
    nb_samples = 100
    proba_res = [None] * nb_atts
    lab_res = [None] * nb_atts

    for i in range(nb_atts):
        proba_res[i] = np.random.rand(nb_samples, nb_classes)

    for i in range(nb_atts):
        lab_res[i] = np.random.choice(10, size=nb_classes, replace=False)

    obs = predict_values_from_proba(proba_res, lab_res)

    assert obs.shape == (nb_samples, nb_atts)

    for a in range(nb_atts):
        assert np.array_equal(np.unique(obs[:, a]), np.unique(lab_res[a]))



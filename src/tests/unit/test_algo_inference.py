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
                                  predict_values_from_proba,
                                  merge_numer,
                                  merge_proba)


def test_init_predictions():
    nb_rows, nb_cols = 10, 10

    obs = init_predictions(nb_rows, nb_cols, dtype=np.float64)

    assert obs.shape == (nb_rows, nb_cols)
    assert isinstance(obs[0, 0], np.float64)


def test_update_X():
    X = np.zeros((100, 4), dtype=np.float64)
    Y = np.random.rand(100, 2)
    act_att_idx = np.array([1, 2])

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

    return


def test_merge_numer():
    # Prelims
    nb_samples = 100
    nb_targ_res = 6
    numer_res = [None] * nb_targ_res

    for i in range(nb_targ_res):
        numer_res[i] = np.random.rand(nb_samples, 1)

    # Multi-target sklearn output
    nb_targ_mod_a = 3
    numer_mod_a = np.random.rand(nb_samples, nb_targ_mod_a)

    # Single-target sklearn output
    nb_targ_mod_b = 1
    numer_mod_b = np.random.rand(nb_samples, nb_targ_mod_b)
    numer_mod_b = np.squeeze(numer_mod_b)

    # Our own output format
    nb_targ_mod_c = 2
    numer_mod_c = [None] * nb_targ_mod_c
    for i in range(nb_targ_mod_c):
        numer_mod_c[i] = np.random.rand(100, 1)

    t_idx_res = 2
    t_idx_mod = 0

    for numer_mod in [numer_mod_a, numer_mod_b, numer_mod_c]:
        obs = merge_numer(numer_res, numer_mod, t_idx_res, t_idx_mod)

        assert isinstance(obs, list)
        assert len(obs) == nb_targ_res

    return


def test_merge_proba():
    # Prelims
    nb_samples = 100
    nb_targ_res = 6
    nb_classes = 3
    proba_res = [None] * nb_targ_res
    lab_res = [None] * nb_targ_res

    for i in range(nb_targ_res):
        proba_res[i] = np.random.rand(nb_samples, nb_classes)

    for i in range(nb_targ_res):
        lab_res[i] = np.random.choice(10, size=nb_classes, replace=False)

    # list (our own output or multi-target sklearn)
    mutual_targets_a = [0, 2, 4]
    nb_targ_mod_a = 3
    assert len(mutual_targets_a) == nb_targ_mod_a

    nb_classes_mod_a = 2
    proba_mod_a = [None] * nb_targ_mod_a
    lab_mod_a = [None] * nb_targ_mod_a

    for i in range(nb_targ_mod_a):
        proba_mod_a[i] = np.random.rand(nb_samples, nb_classes_mod_a)

    lab_mod_a = [lab_res[idx][0:nb_classes_mod_a] for idx in mutual_targets_a]

    # np.ndarray (single-target sklearn)
    mutual_targets_b = [4]
    nb_targ_mod_b = 1
    assert len(mutual_targets_b) == nb_targ_mod_b

    nb_classes_mod_b = 2

    proba_mod_b = np.random.rand(nb_samples, nb_classes_mod_b)

    lab_mod_b = [lab_res[idx][-nb_classes_mod_b:] for idx in mutual_targets_b]

    # Actual Test
    t_idx_mod = 0
    t_idx_res = mutual_targets_a[t_idx_mod]
    print(t_idx_res)
    obs = merge_proba(proba_res, proba_mod_a, lab_res, lab_mod_a, t_idx_res, t_idx_mod)

    for x in obs:
        assert isinstance(x, np.ndarray)
        assert x.shape == (nb_samples, nb_classes)

    t_idx_mod = 0
    t_idx_res = mutual_targets_b[t_idx_mod]
    print(t_idx_res)
    obs = merge_proba(proba_res, proba_mod_b, lab_res, lab_mod_b, t_idx_res, t_idx_mod)

    for x in obs:
        assert isinstance(x, np.ndarray)
        assert x.shape == (nb_samples, nb_classes)

    return





import numpy as np
import os
import pandas as pd
import sys

from os.path import dirname

# Custom import (Add src to the path)
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

import datasets as datasets
from mercs.core import MERCS
from mercs.utils.encoding import encode_attribute
from tests.utils.eval import verify_nominal_prediction, verify_numeric_prediction


def setup():
    train, test = datasets.load_fertility()
    model = MERCS()

    # Ensure attributes are correctly recognized as nominal/numeric
    train['season'] = pd.factorize(train['season'])[0]
    test['season'] = pd.factorize(test['season'])[0]

    ind_parameters = {'ind_type':           'RF',
                      'ind_n_estimators':   10,
                      'ind_max_depth':      4}

    sel_parameters = {'sel_type':           'Base',
                      'sel_its':            8,
                      'sel_param':          2}

    model.fit(train, **ind_parameters, **sel_parameters)

    code = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

    target_boolean = np.array(code) == encode_attribute(2, [1], [2])
    y_true = test[test.columns.values[target_boolean]].values
    return train, test, code, model, y_true, target_boolean


def evaluate(y_true, y_pred, clf_labels_targets):

    for t_idx, clf_labels_targ in enumerate(clf_labels_targets):
        single_y_true = y_true[:, t_idx]
        single_y_pred = y_pred[:, t_idx].astype(int)

        if isinstance(clf_labels_targ, np.ndarray):
            # Nominal target
            verify_nominal_prediction(single_y_true, single_y_pred)
        elif isinstance(clf_labels_targ, list):
            # Numeric target
            assert clf_labels_targ == ['numeric']

            verify_numeric_prediction(single_y_pred, single_y_pred)
        else:
            msg = """
            clf_labels of MERCS are either:\n
            np.ndarray, shape (classlabels,)\n
            \t for nominal attributes\n
            list, shape (1,)\n
            \t ['numeric] for numeric attributes \n"""
            raise TypeError(msg)

    return


def test_MI():
    train, test, code, model, y_true, target_boolean = setup()

    pred_parameters = {'pred_type':     'MI',
                       'pred_param':    0.95,
                       'pred_its':      0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    clf_labels_targets = [model.s['metadata']['clf_labels'][t]
                          for t, check in enumerate(target_boolean)
                          if check]

    evaluate(y_true, y_pred, clf_labels_targets)

    return


def test_MA():
    train, test, code, model, y_true, target_boolean = setup()

    pred_parameters = {'pred_type':     'MA',
                       'pred_param':    0.95,
                       'pred_its':      0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    clf_labels_targets = [model.s['metadata']['clf_labels'][t]
                          for t, check in enumerate(target_boolean)
                          if check]

    evaluate(y_true, y_pred, clf_labels_targets)

    return


def test_MAFI():
    train, test, code, model, y_true, target_boolean = setup()

    pred_parameters = {'pred_type':     'MAFI',
                       'pred_param':    0.95,
                       'pred_its':      0.1}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    clf_labels_targets = [model.s['metadata']['clf_labels'][t]
                          for t, check in enumerate(target_boolean)
                          if check]

    evaluate(y_true, y_pred, clf_labels_targets)

    return


def test_IT():
    train, test, code, model, y_true, target_boolean = setup()

    pred_parameters = {'pred_type':     'IT',
                       'pred_param':    0.1,
                       'pred_its':      0.8}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    clf_labels_targets = [model.s['metadata']['clf_labels'][t]
                          for t, check in enumerate(target_boolean)
                          if check]

    evaluate(y_true, y_pred, clf_labels_targets)

    return

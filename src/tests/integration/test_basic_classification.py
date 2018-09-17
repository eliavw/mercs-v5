"""
Test of basic classification task.
"""

# Standard imports
import numpy as np
import os
from os.path import dirname
import sys
from sklearn.metrics import f1_score

# Custom import (Add src to the path)
root_directory = dirname(dirname(dirname(dirname(__file__))))

msg = "os.listdir: {}".format(os.listdir(root_directory))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))
from mercs.core import MERCS
from mercs.utils import *
import datasets as datasets


def test_basic_classification():
    train, test = datasets.load_nursery()

    model = MERCS()

    ind_parameters = {'ind_type': 'RF',
                      'ind_n_estimators': 30}

    sel_parameters = {'sel_type': 'Base',
                      'sel_its': 4,
                      'sel_param': 2}

    model.fit(train, **ind_parameters, **sel_parameters)

    code = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    y_true = test[test.columns.values[np.array(code) == 1]].values

    pred_parameters = {'pred_type': 'MI',
                       'pred_param': 1.0,
                       'pred_its': 8}

    y_pred = model.predict(test,
                           **pred_parameters,
                           qry_code=code)

    result = f1_score(y_true, y_pred, average='macro')

    assert isinstance(result, (int, float))
    test_one = 0<=result
    test_two = result <= 1

    return test_one and test_two
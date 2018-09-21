import os
import numpy as np
import sys
import warnings

from os.path import dirname
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import Imputer

# Custom imports
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

import datasets as ds
from mercs.algo.inference import perform_imputation
from mercs.utils.encoding import encode_attribute

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def test_perform_imputation():
    # Prelims
    train, test = ds.load_nursery()
    query_code = [0, -1, -1, -1, -1, -1, 0, 0, 1]

    imputator = Imputer(missing_values='NaN',
                        strategy='most_frequent',
                        axis=0)
    imputator.fit(train)

    # Actual test
    obs = perform_imputation(test, query_code, imputator)

    assert test.shape == obs.shape
    assert isinstance(obs, np.ndarray)

    boolean_missing = encode_attribute(0, [1], [2])

    for row in obs[:, boolean_missing].T:
        assert len(np.unique(row)) == 1
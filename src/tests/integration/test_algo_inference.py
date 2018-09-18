"""
Integration tests of test_algo_inference
"""


# Standard imports
import os
import numpy as np
import sys
from os.path import dirname


# Custom imports
root_directory = dirname(os.getcwd())
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

from mercs.algo.inference import *
from mercs.utils.utils import encode_attribute

import datasets as ds
from sklearn.preprocessing import Imputer


def test_perform_imputation():
    # Prelims
    train, test = ds.load_nursery()
    query_code = [0, -1, -1, -1, -1, -1, 0, 0, 1]

    imputator = Imputer(missing_values='NaN',
                        strategy='most_frequent',
                        axis=0)
    obs = imputator.fit(train)

    # Actual test
    assert train.shape == obs.shape
    assert isinstance(obs, np.ndarray)

    boolean_missing = encode_attribute(0, [1], [2])

    for row in obs[:, boolean_missing].T:
        assert len(np.unique(x)) == 1
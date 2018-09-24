# Standard imports
import numpy as np
import os
import sys
from os.path import dirname

# Custom imports
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

from mercs.algo.induction import *


# Actual Things
def test_induce_clf():
    s = {'type': 'DT'}

    clf = induce_clf(s)

    assert isinstance(clf, DecisionTreeClassifier)


def test_induce_rgr():
    s = {'type': 'DT'}

    clf = induce_rgr(s)

    assert isinstance(clf, DecisionTreeRegressor)


def test_induce_model():
    s = {'type': 'DT'}
    is_nominal = np.array([0, 0, 1, 1])

    m_targ_1 = [0]
    m_targ_2 = [2]
    m_targ_3 = [2, 3]
    m_targ_4 = [0, 1]
    m_targ_5 = [1, 2]

    obs_1 = induce_model(s, is_nominal, m_targ_1)
    obs_2 = induce_model(s, is_nominal, m_targ_2)
    obs_3 = induce_model(s, is_nominal, m_targ_3)
    obs_4 = induce_model(s, is_nominal, m_targ_4)

    assert isinstance(obs_1, DecisionTreeRegressor)
    assert isinstance(obs_2, DecisionTreeClassifier)
    assert isinstance(obs_3, DecisionTreeClassifier)
    assert isinstance(obs_4, DecisionTreeRegressor)

    try:
        obs_5 = induce_model(s, is_nominal, m_targ_5)
    except TypeError:
        pass

    return

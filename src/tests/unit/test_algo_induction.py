# Standard imports
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



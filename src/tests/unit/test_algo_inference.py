# Standard imports
import os
import numpy as np
import sys
from os.path import dirname


# Custom imports
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

from mercs.algo.inference import *


def test_init_predictions():
    nb_rows, nb_cols = 10,10

    obs = init_predictions(nb_rows, nb_cols, type=np.float64)

    assert obs.shape == (nb_rows, nb_cols)
    assert isinstance(obs[0, 0], np.float64)
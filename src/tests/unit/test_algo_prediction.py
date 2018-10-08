# Standard imports
import os
import numpy as np
import sys
from os.path import dirname

# Custom imports
root_directory = dirname(dirname(dirname(dirname(__file__))))
for dname in {'src'}:
    sys.path.insert(0, os.path.join(root_directory, dname))

from mercs.algo.prediction_new import (mi_pred_algo,
                                       ma_pred_algo,
                                       mafi_pred_algo)


def mi_setup():
    nb_mods, nb_atts = 3, 5
    nb_qrys = nb_atts # One query per attribute

    m_codes = np.full((nb_mods, nb_atts), 0)
    m_codes[0, -1] = 1
    m_codes[1, -2] = 1
    m_codes[2, 0:3] = 1

    q_codes = np.eye(nb_atts, dtype=int)

    return nb_mods, nb_atts, nb_qrys, m_codes, q_codes


def ma_setup():
    nb_mods, nb_atts, nb_qrys, m_codes, q_codes = mi_setup()

    settings = {'param':    0.95,
                'its':      0.1}



    return nb_mods, nb_atts, nb_qrys, m_codes, q_codes, settings


def mafi_setup():
    nb_mods, nb_atts, nb_qrys, m_codes, q_codes, settings = ma_setup()

    fi = np.random.rand(nb_mods, nb_atts)
    fi[m_codes != 0] = 0

    settings['FI'] = fi

    return nb_mods, nb_atts, nb_qrys, m_codes, q_codes, settings


def test_mi_pred_algo():
    nb_mods, nb_atts, nb_qrys, m_codes, q_codes = mi_setup()

    mas, aas = mi_pred_algo(m_codes, q_codes)

    assert isinstance(mas, np.ndarray)
    assert isinstance(aas, np.ndarray)

    assert mas.shape == (nb_qrys, nb_mods)
    assert aas.shape == (nb_qrys, nb_atts)

    assert np.max(mas) == 1
    assert np.max(aas) == 1

    return


def test_ma_pred_algo():
    nb_mods, nb_atts, nb_qrys, m_codes, q_codes, settings = ma_setup()

    mas, aas = ma_pred_algo(m_codes, q_codes, settings)

    assert isinstance(mas, np.ndarray)
    assert isinstance(aas, np.ndarray)

    assert mas.shape == (nb_qrys, nb_mods)
    assert aas.shape == (nb_qrys, nb_atts)

    assert np.max(mas) == 1
    assert np.max(aas) == 1

    return


def test_mafi_pred_algo():
    nb_mods, nb_atts, nb_qrys, m_codes, q_codes, settings = mafi_setup()

    mas, aas = mafi_pred_algo(m_codes, q_codes, settings)

    assert isinstance(mas, np.ndarray)
    assert isinstance(aas, np.ndarray)

    assert mas.shape == (nb_qrys, nb_mods)
    assert aas.shape == (nb_qrys, nb_atts)

    assert np.max(mas) == 1
    assert np.max(aas) == 1

    return

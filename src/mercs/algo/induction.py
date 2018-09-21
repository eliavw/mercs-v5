from ..utils.keywords import *
import numpy as np

from sklearn.tree import *
from sklearn.ensemble import *


# Algorithms
def base_ind_algo(metadata, settings, m_targ):
    """
    Initialize a model.

    This only means initialization, not training.
    So the only thing that happens is deciding which model is going to be
    trained in a next step. E.g.; DT, RF, etc.

    Here, we mainly verify whether the targets are grouped correctly,
    i.e., are they nominal/numeric etc.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary of the MERCS model
    settings: dict
        Settings dictionary of the MERCS model
    m_targ: list, shape (nb_targ,)
        List of the indices of the attributes that will be targets of the model
        that will be trained afterwards.

    Returns
    -------

    """

    assert isinstance(m_targ, list)

    nb_mod = len(m_targ)
    assert nb_mod > 0

    is_nominal = metadata['is_nominal']

    m_list = [induce_model(settings, is_nominal, m_targ[i])
              for i in range(nb_mod)]

    return m_list


# Inducers
def induce_model(settings, is_nominal, m_targ):
    """
    Initialize classifier/regressor with correct settings

    :param settings:
    :param is_nominal:
    :param m_targ:
    :return:
    """

    if _only_nominal_targ(is_nominal, m_targ):
        model = induce_clf(settings)
    elif _only_numeric_targ(is_nominal, m_targ):
        model = induce_rgr(settings)
    else:
        msg = "Model with mixed targets {}".format(m_targ)
        raise TypeError(msg)

    return model


def induce_clf(s):
    """
    Induce a single classifier.

    Filters the parameters
    Initializes the actual model
    """

    type = s['type']
    params = {k:v for k,v in s.items() if not k in {'type', 'flatten'}}

    if type in kw_ind_trees():
        clf = DecisionTreeClassifier(**params)
    elif type in kw_ind_forests():
        clf = RandomForestClassifier(**params)
    else:
        msg = "Did nog recognize classifier type: {}".format(type)
        raise TypeError(msg)

    return clf


def induce_rgr(s):
    """
    Induce a single regressor.

    Filters the parameters
    Initializes the actual model
    """

    type = s['type']
    params = {k:v for k, v in s.items() if not {'type', 'flatten'}}

    if type in kw_ind_trees():
        rgr = DecisionTreeRegressor(**params)
    elif type in kw_ind_forests():
        rgr = RandomForestRegressor(**params)
    else:
        msg = "Did nog recognize regressor type: {}".format(type)
        raise TypeError(msg)

    return rgr


# Helpers
def _only_nominal_targ(is_nominal, m_targ):
    """
    Check whether given set of targ only contains nominal attributes.

    :param is_nominal:  Boolean array indicating if attribute is nominal
    :param m_targ:      [[idx of targ atts model 0], [idx of targ atts model 1]]
    :return:
    """
    return np.sum(is_nominal[m_targ]) == len(m_targ)


def _only_numeric_targ(is_nominal, m_targ):
    """
    Check whether given set of targ only contains numeric attributes.

    :param is_nominal:  Boolean array indicating if attribute is nominal
    :param m_targ:      [[idx of targ atts model 0], [idx of targ atts model 1]]
    :return:
    """

    return np.sum(is_nominal[m_targ]) == 0
 
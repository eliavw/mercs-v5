import numpy as np
import warnings

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.ensemble import *


def base_selection_algo(metadata, settings, target_atts_list=None):
    """
    Base selection strategy.

    This method implements the base selection strategy, when attributes are grouped together
    randomly into disjoint target sets. More specifically,
        - if sel_param < 1, each model predicts (100*sel_param)% of the dataset attributes
        - if sel_param >= 1, each model predicts exactly (sel_param) attributes
        - default: 1 model per attribute

    For each selection iteration (sel_its parameter), each attribute appears exactly once in the target set.
    For each model, all the attributes that are not in the target set constitute the descriptive set.

    Parameters
    ----------
    metadata: dict
        Dictionary that contains metadata of the training set
    settings: dict
        Dictionary of the settings of MERCS. Relevant settings are:
            1. settings['param']
            2. settings['its']
    target_atts_list: list, shape (nb_targ_atts, )
        List of indices of target attributes.

    Returns
    -------
    codes: np.ndarray, shape (nb_models, nb_atts)
        Two-dimensional array where each row encodes a single model.
    """

    nb_atts = metadata['nb_atts']
    param = settings['param']
    nb_partitions = settings['its']

    # If not specified, all attributes can appear as targets
    if target_atts_list is None:
        target_atts_list = list(range(nb_atts))
    # Otherwise, use only indicated attributes

    nb_target_atts = len(target_atts_list)

    if (param > 0) & (param < 1):
        nb_out_atts = int(np.ceil(param * nb_atts))
    elif (param >= 1) & (param < nb_atts):
        nb_out_atts = int(param)
    else:
        msg = """
        Impossible number of output attributes per model: {}\n
        This means the value of settings['selection']['param'] was set
        incorrectly.\n
        Re-adjusted to default; one model per attribute.
        """.format(param)
        warnings.warn(msg)
        nb_out_atts = 1

    # Number of models per partition
    nb_models_part = int(np.ceil(nb_target_atts/nb_out_atts))
    # Total number of models
    nb_models = int(nb_partitions*nb_models_part)

    # One code per model
    codes = [[]]*nb_models

    # We start with everything descriptive
    for tree in range(nb_models):
        codes[tree] = [0]*nb_atts

    for partition in range(nb_partitions):
        for attribute in target_atts_list:
            # Randomly pick up a model to assign the attribute to
            random_model = np.random.randint(nb_models_part)
            iter = 0
            # Move to the first model that can still have additional target attribute
            while np.sum(codes[partition * nb_models_part + random_model]) >= nb_out_atts:
                random_model = np.mod(random_model + 1, nb_models_part)
                iter = iter + 1
                # Avoiding infinite loop
                if iter > nb_models_part:
                    break
            codes[partition*nb_models_part + random_model][attribute] = 1

    codes = np.array(codes)

    return codes


def fi_selection_algo(metadata, settings, X, target_atts_list = None):
    fi_scores = get_fi_scores(X, target_atts_list, metadata)
    n_clusters = (int(settings['selection']['param']), 2)
    model = SpectralBiclustering(n_clusters = n_clusters,
                                 method='log')
    model.fit(fi_scores)
    cluster_labels = model.row_labels_
    codes = labels_to_codes(cluster_labels, target_atts_list)
    return codes


def get_fi_scores(X, target_atts_list, metadata):
    nb_atts = X.shape[1]
    nb_target_atts = len(target_atts_list)

    fi = [0]*nb_target_atts

    for att in range(nb_target_atts):
        if (att in metadata['att_types']['numerical']):
            rf = RandomForestRegressor(n_estimators = 30,
                                       max_features = 'auto')
        else:
            rf = RandomForestClassifier(n_estimators=30,
                                        max_features='auto')
        targets = list(range(nb_atts))
        targets.remove(att)
        rf.fit(X[targets], X[[att]])
        fi[att] = rf.feature_importances_

    F = np.zeros((nb_target_atts, nb_atts), float)
    for i in range(nb_target_atts):
        for j in range(nb_atts):
            if i > j:
                F[i][j] = fi[i][j]
            if i < j:
                F[i][j] = fi[i][j - 1]

    return F


def labels_to_codes(cluster_labels, target_atts_list):
    nb_models = np.max(cluster_labels) + 1
    nb_atts = cluster_labels.size
    codes = [[]]*nb_models

    # We start with everything descriptive
    for model in range(nb_models):
        codes[model] = [0] * nb_atts

    for model in range(nb_models):
        for att in range(nb_atts):
            if (cluster_labels[att] == model):
                codes[model][att] = 1
    return codes


def random_selection_algo(metadata, settings, target_atts_list = None):
    """
    A random selection algorithm, to evaluate the performance of both the prediction algorithms.

    """

    # Total number of attributes
    nb_atts = metadata['nb_atts']

    # If not specified, all attributes can appear as targets.
    # Otherwise, use only indicated attributes
    if target_atts_list is None:
        target_atts_list = list(range(nb_atts))

    # Number of possible targets
    nb_target_atts = len(target_atts_list)
    sel_param, sel_its = settings['param'], settings['its']

    # Number of output attributes per model
    if sel_param >= 0.4:
        nb_out_atts = int(np.ceil(sel_param))
    else:
        perc_targ_atts = sel_param
        nb_out_atts = int(np.ceil(perc_targ_atts * nb_target_atts))

    # Number of models
    nb_models = int(np.ceil(nb_target_atts / nb_out_atts)) * sel_its
    # One code per model
    codes = [[]] * nb_models

    for it in range(nb_models):
        # Varying the number of desc atts
        nb_desc_atts = np.random.randint(nb_out_atts, nb_atts - nb_out_atts)
        # Setting missing attributess
        code = [-1] * nb_atts
        # Setting target attributes
        for i in range(0, nb_out_atts): code[i] = 1
        # Setting desc attributes
        for i in range(nb_out_atts, nb_out_atts + nb_desc_atts + 1):
            code[i] = 0
        np.random.shuffle(code)
        codes[it] = code

    codes = np.array(codes)

    # Now we repair after possible deficiencies
    # Counts of 'being target'
    occ_as_targ = [np.count_nonzero(codes[:, i] == 1) for i in range(codes.shape[1])]
    mean_occ_as_target = int(np.ceil(np.mean(occ_as_targ)))

    for i, v in enumerate(occ_as_targ):
        if v == 0:
            models_to_alter = np.random.randint(1, codes.shape[0], size=mean_occ_as_target)
            for m in models_to_alter: codes[m, i] = 1

    return codes

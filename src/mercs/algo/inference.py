import numpy as np
import pandas as pd
from ..utils.utils import encode_attribute


# Imputation
def perform_imputation(test_data_df, query_code, imputator):
    """
    Creates the test data for a given query.

    This means that it sets the unknown attributes first to NaN and afterwards
    imputes them.

    :param test_data_df:    DataFrame that contains the test portion of the
                            dataset. With all attributes.
    :param query_code:      Code that conveys the functions of all the
                            attributes.
    :param imputator:       The thing used to impute, sklearn standard.
    :return:
    """
    assert isinstance(test_data_df, pd.DataFrame)
    assert isinstance(query_code, (np.ndarray, list))
    assert len(test_data_df.columns.values) == len(query_code)

    missing_attribute_encoding = encode_attribute(0, [1], [2])
    query_data_df = test_data_df.copy()

    for i, v in enumerate(query_code):
        if v == missing_attribute_encoding:
            query_data_df.iloc[:, i] = np.nan

    query_data = imputator.transform(query_data_df)
    return query_data


# Merging outcomes
def merge_proba(proba_res,
                proba_mod,
                lab_res,
                lab_mod,
                t_idx_res,
                t_idx_mod,
                nb_targ=1):
    """
    Merge the probabilities from a single model with another array containing
    all probabilities.

    This based on the indices passed to this function. Also, take into account
    that proba array of a single model possibly relies on other classlabels.

    :param proba_res:           Datastructure to contain the result (proba)
    :param proba_mod:           Datastructure to contain the result
                                of the current model (proba)
    :param lab_res:             Classlabels of the result
    :param lab_mod:             Classlabels of the model
    :param t_idx_res:           Index of current target attr in result
    :param t_idx_mod:           Index of current target attr in  current model
    :param nb_targ:             Number of targets of the model
    :return:
    """
    # TODO: Fix the hotfix

    mask = _get_mask(lab_res, lab_mod, t_idx_res, t_idx_mod)

    # Single target case
    if nb_targ == 1:
        # This means it comes from a model WE made.
        if isinstance(proba_mod, list):
            proba_mod = proba_mod[0]                                # Hotfix.

        proba_res[t_idx_res][:, mask] += proba_mod

    # Multi-target case
    else:
        proba_res[t_idx_res][:, mask] += proba_mod[t_idx_mod]

    return proba_res


def merge_pred(pred_res, pred_mod, t_idx_res, t_idx_mod, nb_targ):
    """
    Merge non-probabilistic predictions.

    :param pred_res:
    :param pred_mod:
    :param t_idx_res:
    :param t_idx_mod:
    :param nb_targ:
    :return:
    """

    if isinstance(pred_mod, list):
        # This means it comes from a model WE made.
        # TODO(elia): We should probably rely completely on np.array too, since it is better.
        pred_res[t_idx_res] += pred_mod[t_idx_mod]
        return pred_res
    elif nb_targ == 1:
        # Single target sklearn output (needs reformatting, yields only np.array)
        broadcast = np.atleast_2d(pred_mod).T

        pred_res[t_idx_res] += broadcast[:, [t_idx_mod]]
        del broadcast
        return pred_res
    else:
        # Multi-target sklearn np.array (does not need reformat)
        pred_res[t_idx_res] += pred_mod[:, [t_idx_mod]]
        return pred_res


# Converting to actual output values
def predict_values_from_proba(proba_res, lab_res):
    """
    Convert probabilities of outcomes to actual labels.

    Parameters
    ----------
    proba_res: list of np.ndarray, shape (targets, (samples, classlabels))
        Probabilities of all the classes of all the targets of
        the result.

    lab_res: list of np.ndarray, shape (targets, (classlabels,))
        Classlabels of all the targets of the result.

    Returns
    -------
    """

    nb_samples = proba_res[0].shape[0]
    nb_attribs = len(proba_res)
    predictions = init_predictions(nb_samples, nb_attribs)

    assert nb_attribs == len(lab_res)
    for i in range(nb_attribs):
        my_result = lab_res[i].take(np.argmax(proba_res[i], axis=1), axis=0)
        np.rint(my_result)
        predictions[:, i] = my_result

    return predictions.astype(int)


def predict_values_from_numer(numer_res, counts):
    """
    Average numeric predictions

    :param numer_res:       Sum of numeric predictions
    :param counts:          Amount of predictions that was summed
    :return:
    """
    nb_samples = numer_res[0].shape[0]
    nb_attribs = len(numer_res)
    predictions = init_predictions(nb_samples, nb_attribs)

    assert nb_attribs == len(counts)
    for i in range(nb_attribs):
        my_result = numer_res[i] / counts[i]
        predictions[:, [i]] = my_result

    return predictions


# Utilities
def update_X(X, Y, act_att_idx):
    """
    Replace values in X with new values given by Y, in the correct places.

    The array act_att_idx tells us what is contained in the Y-array.
    When we enumerate act_att_idx, the index gives us the corresponding
    column in Y, whereas the value tells us the corresponding column in X.

    :param X:               Matrix X
    :param Y:               Matrix Y
    :param act_att_idx:     Attributes contained in Y
    :return:
    """
    assert len(X.shape) == len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] >= Y.shape[1]
    assert Y.shape[1] == len(act_att_idx)
    assert X.shape[1] >= np.max(act_att_idx)

    for y_idx, attr_idx in enumerate(act_att_idx):
        X[:, attr_idx] = Y[:, y_idx]
    return X


def init_predictions(nb_rows, nb_cols, type=np.float64):
    """
    Initialize an empty array to contain our results.

    This is in a separate method because it can be influential
    and occurs in multiple places in our code.

    We want consistency to easily locate eventual bugs.

    :param type:        Type of entries in the np.ndarray.
    :param nb_rows:
    :param nb_cols:
    :return:
    """
    return np.zeros((nb_rows, nb_cols), dtype=type)


# Internal methods
def _get_mask(lab_res, lab_mod, t_idx_res, t_idx_mod):
    """
    Check which classlabels predicted by the model also occur in the
    classlabels in the result

    This is easily achieved with the np.isin which yields a boolean mask.

    :param lab_res:         Classlabels of the result
    :param lab_mod:         Classlabels of the model
    :param t_idx_res:       Index of the current target in result
    :param t_idx_mod:       Index of the current target in current model
    :return:
    """

    mask = np.isin(lab_res[t_idx_res], lab_mod[t_idx_mod], assume_unique=True)

    return mask

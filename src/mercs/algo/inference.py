import numpy as np
from ..utils.utils import collect_classlabels


# Imputation
def perform_imputation(test_data_df, query_code, imputator):
    """
    Creates the test data_csv for a given queries.

    This means that it sets the unknown attributes first to NaN and then imputes them.

    :param test_data_df:
    :param query_code:
    :param imputator:
    :return:
    """

    query_data_df = test_data_df.copy()

    for i, v in enumerate(query_code):
        if v == -1:
            query_data_df.iloc[:, i] = np.nan

    query_data = imputator.transform(query_data_df)
    return query_data


## Merging outcomes
def merge_proba(res_proba,
                mod_proba,
                t_idx_res,
                t_idx_mod,
                res_lab,
                mod_lab,
                nb_target=1):
    """
    Add mod_proba to the correct entry in all_proba.

    This based on the indices passed to this function.

    Also, take into account that mod_proba possibly relies on other classlabels.

    :param res_proba:          Datastructure to contain the result (proba)
    :param mod_proba:          Output of the current model (proba)
    :param t_idx_res:          Index of current target attr in result
    :param t_idx_mod:          Index of current target attr in  current model
    :param nb_target:          Number of targets of the model
    :return:
    """

    mask = get_mask(res_lab, mod_lab, t_idx_res, t_idx_mod)

    if nb_target == 1:
        if type(mod_proba) is list:
            mod_proba = mod_proba[0]                        # Hotfix. TODO: DO THIS MORE NICELY

        res_proba[t_idx_res][:, mask] += mod_proba          # Single target case
    else:
        res_proba[t_idx_res][:, mask] += mod_proba[t_idx_mod]   # Multiple target case

    return res_proba


def merge_pred(res_pred, mod_pred, t_idx_res, t_idx_mod, nb_targ):
    """
    Merge non-probabilistic predictions

    :param res_pred:
    :param mod_pred:
    :param t_idx_res:
    :param t_idx_mod:
    :param nb_targ:
    :return:
    """

    if type(mod_pred) is list:
        # This means it comes from a model WE made.
        # TODO(elia): We should probably rely completely on np.array too, since it is better.
        res_pred[t_idx_res] += mod_pred[t_idx_mod]
        return res_pred
    elif nb_targ == 1:
        # Single target sklearn output (needs reformatting)
        broadcast = np.atleast_2d(mod_pred).T

        res_pred[t_idx_res] += broadcast[:,[t_idx_mod]] # Single target sklearn yields only np.array
        del broadcast
        return res_pred
    else:
        # Multi-target sklearn np.array (does not need reformat)
        res_pred[t_idx_res] += mod_pred[:,[t_idx_mod]]
        return res_pred


## Converting to values
def predict_values_from_proba(res_proba, res_lab):
    """
    Convert probabilities of outcomes to actual labels

    :param res_proba:   Probabilities of all the classes of all the targets of
                        the result.
    :param res_lab:     Classlabels of all the targets of the result.
    :return:
    """

    assert len(res_proba) == len(res_lab)
    nb_samples = res_proba[0].shape[0]
    nb_targets = len(res_proba)

    predictions = init_predictions(nb_samples, nb_targets)

    for i in range(nb_targets):
        my_result = res_lab[i].take(np.argmax(res_proba[i], axis=1), axis=0)
        np.rint(my_result)
        predictions[:, i] = my_result

    return predictions.astype(int)


def predict_values_from_numer(res_numer, counts):
    """
    Average numeric predictions

    :param res_numer:       Sum of numeric predictions
    :param counts:          Amount of predictions that was summed
    :return:
    """

    assert len(res_numer) == len(counts)
    nb_samples = res_numer[0].shape[0]
    nb_targets = len(res_numer)

    predictions = init_predictions(nb_samples, nb_targets)

    for i in range(nb_targets):
        my_result = res_numer[i] / counts[i]
        predictions[:, [i]] = my_result

    return predictions


# Helpers
def init_predictions(nb_rows, nb_columns):
    """
    Initialize an empty array to contain our results.

    This is in a separate method because it can be influential
    and occurs in multiple places in our code.

    We want consistency to easily locate eventual bugs.

    :param nb_rows:
    :param nb_columns:
    :return:
    """
    return np.zeros((nb_rows, nb_columns), dtype=np.float64)


def update_X(X, Y, act_att_idx):
    for i, v in enumerate(act_att_idx):
        X[:, v] = Y[:, i]
    return X


def get_mask(res_lab, mod_lab, t_idx_res, t_idx_mod):
    """
    Check which labels in mod_lab also occur in res_lab.

    This is easily achieved with the np.isin which yields a boolean mask.

    :param res_lab:     Classlabels of the result
    :param mod_lab:     Classlabels of the model
    :param t_idx_res:   Index of the current target in result
    :param t_idx_mod:   Index of the current target in current model
    :return:
    """

    mask = np.isin(res_lab[t_idx_res], mod_lab[t_idx_mod], assume_unique=True)

    return mask

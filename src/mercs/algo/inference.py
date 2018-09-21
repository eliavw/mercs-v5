import numpy as np
import pandas as pd

from ..utils.encoding import encode_attribute

from ..utils.debug import debug_print
VERBOSITY = 0


# Imputation
def perform_imputation(test_data_df, query_code, imputator):
    """
    Creates the test data for a given query.

    This means that it sets the unknown attributes first to NaN and afterwards
    imputes them.

    Parameters
    ----------
    test_data_df: DataFrame, shape (nb_samples, nb_attributes)
        Contains the test portion of the dataset. With all attributes.
    query_code: list, shape (nb_attributes, )
        Code that conveys the functions of all the attributes.
    imputator: sklearn.preprocessing.imputation.Imputer
        The thing used to impute, sklearn standard.

    Returns
    -------

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
def merge_proba(proba_res, proba_mod, lab_res, lab_mod, t_idx_res, t_idx_mod):
    """
    Merge probabilistic outcomes from a single model with the current result.

    This based on the indices passed to this function. Also, take into account
    that proba array of a single model possibly relies on other classlabels.

    Parameters
    ----------
    proba_res: list, shape (nb_targets, (nb_samples, nb_labels))
        Contains the result (proba)
    proba_mod: {list, np.ndarray}, shape    (nb_targets, (nb_samples, nb_labels))
                                            (nb_samples, nb_labels)
        Contains the result of the current model (proba)
    lab_res
        Classlabels of the result
    lab_mod
        Classlabels of the model
    t_idx_res: int
        Index of current target attr in result
    t_idx_mod: int
        Index of current target attr in  current model

    Returns
    -------

    Legacy
    -------
    Legacy version, keeping it here for the unlikely case something's off.
    # Single target case
    if nb_targ == 1:
        # This means it comes from a model WE made.
        if isinstance(proba_mod, list):
            proba_mod = proba_mod[0]

        proba_res[t_idx_res][:, mask] += proba_mod

    # Multi-target case
    else:
        proba_res[t_idx_res][:, mask] += proba_mod[t_idx_mod]

    """

    assert isinstance(proba_res, list)
    assert isinstance(proba_mod, (list, np.ndarray))

    mask = _get_mask(lab_res, lab_mod, t_idx_res, t_idx_mod)

    if isinstance(proba_mod, np.ndarray):
        # This means a single-target sklearn output
        proba_mod = [proba_mod]

    assert isinstance(proba_mod[t_idx_mod], np.ndarray)
    assert proba_res[t_idx_res][:, mask].shape == proba_mod[t_idx_mod].shape
    proba_res[t_idx_res][:, mask] += proba_mod[t_idx_mod]

    return proba_res


def merge_numer(numer_res, numer_mod, t_idx_res, t_idx_mod):
    """
    Merge non-probabilistic predictions.

    Parameters
    ----------
    numer_res: [np.ndarray], shape (nb_targ_total, (nb_samples, 1))
        Predictions of a higher-order model. This method is quite oblivious
        to this array. The only thing that matters is that we add the data
        from the single model to the correct entries (correct columns) in
        this bigger array.
    numer_mod: {list, np.ndarray}, shape (nb_samples, nb_targ_mod)
        Predictions of a single model.
        This is a list if it comes from a model we made, and an array if
        it comes directly from sklearn.
    t_idx_res: int
        Index of the column that corresponds to the target attribute t in
        the array pred_res
    t_idx_mod: int
        Index of the column that corresponds to the target attribute t in
        the array pred_mod

    Returns
    -------

    """
    # TODO(elia): Own models might also better provide np.ndarray...

    con_1 = (type(numer_res), type(numer_res[0]), numer_res[0].shape)
    msg_1 = "Type of numer_res: {} \n" \
          "And type of numer_res[0]: {}\n" \
          "And shape of numer_res[0]: {}\n".format(*con_1)
    debug_print(msg_1, level=1, V=VERBOSITY)

    if isinstance(numer_mod, list):
        # Happens when it is a model WE built ourselves
        broadcast = np.squeeze(np.atleast_2d(numer_mod).T)
    elif isinstance(numer_mod, np.ndarray):
        if len(numer_mod.shape) < 2:
            # Single target sklearn output (needs reformatting, yields shape(x,))
            broadcast = np.atleast_2d(numer_mod).T
        else:
            broadcast = numer_mod
    else:
        msg = "numer_mod has wrong type. This method needs a list or a np.ndarray"
        raise TypeError(msg)

    assert len(numer_res) > 0
    assert broadcast.shape[0] == numer_res[0].shape[0]
    assert numer_res[t_idx_res].shape == broadcast[:, [t_idx_mod]].shape

    numer_res[t_idx_res] += broadcast[:, [t_idx_mod]]
    del broadcast
    return numer_res


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

    Parameters
    ----------
    lab_res:
        Classlabels of the result
    lab_mod:
        Classlabels of the model
    t_idx_res: int
        Index of the current target in result
    t_idx_mod: int
        Index of the current target in current model
    Returns
    -------

    """

    mask = np.isin(lab_res[t_idx_res], lab_mod[t_idx_mod], assume_unique=True)

    return mask

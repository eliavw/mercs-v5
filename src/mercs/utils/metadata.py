import numpy as np
import pandas as pd

from .encoding import codes_to_query


def get_metadata_df(df):
    """
    Get some useful statistics from a Pandas DataFrame.

    We extract:
        1) nb_samples
            The total number of rows in the DataFrame.
        2) nb_atts
            The total number of columns in the DataFrame.
        3) is_nominal
            Type of the attribute (numeric/nominal)
        4) nb_uvalues
            Number of unique values in case of a nominal attribute.

    N.b.:   For an attribute to be considered nominal,
            it should at least be an integer.


    Parameters
    ----------
    df: pd.DataFrame
        Data is given in the form of a pandas DataFrame

    Returns
    -------

    """
    # TODO(elia) is_nominal should become is_type

    nb_tuples = df.shape[0]
    nb_atts = df.shape[1]

    nb_uvalues = df.nunique().values
    has_nan = df.isnull().any().values
    types = df.dtypes.values

    is_nominal = [check_nominal_att(t, v, n)
                  for t, v, n in zip(types, nb_uvalues, has_nan)]
    is_nominal = np.array(is_nominal).astype(int)

    metadata = {'types':        types,
                'has_nan':      has_nan,
                'is_nominal':   is_nominal,
                'nb_atts':      nb_atts,
                'nb_tuples':    nb_tuples,
                'nb_values':    nb_uvalues}

    return metadata


def check_nominal_att(attribute_type,
                      attribute_unique_values,
                      check_nan,
                      nominal_attribute_unique_values_threshold=20):
    """
    Check if an attribute is nominal, according to our definition.

    Def:
        A nominal attribute is an attribute which is of an integer type and
        has fewer than nominal_attribute_unique_values_threshold distinct values.

    Parameters
    ----------
    attribute_type: numpy.dtype
        Type of the attribute, as extracted from the DataFrame
    attribute_unique_values: np.ndarray
        Unique values of the attribute under consideration
    check_nan: boolean
        Whether or not the attribute has some np.nan values
    nominal_attribute_unique_values_threshold: int
        When the number of distinct values is equal to or exceeds this value,
        we no longer regard it as a nominal attribute.

        Default = 20

    Returns
    -------

    """

    check_uvalues = attribute_unique_values < nominal_attribute_unique_values_threshold
    check_integer_type = pd.api.types.is_integer_dtype(attribute_type)
    check_float_type = pd.api.types.is_float_dtype(attribute_type)

    check_case_one = check_integer_type and check_uvalues
    check_case_two = check_float_type and check_nan and check_uvalues

    if check_case_one:
        return True
    elif check_case_two:
        return True
    else:
        return False


def collect_feature_importances(m_list, m_codes):
    """
    Collect the feature importance of the models.

    :param m_codes:
    :param m_list:
    :return:
    """

    feature_importances = np.zeros(m_codes.shape)
    m_desc, _, _ = codes_to_query(m_codes)

    for mod_i in range(len(m_list)):
        for desc_i, attr_i in enumerate(m_desc[mod_i]):
            feature_importances[mod_i, attr_i] = m_list[mod_i].feature_importances_[desc_i]

    return feature_importances
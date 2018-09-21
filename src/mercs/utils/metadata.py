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

    # Initialize arrays (-1/False are no arbitrary choices!)
    nb_uvalues = df.nunique().values
    is_nominal = np.full(nb_atts, 0)
    tr_nominal = 20 # Our threshold of nb classes to call a column 'nominal'

    # Fill when necessary
    for col_idx, col in enumerate(df.columns):
        if pd.api.types.is_integer_dtype(df[col]):
            is_nominal[col_idx] = 1 if nb_uvalues[col_idx] < tr_nominal else 0

    metadata = {'is_nominal':   is_nominal,
                'nb_atts':      nb_atts,
                'nb_tuples':    nb_tuples,
                'nb_values':    nb_uvalues}

    return metadata


def collect_FI(m_list, m_codes):
    """
    Collect the feature importance of the models.

    :param m_codes:
    :param m_list:
    :return:
    """

    FI = np.zeros(m_codes.shape)
    m_desc, _, _ = codes_to_query(m_codes)

    for mod_i in range(len(m_list)):
        for desc_i, attr_i in enumerate(m_desc[mod_i]):
            FI[mod_i, attr_i] = m_list[mod_i].feature_importances_[desc_i]

    return FI
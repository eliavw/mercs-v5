import pandas as pd


def save_output_data(Y, q_targ, filename):
    """
    Write predictions to file

    Parameters
    ----------
    Y: np.ndarray, shape (nb_samples, nb_targ_attributes)
        Numpy array of results
    q_targ: list, shape (nb_targ,)
        List of integers which are the indices of the attributes that
        compose the target set of attributes of the current prediction.

        These are used to compose a default header for the output csv file.
    filename: str
        Filename where the results should be saved

    Returns
    -------

    """
    """
    Given filename + Numpy array of predictions, write predictions to file.

    :param Y:               Numpy array of results
    :param q_targ:          Query code
    :param filename:        Filename of where to save results
    :return:
    """

    # Building header (names of target attributes)
    header = _compose_default_header(q_targ)

    df = pd.DataFrame(Y)
    df.to_csv(filename, header=header, index=False)

    return


def _compose_default_header(attributes):
    """
    Compose default header for the csv file.

    Parameters
    ----------
    attributes: list, shape (nb_targ,)
        List of integers which are the integers of the attributes that
        should be mentioned in the header.

    Returns
    -------
    """

    header = ["feature {}".format(i) for i in attributes]
    return header
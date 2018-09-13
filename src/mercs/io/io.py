import pandas as pd


def save_output_data(Y, q_targ, filename):
    """
    Given filename + Numpy array of predictions, write predictions to file.

    :param Y:               Numpy array of results
    :param q_targ:          Query code
    :param filename:        Filename of where to save results
    :return:
    """

    # Building header (names of target attributes)
    header = ["feature {}".format(i) for i in q_targ]

    df = pd.DataFrame(Y)
    df.to_csv(filename, header=header, index=False)

    return

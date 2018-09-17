# Python scripts to handle some example datasets
import os
import sys
from os.path import dirname
import pandas as pd

# Custom import (Add src to the path)
root_directory = dirname(dirname(dirname(__file__)))
for dname in {'src'}:
    sys.path.append(os.path.join(root_directory, dname))
from src.mercs.utils.debug import debug_print

VERBOSITY = 1

# Actual method
def load_example_dataset(dataset_name, specifier=None, extension=None):
    """
    Load example dataset (.data_csv) with given name.

    :param dataset_name:    Name of example dataset
    :param specifier:       {'Test', 'Train}
    :return:                Pandas Dataframe with the dataset
    """
    root_dir = dirname(dirname(dirname(__file__)))
    data_dir = os.path.join(root_dir, 'resc', 'data')

    extension = '.csv' if extension is None else '.' + extension

    appendix = '' if specifier is None else '_' + specifier

    fname = os.path.join(data_dir, dataset_name + appendix + extension)

    msg = "load_example_dataset is loading fname: {}".format(fname)
    debug_print(msg, level=1, V=VERBOSITY)

    df = pd.read_csv(fname)
    assert isinstance(df, pd.DataFrame)
    return df

# Custom shortcuts
def load_fertility():
    """
    Load the fertility dataset.

    This dataset contains both nominal and numeric attributes.

    :return:
    """

    train = load_example_dataset('fertility', 'train')
    test = load_example_dataset('fertility', 'test')

    return train, test


def load_jester():
    """
    Load the jester dataset

    :return:
    """

    train = load_example_dataset('jester', 'train')
    test = load_example_dataset('jester', 'test')

    return train, test


def load_netflix():
    """
    Load the netflix dataset

    :return:
    """

    train = load_example_dataset('netflix', 'train')
    test = load_example_dataset('netflix', 'test')

    return train, test


def load_nursery():
    """
    Load the nursery dataset

    :return:
    """

    train = load_example_dataset('nursery', 'train')
    test = load_example_dataset('nursery', 'test')

    return train, test


def load_slump():
    """
    Load the slump dataset

    :return:
    """

    train = load_example_dataset('slump', 'train')
    test = load_example_dataset('slump', 'test')

    return train, test


def load_fertility_full():
    """
    Load the full fertility.data_csv file.

    :return:    Pandas DataFrame of this dataset.
    """
    return load_example_dataset('fertility')


def load_jester_full():
    """
    Load the full jester.data_csv file.

    :return:    Pandas DataFrame of this dataset.
    """
    return load_example_dataset('jester')


def load_netflix_full():
    """
    Load the full netflix.data_csv file.

    :return:    Pandas DataFrame of this dataset.
    """
    return load_example_dataset('netflix')


def load_nursery_full():
    """
    Load the full nursery.data_csv file.

    :return:    Pandas DataFrame of this dataset.
    """
    return load_example_dataset('nursery')


def load_slump_full():
    """
    Load the full slump.data_csv file.

    :return:    Pandas DataFrame of this dataset.
    """
    return load_example_dataset('slump')

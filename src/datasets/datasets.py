# Python scripts to handle some example datasets
import os
import sys
from os.path import dirname
import pandas as pd

# Custom import (Add src to the path)
root_directory = dirname(dirname(dirname(__file__)))
for dname in {'src'}:
    sys.path.append(os.path.join(root_directory, dname))
from mercs.utils.debug import debug_print

VERBOSITY = 0


# Actual method
def load_example_dataset(dataset_name, specifier=None, extension=None):
    """
    Load example dataset (.csv) with given name.

    Parameters
    ----------
    dataset_name: string
        Name of the example dataset
    specifier: {string, NoneType}
        x in  {'Train', 'Test', None}
    extension: string
        Extension of the filename

    Returns
    -------
    df: pd.DataFrame
        DataFrame of the example dataset

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


# Shortcuts
def load_fertility(full=False):
    """
    Load the fertility dataset.

    This dataset contains both nominal and numeric attributes.

    :return:
    """

    if full:
        return load_example_dataset('fertility')
    else:
        train = load_example_dataset('fertility', 'train')
        test = load_example_dataset('fertility', 'test')

        return train, test


def load_jester(full=False):
    """
    Load the jester dataset

    Nominal dataset

    :return:
    """

    if full:
        return load_example_dataset('jester')
    else:
        train = load_example_dataset('jester', 'train')
        test = load_example_dataset('jester', 'test')

        return train, test


def load_netflix(full=False):
    """
    Load the netflix dataset

    Nominal dataset

    :return:
    """

    if full:
        return load_example_dataset('netflix')
    else:
        train = load_example_dataset('netflix', 'train')
        test = load_example_dataset('netflix', 'test')

        return train, test


def load_nursery(full=False):
    """
    Load the nursery dataset

    Nominal dataset

    :return:
    """

    if full:
        return load_example_dataset('nursery')
    else:
        train = load_example_dataset('nursery', 'train')
        test = load_example_dataset('nursery', 'test')

        return train, test


def load_slump(full=False):
    """
    Load the slump dataset

    Numeric dataset

    :return:
    """

    if full:
        return load_example_dataset('slump')
    else:
        train = load_example_dataset('slump', 'train')
        test = load_example_dataset('slump', 'test')

        return train, test


# Python scripts to handle some example data
import os
from os.path import dirname
import pandas as pd

# Actual method
def load_example_dataset(dataset_name, specifier=None):
    """
    Load example dataset (.data_csv) with given name.

    :param dataset_name:    Name of example dataset
    :param specifier:       {'Test', 'Train}
    :return:                Pandas Dataframe with the dataset
    """
    current_dir = dirname(__file__)
    data_dir = os.path.join(current_dir, 'data_csv')
    appendix = '' if specifier is None else '_' + specifier

    df = pd.read_csv(os.path.join(data_dir, dataset_name + appendix + '.data_csv'))
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

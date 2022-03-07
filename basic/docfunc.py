from numpy import ndarray
from basic.types import Sequence, Bow
from typing import NoReturn


def lighten_args(*args):
    assert tuple(*args) is tuple


def en_VonMisesFisher(model_import: ndarray, level: float) -> NoReturn:
    """
    Von-Mises Fisher distribution instantiation

    :param model_import: ndarray, import matrix-like data for creating Von-Mises Fisher distribution model
    :param level: float from 0 to 1, default value uses 0.05, level of significance test
    """
    lighten_args(model_import, level)


def en_VonMisesFisher_predict(data_import: ndarray) -> ndarray:
    """
    make prediction whether data was of the created Von-Mises Fisher distribution

    :param data_import: ndarray, import matrix-like data to be predicted
    :return: ndarray, a bool sequence
    """
    lighten_args(data_import)


def en_Hotelling(model_import: ndarray, level: float) -> NoReturn:
    """
    Hotelling T2 statistics instantiation

    :param model_import: ndarray, import matrix-like data for creating Hotelling T2 statistics
    :param level: float from 0 to 1, default value uses 0.05; level of significance test
    """
    lighten_args(model_import, level)


def en_Hotelling_predict(data_import: ndarray) -> ndarray:
    """
    make prediction whether data was of the created Hotelling T2 statistics

    :param data_import: ndarray, import matrix-like data to be predicted
    :return: ndarray, a bool sequence
    """
    lighten_args(data_import)


def en_NaiveBayes(gamma: float, validation_rate: float) -> NoReturn:
    """
    Naive Bayes distribution instantiation

    :param gamma: float from 0 to 1, default value uses 1; a parameter uses uniform as prior distribution
    :param validation_rate: float from 0 to 1, default value uses 0.2; parameter for valid set rate
    """
    lighten_args(gamma, validation_rate)


def en_NaiveBayes_fit(x: Sequence[Bow], y: ndarray) -> NoReturn:
    """
    fit data based on created Naive Bayes distribution

    :param x: ndarray of Bow, import sparse representation data to be fitted
    :param y: ndarray, labels for sparse representation data x
    """
    lighten_args(x, y)


def en_NaiveBayes_predict(x: Sequence[Bow]) -> ndarray:
    """
    make prediction based on the fitted Naive Bayes distribution

    :param x: ndarray of Bow, import sparse representation data to be predicted
    :return: ndarray, a sequence of predicted labels
    """
    lighten_args(x)


def en_NaiveBayes_update(x: Sequence[Bow], y: ndarray) -> NoReturn:
    """
    update existed Naive Bayes distribution based on new import data

    :param x: ndarray of Bow, import sparse representation data to be updated
    :param y: ndarray, labels for sparse representation data x
    """
    lighten_args(x, y)


if __name__ == '__main__':
    pass

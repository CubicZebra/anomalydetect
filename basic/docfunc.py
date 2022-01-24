from numpy import ndarray
from basic.types import Sequence, Bow


def lighten_args(*args):
    assert tuple(*args) is tuple


def en_VonMisesFisher(model_import: ndarray, level: float):
    """
    Von-Mises Fisher distribution instantiation

    :param model_import:
    :param level:
    :return:
    """
    lighten_args(model_import, level)


def en_VonMisesFisher_predict(data_import: ndarray) -> ndarray:
    """

    :param data_import:
    :return:
    """
    lighten_args(data_import)


def en_Hotelling(model_import: ndarray, level: float):
    """

    :param model_import:
    :param level:
    :return:
    """
    lighten_args(model_import, level)


def en_Hotelling_predict(data_import: ndarray) -> ndarray:
    """

    :param data_import:
    :return:
    """
    lighten_args(data_import)


def en_NaiveBayes(gamma: float, validation_rate: float):
    """

    :param gamma:
    :param validation_rate:
    :return:
    """
    lighten_args(gamma, validation_rate)


def en_NaiveBayes_fit(x: Sequence[Bow], y: ndarray):
    """

    :param x:
    :param y:
    :return:
    """
    lighten_args(x, y)


def en_NaiveBayes_predict(data_import: Sequence[Bow]) -> ndarray:
    """

    :param data_import:
    :return:
    """
    lighten_args(data_import)


def en_NaiveBayes_update(x: Sequence[Bow], y: ndarray):
    """

    :param x:
    :param y:
    :return:
    """
    lighten_args(x, y)


if __name__ == '__main__':
    pass

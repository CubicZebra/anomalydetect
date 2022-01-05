from numpy import ndarray


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


if __name__ == '__main__':
    pass

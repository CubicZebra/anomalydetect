import copy
import numpy as np
from typing import Tuple, Callable
from scipy.stats import chi2
from pynverse import inversefunc
from basic.decorators import type_checker, document
from basic.types import vector, matrix, Configuration, TP_VONMISESFISHER, available_types
import basic.docfunc as doc


def unitize(x: vector) -> vector:
    """scalarize a vector"""
    return x/(np.linalg.norm(x, ord=2))


def miu(x: matrix) -> vector:
    """mean direction of samples"""
    assert len(x.shape) == 2, 'arg x must be matrix-like'
    x = x.mean(axis=0)
    return unitize(x)


def a(x: matrix, mean: vector) -> vector:
    """anomalies of samples: 0~2, from perfectly coincident to totally opposite direction"""
    x_reg = np.array([unitize(x[i]) for i in range(len(x))])
    return np.array([1-np.dot(mean, x_reg[i]) for i in range(len(x_reg))])


def _moment_2_origin(x: vector) -> float:
    return np.multiply(x, x).mean()


def _chi2_params(x: vector) -> Tuple[float]:
    m1, m2 = x.mean(), _moment_2_origin(x)
    df = 2*(m1**2) / (m2-m1**2)
    scale = (m2-m1**2) / (2*m1)
    return df, scale


def _upper_domain(func: Callable, scale: float):
    upper = scale
    while True:
        try:
            res = inversefunc(func, y_values=1, domain=[0, upper])
        except ValueError:
            upper += scale  # heuristically increase upper of domain
        else:
            return res


def chi2_threshold(x: matrix, level: float = 0.05) -> float:
    """calculate the threshold of chi2 distribution, by a given test level"""
    df, scale = _chi2_params(a(x, miu(x)))
    chi2_dis = chi2(df=df, scale=scale)

    upper = _upper_domain(chi2_dis.cdf, scale)
    return inversefunc(chi2_dis.cdf, y_values=1-level, domain=[0, upper])


@document(doc.en_VonMisesFisher)
class VonMisesFisher:

    settings: Configuration = {
        'model_import': np.array([]),  # ndarray, matrix-like
        'level': 0.05,  # float, 0~1
        'data_import': np.array([]),  # ndarray, matrix-like
    }

    @type_checker(in_class=True, kwargs_types=TP_VONMISESFISHER, elemental_types=available_types)
    def __init__(self, **settings: Configuration):
        for k, v in settings.items():
            self.settings.update({k: v})
        self.mean = miu(self.settings.get('model_import'))
        self.a = a(self.settings.get('model_import'), self.mean)
        self.threshold = chi2_threshold(self.settings.get('model_import'), self.settings.get('level'))

    @type_checker(in_class=True, kwargs_types=TP_VONMISESFISHER, elemental_types=available_types)
    @document(doc.en_VonMisesFisher_predict)
    def predict(self, **settings: Configuration):
        _settings = copy.deepcopy(self.settings)
        for k, v in settings.items():
            _settings.update({k: v})
        return a(_settings.get('data_import'), self.mean) <= self.threshold


if __name__ == '__main__':
    pass

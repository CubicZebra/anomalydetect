import numpy as np
from numpy import ndarray
from typing import Callable, Tuple
from pynverse import inversefunc
from scipy.stats import chi2
from basic.types import vector, matrix, Configuration, available_types, TP_HOTELLING
from basic.decorators import document, type_checker
import basic.docfunc as doc


def _mvn_params(x: matrix) -> Tuple[vector, matrix]:
    return x.mean(axis=0), np.cov(x.T, bias=True)


def _upper_domain(func: Callable, heuristic_step: float = 10) -> int:
    upper = heuristic_step
    while True:
        try:
            res = inversefunc(func, y_values=1, domain=[0, upper])
        except ValueError:
            upper += heuristic_step  # heuristically increase upper of domain
        else:
            return res


def hotelling_threshold(df: int, level: float = 0.05) -> float:
    """calculate the threshold of chi2 distribution, by a given test level"""
    dis = chi2(df=df)
    upper = _upper_domain(dis.cdf, 10)
    return inversefunc(dis.cdf, y_values=1-level, domain=[0, upper])


def a(x: matrix, miu: vector, sigma: matrix) -> ndarray:
    res = []
    m_inv = np.array(np.matrix(sigma).I)  # inverse matrix
    for i in range(len(x)):
        _ = x[i] - miu
        res.append(np.matmul(np.matmul(_, m_inv), _.T))
    return np.array(res)


@document(doc.en_Hotelling)
class Hotelling:

    settings: Configuration = {
        'model_import': np.array([]),  # ndarray, matrix-like
        'level': 0.05,  # float, 0~1
        'data_import': np.array([]),  # ndarray, matrix-like
    }

    @type_checker(in_class=True, kwargs_types=TP_HOTELLING, elemental_types=available_types)
    def __init__(self, **settings: Configuration):
        for k, v in settings.items():
            self.settings.update({k: v})
        self.mean, self.sigma = _mvn_params(self.settings.get('model_import'))
        self.threshold = hotelling_threshold(self.settings.get('model_import').shape[1], self.settings.get('level'))

    @type_checker(in_class=True, kwargs_types=TP_HOTELLING, elemental_types=available_types)
    @document(doc.en_Hotelling_predict)
    def predict(self, **settings: Configuration):
        for k, v in settings.items():
            self.settings.update({k: v})
        return a(self.settings.get('data_import'), self.mean, self.sigma) <= self.threshold


if __name__ == '__main__':
    pass

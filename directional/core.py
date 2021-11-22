import numpy as np
from numpy import ndarray
from typing import NewType, Tuple
from scipy.stats import chi2
# from pynverse import inversefunc


items, dims = 20, 5
np.random.seed(1)
dt = np.random.random(items*dims).reshape(dims, -1)
sqrt, bias = np.random.randint(1, 4, dims), np.random.randint(1, 5, dims)
dt = np.array([dt[i]*np.sqrt(sqrt[i])+bias[i] for i in range(len(dt))]).T
# print(sqrt, bias)
# print(dt)

vector = NewType('vector', ndarray)
matrix = NewType('matrix', ndarray)


def unitize(x: vector) -> vector:
    """scalarize a vector"""
    return x/(np.linalg.norm(x, ord=2))


def miu(x: matrix) -> vector:
    """mean direction of samples"""
    assert len(x.shape) == 2, 'arg x must be matrix-like'
    x = x.mean(axis=0)
    return unitize(x)


def a(x: matrix) -> str:
    """anomalies of samples: 0~2, from perfectly coincident to totally opposite direction"""
    x_reg = np.array([unitize(x[i]) for i in range(len(x))])
    mean = miu(x)
    return np.array([1-np.dot(mean, x_reg[i]) for i in range(len(x_reg))])


def _moment_2_origin(x: vector) -> float:
    return np.multiply(x, x).mean()


def _chi2_params(x: vector) -> Tuple[float]:
    m1, m2 = x.mean(), _moment_2_origin(x)
    df = 2*(m1**2) / (m2-m1**2)
    scale = (m2-m1**2) / (2*m1)
    return df, scale


def chi2_threshold(x: matrix, level: float = 0.05) -> float:
    df, scale = _chi2_params(a(x))
    chi2_dis = chi2(df=df, scale=scale)
    _x = [0.0001*i for i in range(400)]
    _y = chi2_dis.cdf(_x)
    import matplotlib.pyplot as plt
    plt.plot(_x, _y)
    plt.show()


# calculate inverse function for threshold


if __name__ == '__main__':
    pass

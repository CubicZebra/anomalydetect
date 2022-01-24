from basic.types import TP_BAYES, Bow, Configuration, Tuple, Union, elemental
from basic.decorators import type_checker, document
import basic.docfunc as doc
from typing import Sequence
import numpy as np
from numpy import ndarray
import copy


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """function to obtain interaction point between two curves"""
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except Exception:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


def _find_dim(x: Sequence[Bow]):
    res = 0
    for item in x:
        for sub_item in item:
            res = max(res, sub_item[0])
    return res+1


def _vectorize(x: Sequence[Bow], dim: int):
    res = np.repeat(0, dim)
    for item in x:
        for sub_item in item:
            res[sub_item[0]] += sub_item[1]
    return res


def idx_splitter(x: Sequence, rate: float) -> Tuple[ndarray]:
    uni, counts = np.unique(x, return_counts=True)
    valid_counts = np.array(counts*rate, dtype=int)
    _mediate = np.array([np.where(x == item)[0] for item in uni], dtype=object)
    valid_idx = np.array([np.random.choice(_mediate[_], size=valid_counts[_], replace=False)
                          for _ in range(len(uni))], dtype=object)
    train_idx = np.array([np.array(list(set(_mediate[_]).difference(set(valid_idx[_])))) for _ in range(len(uni))],
                         dtype=object)
    return train_idx, valid_idx


def _a_updater(x: Sequence[Bow], idx: ndarray, dim: int, prior: Union[float, ndarray]) -> ndarray:
    train = np.array([_vectorize(x[idx[_]], dim) for _ in range(len(idx))])
    if not isinstance(prior, ndarray):
        temp1, temp2 = train.sum(axis=1), prior * len(train[0])
        posterior = np.array([(train[_] + prior) / (temp1[_] + temp2) for _ in range(len(train))])
    else:  # prior a vector exists
        temp1, temp2 = train.sum(axis=1), prior.sum(axis=1)
        posterior = np.array([(train[_]+prior[_]) / (temp1[_] + temp2[_]) for _ in range(len(train))])
    _scores = np.log(posterior[0]) - np.log(posterior[1])
    return posterior, _scores/_scores.sum()


def _sparse_a(x: Bow, a_vec: ndarray) -> float:
    res = 0
    for item in x:
        res += item[1] * a_vec[item[0]]
    return res


def a(x: Sequence[Bow], a_vec: ndarray) -> ndarray:
    return np.array([_sparse_a(x[_], a_vec) for _ in range(len(x))])


def harmonic_mean(x: ndarray) -> float:
    cls1, cls2 = x[0], x[1]
    len1, len2 = len(cls1), len(cls2)
    x1 = np.linspace(cls1.min(), cls1.max(), 2000)
    y1 = [(cls1 > x1[_]).sum() / len1 for _ in range(len(x1))]
    x2 = np.linspace(cls2.min(), cls2.max(), 2000)
    y2 = [(cls2 <= x2[_]).sum() / len2 for _ in range(len(x2))]
    return intersection(x1, y1, x2, y2)[0][0]


@document(doc.en_NaiveBayes)
class NaiveBayes:

    settings: Configuration = {
        'gamma': 1.0,  # float, 0~1
        'validation_rate': .3,  # float, 0~1
        'data_import': np.array([]),  # ndarray, matrix-like
    }

    @type_checker(in_class=True, kwargs_types=TP_BAYES, elemental_types=elemental)
    def __init__(self, **settings: Configuration):
        self.settings.update({k: v for k, v in settings.items()})
        self.unique = None
        self.dim = None

        # bayes related params:
        self.X = None
        self.y = None
        self.prior = None
        self.a = None
        self.threshold = None

    @type_checker(in_class=True, kwargs_types=TP_BAYES, elemental_types=elemental)
    @document(doc.en_NaiveBayes_fit)
    def fit(self, **settings: Configuration):
        assert np.all([k in settings.keys() for k in ['x', 'y']]) == 1, 'missing required args x and y.'
        self.settings.update({k: v for k, v in settings.items()})
        self.X, self.y = self.settings.get('x'), self.settings.get('y')
        assert np.unique(self.y).__len__() == 2, 'label y should be of 2 classes.'
        self.dim, self.unique = _find_dim(self.X), np.unique(self.y)
        train_idx, test_idx = idx_splitter(self.y, self.settings.get('validation_rate'))
        self.prior, self.a = _a_updater(self.X, train_idx, self.dim, self.settings.get('gamma'))
        a_scores = np.array([a(self.X[test_idx[_]], self.a) for _ in range(len(test_idx))], dtype=object)  # threshold
        self.threshold = harmonic_mean(a_scores)

    @type_checker(in_class=True, kwargs_types=TP_BAYES, elemental_types=elemental)
    @document(doc.en_NaiveBayes_predict)
    def predict(self, **settings: Configuration) -> ndarray:
        assert np.all([k in settings.keys() for k in ['data_import']]) == 1, 'missing required arg data_import.'
        _settings = copy.deepcopy(self.settings)
        _settings.update({k: v for k, v in settings.items()})
        X = _settings.get('data_import')
        assert self.a is not None, 'predict method should be called after fitting.'
        a_scores = a(X, self.a)
        return np.array([self.unique[0] if a_scores[_] > 0 else self.unique[1] for _ in range(len(a_scores))])

    @type_checker(in_class=True, kwargs_types=TP_BAYES, elemental_types=elemental)
    @document(doc.en_NaiveBayes_update)
    def update(self, **settings: Configuration):
        assert np.all([k in settings.keys() for k in ['x', 'y']]) == 1, 'missing required args x and y.'
        self.settings.update({k: v for k, v in settings.items()})
        X, y = self.settings.get('x'), self.settings.get('y')
        update_idx, _ = idx_splitter(y, 0)
        self.prior, self.a = _a_updater(X, update_idx, self.dim, self.prior)  # for posterior a vector
        self.X = np.concatenate([self.X, X], axis=0)
        self.y = np.concatenate([self.y, y], axis=0)
        _, update_idx = idx_splitter(self.y, self.settings.get('validation_rate'))  # update threshold
        a_scores = np.array([a(self.X[update_idx[_]], self.a) for _ in range(len(update_idx))], dtype=object)
        self.threshold = harmonic_mean(a_scores)


if __name__ == '__main__':
    pass

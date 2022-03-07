import numpy as np
from basic.types import vector, matrix
from typing import Optional
from basic.tests import dt


# print(dt)
# print(dt.mean(axis=0))


_ptr = np.array([1, 1, 4, 4, 8])
# print(dt - _ptr)


def _is_broadcastable(x: matrix, _x: vector) -> Optional[TypeError]:
    if x.shape[1] != _x.shape[0]:
        raise TypeError(r'arg {} is not broadcastable to target matrix'.format(_x))


def _euclidean_ord(x: matrix, _x: vector) -> vector:
    _is_broadcastable(x, _x)
    return np.linalg.norm(x - _x, axis=1, ord=2).argsort(kind='mergesort')


def to_a_table(x: matrix, tag: vector, k: Optional[int] = None):
    idx_tab = np.array([_euclidean_ord(x, item) for item in x])
    _ = np.array([item for item in 'abcdefghijklmnopqrst'])  # labels for test
    # for v in range(len(idx_tab)):
    #     print(_[idx_tab[v]])
    # k -> np.unique(), 二值化的
    cls, counts = np.unique(tag, return_counts=True)
    proportions = counts/counts.sum()
    np.where(tag == '2')
    # print(np.log(v[0]) - np.log(v[1]))  # 计算加速: 当数据量大时，最好对k的值（最大）作出限制


# to_a_table(dt, ['a', 'a', 'b', 'b', 'b'])


# v = _euclidean_dis(dt, _ptr)
# print(v)
# print(np.argsort(v, kind='mergesort'))


from multiprocessing.pool import ThreadPool
import time


def print_hello(x):
    print(r'hello, {}'.format(x))
    time.sleep(2)
    return 'name_' + x


with ThreadPool(2) as p:
    res = p.map(print_hello, ['aa', 'bb'])

print(res)


if __name__ == '__main__':
    # with ThreadPool() as p:
    #     res = p.map(print_hello, ['aa', 'bb'])
    #
    # print(res)
    pass

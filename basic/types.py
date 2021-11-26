from typing import Callable, Any
from numpy import ndarray

basic_types = [type(None), str, bool, int, float, tuple, list, dict]
user_defined_types = [ndarray, Callable[[Any], Any].__origin__]
available_types = basic_types + user_defined_types


TP_VONMISESFISHER = {
    'model_import': ndarray,
    'level': float,
    'data_import': ndarray,
}


if __name__ == '__main__':
    pass

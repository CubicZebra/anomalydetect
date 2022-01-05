from typing import Callable, Any, NewType, Dict, Union
from numpy import ndarray

vector = NewType('vector', ndarray)
matrix = NewType('matrix', ndarray)
Configuration = Dict[str, Union[ndarray, float]]

basic_types = [type(None), str, bool, int, float, tuple, list, dict]
user_defined_types = [ndarray, Callable[[Any], Any].__origin__]
available_types = basic_types + user_defined_types


TP_VONMISESFISHER = {
    'model_import': ndarray,
    'level': float,
    'data_import': ndarray,
}

TP_HOTELLING = TP_VONMISESFISHER


if __name__ == '__main__':
    pass

from typing import Callable, Tuple, List, Dict, Any, Optional, TypeVar
from functools import wraps
import copy


__all__ = ['Type', 'expand_attribute', 'document', 'type_checker']


Type = TypeVar('Type')


def expand_attribute(**kwargs: Dict) -> Callable:
    if len(kwargs) == 0:
        def inner(func):
            return func
        return inner
    else:
        def inner(func):
            class Inner(func):
                for k, v in kwargs.items():
                    exec(k+' = v')
            return Inner
        return inner


def document(func: Callable) -> Callable:
    def add_doc(_func):
        _func.__doc__ = func.__doc__
        return _func
    return add_doc


def _structuralize_params(args: tuple, kwargs: dict, annotations: dict, in_class: bool) -> Tuple[List]:
    if in_class:
        _args = list(args[1:])
    else:
        _args = list(args)

    _annotations = copy.deepcopy(annotations)
    args_num = len(_annotations)
    if 'return' in _annotations.keys():
        del _annotations['return']
        args_num = args_num - 1

    if len(kwargs) > 0:
        del _annotations[list(_annotations.keys())[-1]]
        args_num = args_num - 1
        params = [_args.pop(0) if len(_args) != (args_num + 1) else tuple(_args) for _ in range(args_num)]
    else:  # logic for empty signature
        params = []

    keys_ref = list(_annotations.keys())
    return keys_ref, params


def _get_elemental_types(x: Type, types: List[Type]) -> Tuple[Type]:

    res = []

    def _flatten_elemental_types(_x):
        if hasattr(_x, '__origin__'):
            if _x.__origin__ in types:
                res.append(_x.__origin__)
            if hasattr(_x, '__args__') and len(_x.__args__) > 0:
                for _ in range(len(_x.__args__)):
                    _flatten_elemental_types(_x.__args__[_])
        elif _x in types:  # for available types
            res.append(_x)

    _flatten_elemental_types(x)

    return tuple(res)


def _generic_type_checker(x: Any, tp: Any, types: List[Type]) -> Optional[AssertionError]:

    if tp in types:
        assert isinstance(x, (tp, )) is True

    elif repr(type(tp)) == "<class 'collections.abc.Callable'>":  # for Callable with args: Callable[[...], Any]
        assert isinstance(x, Callable) is True

    elif repr(type(tp)) == "<class 'typing._UnionGenericAlias'>":  # for union generic: Union[...], Optional[...]
        check_types = ()
        for _ in range(len(tp.__args__)):
            check_types += _get_elemental_types(tp.__args__[_], types)
        assert isinstance(x, check_types) is True

    elif hasattr(tp, '__origin__') and tp.__origin__ is tuple:  # for Tuple generic: Tuple[...]
        assert isinstance(x, tuple) is True
        check_types = ()
        if hasattr(tp, '__args__'):
            for _ in range(len(tp.__args__)):
                check_types += _get_elemental_types(tp.__args__[_], types)
        for _ in range(len(x)):
            assert isinstance(x[_], check_types) is True

    elif hasattr(tp, '__origin__') and tp.__origin__ is list:  # for list generic: List[...]
        assert isinstance(x, list) is True
        check_types = _get_elemental_types(tp.__args__[0], types)
        for _ in range(len(x)):
            assert isinstance(x[_], check_types) is True

    elif hasattr(tp, '__origin__') and tp.__origin__ is dict:  # for Dict generic: Dict[str, ...]
        assert isinstance(x, dict) is True
        check_types = _get_elemental_types(tp.__args__[1], types)
        for k, v in x.items():
            assert isinstance(v, check_types) is True


def type_checker(in_class: bool, kwargs_types: Dict, elemental_types: Type):

    def decorator_func(func):

        @wraps(func)
        def inner_func(*args, **kwargs):
            keys_ref, params = _structuralize_params(args, kwargs, func.__annotations__, in_class)

            for m in range(len(params)):
                try:
                    _generic_type_checker(params[m], func.__annotations__[keys_ref[m]], elemental_types)
                except AssertionError:
                    raise TypeError('argument {} must be {}, however the current signature is {}'.
                                    format(keys_ref[m], func.__annotations__[keys_ref[m]], params[m]))

            for k, v in kwargs.items():
                try:
                    _generic_type_checker(v, kwargs_types.get(k), elemental_types)
                except AssertionError:
                    raise TypeError('argument {} must be {}, however the current signature is {}'.
                                    format(k, kwargs_types.get(k), v))
            return func(*args, **kwargs)

        return inner_func

    return decorator_func


if __name__ == '__main__':
    pass

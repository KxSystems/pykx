"""Functionality for the registration of conversion functions between PyKX and Python"""
from .toq import _converter_from_python_type
from .wrappers import Column

from typing import Any, Callable

__all__ = [
    'py_toq',
    'column_function',
]


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


def py_toq(py_type: Any,
           conversion_function: Callable,
           *,
           overwrite: bool = False
) -> None:
    """
    Register conversion logic for a specified Python type when converting it to
    a PyKX object.

    !!! Note
        The return of registered functions should be a valid `pykx` object type
        returns of Pythonic types can result in unexpected errors

    !!! Warning
        Application of this functionality is at a users discretion, issues
        arising from overwritten default conversion types are unsupported

    Parameters:
        py_type: The `type` signature used for determining when a conversion
            should be triggered for PyKX, in particular this will check the
            `type(x)` on incoming data to determine this.
        conversion_function: The function/callable which will be used to convert
            the supplied object to a PyKX object specified by the user.
        *,
        overwrite: If a definition for this type already exists should it be overwritten
            by default this is set to False to avoid accidental overwriting of
            conversion logic used within the library

    Returns:
        A `None` object on successful invocation

    Examples:

    Register conversion logic for complex Python object types

    ```python
    >>> import pykx as kx
    >>> kx.toq(complex(1, 2))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "pykx/toq.pyx", line 2543, in pykx.toq.ToqModule.__call__
      File "pykx/toq.pyx", line 245, in pykx.toq._default_converter
    TypeError: Cannot convert <class 'complex'> '(1+2j)' to K object
    >>> def complex_toq(data):
    ...    return kx.toq([data.real, data.imag])
    >>> kx.register.py_toq(complex, complex_toq)
    >>> kx.toq(complex(1, 2))
    pykx.FloatVector(pykx.q('1 2f'))
    ```

    Register conversion logic for complex Python objects overwriting previous logic above

    ```python
    >>> def complex_toq_upd(data):
    ...    return kx.q('{`real`imag!(x;y)}', kx.toq(data.real), kx.toq(data.imag)
    >>> kx.register.py_toq(complex, complex_toq_upd, overwrite=True)
    >>> kx.toq(complex(1, 2))
    pykx.Dictionary(pykx.q('
    real| 1
    imag| 2
    '))
    >>>
    ```

    """
    if not overwrite and py_type in _converter_from_python_type:
        raise Exception("Attempting to overwrite already defined type :" + str(py_type))

    def wrap_conversion(data, ktype=None, cast=False, handle_nulls=False, strings_as_char=False):
        return conversion_function(data)

    _converter_from_python_type.update({py_type: wrap_conversion})
    return None


def column_function(name: str,
                    conversion_function: Callable,
                    overwrite: bool = False
) -> None:
    """
    Register a function to be accessible as a callable function off the kx.Column
        objects

    !!! Note
        The return of this function should be a `QueryPhrase` object

    !!! Warning
        Application of this functionality is at a users discretion, issues
        arising from overwritten default conversion types are unsupported

    Parameters:
        name: The name to be given to the method which can be used on a column
        conversion_function: The function/callable which will be applied when calling
            a query

    Returns:
        A `None` object on successful invocation

    Examples:

    Register min-max scaler function for application on column

    ```python
    >>> import pykx as kx
    >>> tab = kx.Table(data = {
    ...     'sym': kx.random.random(100, ['a', 'b', 'c']),
    ...     'true': kx.random.random(100, 100.0),
    ...     'pred': kx.random.random(100, 100.0)
    ... })
    >>> def min_max_scaler(self):
    ...     return self.call('{(x-minData)%max[x]-minData:min x}')
    >>> kx.register.column_function('minmax', min_max_scaler)
    >>> tab.select(kx.Column('true') & kx.Column('true').minmax().rename('scaled_true'))
    ```

    Register mean-absolute error function to be applied between 'true' and 'pred' columns

    ```python
    >>> import pykx as kx
    >>> tab = kx.Table(data = {
    ...     'sym': kx.random.random(100, ['a', 'b', 'c']),
    ...     'true': kx.random.random(100, 100.0),
    ...     'pred': kx.random.random(100, 100.0)
    ... })
    >>> def mean_abs_error(self, other):
    ...     return self.call('{avg abs x-y}', other)
    >>> kx.register.column_function('mean_abs_error', mean_abs_error)
    >>> tab.exec(kx.Column('pred').mean_abs_error(kx.Column('true')))
    >>> tab.select(kx.Column('pred').mean_abs_error(kx.Column('true')), by=kx.Column('sym'))
    ```
    """
    if not overwrite:
        try:
            getattr(Column, name)
            raise Exception(f"Attribute {name} already defined, please use 'overwrite' keyword")
        except AttributeError:
            pass
    setattr(Column, name, conversion_function)

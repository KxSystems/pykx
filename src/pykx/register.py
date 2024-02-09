"""Functionality for the registration of conversion functions between PyKX and Python"""
from .toq import _converter_from_python_type


__all__ = [
    'py_toq',
]


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


def py_toq(py_type,
           conversion_function,
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

    def wrap_conversion(data, ktype=None, cast=False, handle_nulls=False):
        return conversion_function(data)

    _converter_from_python_type.update({py_type: wrap_conversion})
    return None

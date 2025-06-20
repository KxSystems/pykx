"""PyKX exceptions and warnings.

Standard Python exceptions (e.g. `TypeError`, `ValueError`, etc.) are used as convention dictates,
but for PyKX and q specific issues custom exceptions are used.
"""

__all__ = [
    'DBError',
    'FutureCancelled',
    'LicenseException',
    'NoResults',
    'QError',
    'PyArrowUnavailable',
    'PyKXException',
    'PyKXWarning',
    'UninitializedConnection',
]


def __dir__():
    return __all__


class PyKXWarning(Warning):
    """Warning type for PyKX-specific warnings."""
    pass


class PyKXException(Exception):
    """Base exception type for PyKX-specific exceptions."""
    pass


class LicenseException(PyKXException):
    """Exception for when a feature that requires a valid q license is used without one."""
    def __init__(self, feature_msg='use this feature', *args, **kwargs):
        super().__init__(
            'A valid q license must be in a known location (e.g. `$QLIC`) to '
            f'{feature_msg}.',
            *args,
            **kwargs
        )


class FutureCancelled(PyKXException):
    """Exception for when a QFuture is cancelled."""
    def __init__(self, msg='', *args, **kwargs):
        super().__init__(
            'This QFuture instance has been cancelled and cannot be awaited. '
            f'{msg}.',
            *args,
            **kwargs
        )


class NoResults(PyKXException):
    """Exception for when a QFutures result is not yet ready."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            'The result is not ready.',
            *args,
            **kwargs
        )


class UninitializedConnection(PyKXException):
    """Exception for when a QConnection is used before it is initialized."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            'The QConnection has not been initialized.',
            *args,
            **kwargs
        )


class PyArrowUnavailable(PyKXException):
    """Exception for features that depend on PyArrow when PyArrow is not available."""
    def __init__(self, msg=None, *args, **kwargs):
        if msg is None: # nocov
            msg = 'PyArrow could not be loaded. Check `pykx._pyarrow.import_attempt_output` ' \
                  'for the reason.'
        super().__init__(msg, *args, **kwargs)


class QError(PyKXException):
    """
    Exception type for q errors.

    Refer to https://code.kx.com/q/basics/errors/ for clarification about error messages.
    """
    pass


class DBError(PyKXException):
    """Exceptions that relate to errors in database usage unrelated to q execution"""
    pass

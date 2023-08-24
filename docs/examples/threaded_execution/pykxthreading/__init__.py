from .pykxthreading import close as shutdown_q, q

import pykx


pykx.q = q

__all__ = sorted([
    'shutdown_q',
    'pykx',
    'q'
])


def __dir__():
    return __all__

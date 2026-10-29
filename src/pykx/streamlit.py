"""Streamlit integration for KDB-X Python."""
# Importing streamlit mutates global state, notably overwriting
# `plotly.io.templates.default` with `"streamlit"`. To keep `import pykx` free of
# those side effects, `PyKXConnection` is defined in `pykx._streamlit_conn` and
# imported here on first access.
from typing import TYPE_CHECKING

__all__ = ['PyKXConnection']

_lazy_attrs = ('PyKXConnection', '_check_streamlit', '_streamlit_unavailable')

if TYPE_CHECKING:  # pragma: no cover - for static analysis and documentation
    from ._streamlit_conn import PyKXConnection  # noqa: F401


def __getattr__(name):
    if name in _lazy_attrs:
        from . import _streamlit_conn
        return getattr(_streamlit_conn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_lazy_attrs)

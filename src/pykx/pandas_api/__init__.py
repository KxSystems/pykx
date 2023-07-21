"""Pandas API for `pykx.Table`.

This is currently a beta feature to be enabled with the following environment variable
`PYKX_ENABLE_PANDAS_API=true` before to import PyKX.
"""


class MetaAtomic:
    tab = None


def api_return(func):
    def return_val(*args, **kwargs):
        tab = args[0]
        if issubclass(type(tab), MetaAtomic):
            tab = tab.tab
        res = func(*args, **kwargs)
        if not issubclass(type(res), PandasIndexing):
            return res
        if tab.replace_self:
            tab.__dict__.update(res.__dict__)
            tab.replace_self = True
        else:
            tab = res
        return tab
    return return_val


from .pandas_meta import _init as _meta_init, PandasMeta
from .pandas_conversions import _init as _conv_init, PandasConversions
from .pandas_indexing import _init as _index_init, PandasIndexing, PandasReindexing, TableLoc
from .pandas_merge import _init as _merge_init, PandasMerge
from .pandas_set_index import _init as _set_index_init, PandasSetIndex


def _init(_q):
    global q
    q = _q
    _meta_init(q)
    _index_init(q)
    _conv_init(q)
    _merge_init(q)
    _set_index_init(q)


class PandasAPI(PandasMeta, PandasIndexing, PandasReindexing,
                PandasConversions, PandasMerge, PandasSetIndex):
    """PandasAPI mixin class

    This is inherited by `pykx.Table` when `PYKX_ENABLE_PANDAS_API=true` is set.
    This class should not be used directly.
    """
    replace_self = False
    prev_locs = {}

    def __init__(self, *args, **kwargs):
        if type(self) == PandasAPI:
            raise Exception(
                'This class must not be instantiated directly. '
                'It is inherited by pykx.Table if PYKX_ENABLE_PANDAS_API=true'
            ) # nocov

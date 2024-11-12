"""Pandas API for `pykx.Table`."""


from warnings import warn


class MetaAtomic:
    tab = None


def handle_groupby_tab(func, *args, **kwargs):
    _kwargs = kwargs
    if 'axis' in _kwargs.keys():
        del _kwargs['axis']
        warn('The axis Keyword argument does not work on GroupbyTable objects.')
    _args = list(args)
    _tab = _args[0]
    _args = _args[1:]
    key, tab = q('{[x] (key x; value x)}', _tab.tab)
    res = q(
        '{[tab; f; args; kwargs]'
        'f[;pyarglist args; pykwargs kwargs] each flip each tab}',
        tab,
        func,
        _args,
        _kwargs
    )
    ungroup = False
    if 'List' in str(type(res)):
        res = q('{[x] flip each x}', res)
        ungroup = True
    if 'Dictionary' in str(type(res)):
        res = q('flip', res)
    if _tab.as_index:
        res = q('{[x; y] x!y}', key, res)
    else:
        if q('{[t] any null first value flip t}', key):
            res = q('{[x; y] x,\'y}', key, res)
        if not _tab.was_keyed:
            if not any([c in key for c in q.cols(res)]):
                res = q('{[x; y] x,\'y}', key, res)
        res = q('{[x; y] x!y}', q(f'([] idx: til {len(res)})'), res)
    if ungroup:
        res = res.ungroup()
    if _tab.as_vector is not None and 'Table' in str(type(res)):
        return res[_tab.as_vector]
    return res


def api_return(func):
    def return_val(*args, **kwargs):
        tab = args[0]
        if 'GroupbyTable' in str(type(tab)):
            return handle_groupby_tab(func, *args, **kwargs)
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
from .pandas_merge import _init as _merge_init, GTable_init, PandasGroupBy, PandasMerge
from .pandas_set_index import _init as _set_index_init, PandasSetIndex
from .pandas_reset_index import _init as _reset_index_init, PandasResetIndex
from .pandas_apply import _init as _apply_init, PandasApply
from .pandas_map import _init as _map_init, PandasMap
from .pandas_sorting import _init as _sorting_init, PandasSorting
from .pandas_replace import _init as _replace_init, PandasReplace


def _init(_q):
    global q
    q = _q
    _meta_init(q)
    _index_init(q)
    _conv_init(q)
    _merge_init(q)
    _set_index_init(q)
    _apply_init(q)
    _map_init(q)
    _sorting_init(q)
    _reset_index_init(q)
    _replace_init(q)


class PandasAPI(PandasApply, PandasMeta, PandasIndexing, PandasReindexing,
                PandasConversions, PandasMerge, PandasSetIndex, PandasGroupBy,
                PandasSorting, PandasReplace, PandasResetIndex, PandasMap):
    """PandasAPI mixin class"""
    replace_self = False
    prev_locs = {}

    def __init__(self, *args, **kwargs):
        if type(self) == PandasAPI:
            raise Exception(
                'This class must not be instantiated directly. It is inherited by pykx.Table'
            ) # nocov

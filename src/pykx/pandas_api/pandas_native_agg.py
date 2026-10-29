from ..wrappers import SymbolVector

_BOOL = frozenset({1})
_NUMERIC = frozenset({1, 4, 5, 6, 7, 8, 9})
_TEMPORAL = frozenset({12, 13, 14, 15, 16, 17, 18, 19})
_SYMBOL = frozenset({11})
_GUID = frozenset({2})

_ALL = '{$[11h=type x; 1b; not any 0=x]}'
_ANY = '{$[11h=type x; any not null x; any (x<>0) and not null x]}'
_ANY_NO_SKIP = '{$[11h=type x; any not null x; any x<>0]}'

_COUNT = (
    '{$[0h=type x; count where {$[0>type x; not null x; 1b]} each x;'
    ' $[0Wi<count x;{count where x};{`long$sum x}] not null x]}'
)
_MEDIAN = '{[vals] $[any null vals; med vals where not null vals; med vals]}'

_SKEW = (
    '{[vals] vals:$[any null vals; vals where not null vals; vals];'
    ' cnt:count vals; dlt:vals-avg vals;'
    ' mom2:avg dlt*dlt; mom3:avg dlt*dlt*dlt;'
    ' $[(cnt<3) or (mom2=0) or any null dlt; 0n; (mom3%mom2 xexp 1.5)*sqrt[cnt*cnt-1]%cnt-2]}'
)
_KURT = (
    '{[vals] vals:$[any null vals; vals where not null vals; vals];'
    ' cnt:count vals; dlt:vals-avg vals;'
    ' ss2:sum dlt*dlt; ss4:sum dlt*dlt*dlt*dlt;'
    ' $[(cnt<4) or (ss2=0) or any null dlt; 0n;'
    ' ((cnt*(cnt+1)*(cnt-1)*ss4)%(cnt-2)*(cnt-3)*ss2*ss2)-3*((cnt-1)*cnt-1)%(cnt-2)*cnt-3]}'
)
_STD = (
    '{[vals] cnt:sum not null vals;'
    ' $[cnt>DDOF; dev[vals]*sqrt cnt%cnt-DDOF; 0n]}'
)
_SEM = (
    '{[vals] cnt:sum not null vals;'
    ' $[cnt>DDOF; dev[vals]%sqrt cnt-DDOF; 0n]}'
)

_MIN = (
    '{[vals] $[11h=type vals;'
    ' $[count nn:vals where not null vals; first asc nn; first 0#vals];'
    ' 2h=type vals;'
    ' $[count nn:vals where not null vals; min nn; first 0#vals]; min vals]}'
)
_MAX = (
    '{[vals] $[11h=type vals;'
    ' $[count nn:vals where not null vals; last asc nn; first 0#vals];'
    ' 2h=type vals;'
    ' $[count nn:vals where not null vals; max nn; first 0#vals]; max vals]}'
)
_IDXMAX = '{[vals; rows] $[all null vals; 0N; rows first where vals=' + _MAX + ' vals]}'
_IDXMIN = '{[vals; rows] $[all null vals; 0N; rows first where vals=' + _MIN + ' vals]}'


def _propagate_nulls(func, skipna):
    if skipna:
        return func
    return '{[vals] $[any null vals; first 0#vals; ' + func + ' vals]}'


def _init(_q):
    global q
    q = _q


def _resolve(name, kwargs):
    """Return the q function, the allowed column types and if it needs the row numbers.

    The result is `None` if `kwargs` contain anything other than values the native version
    supports, or `name` is not a natively supported aggregation. `numeric_only` and `bool_only`
    are handled by the caller and are not checked here.
    """
    kwargs = {k: v for k, v in kwargs.items() if k not in ('numeric_only', 'bool_only')}
    skipna = kwargs.pop('skipna', True)
    if name in ('sum', 'prod'):
        if kwargs.pop('min_count', 0) != 0 or kwargs:
            return None
        return _propagate_nulls('sum' if name == 'sum' else 'prd', skipna), _NUMERIC, False
    if name in ('min', 'max'):
        if kwargs:
            return None
        return (_propagate_nulls(_MIN if name == 'min' else _MAX, skipna),
                _NUMERIC | _TEMPORAL | _GUID | _SYMBOL, False)
    if name in ('std', 'sem'):
        ddof = kwargs.pop('ddof', 1)
        if kwargs or not isinstance(ddof, int) or ddof < 0:
            return None
        if name == 'std' and ddof in (0, 1):
            return ('sdev' if ddof == 1 else 'dev'), _NUMERIC, False
        return (_STD if name == 'std' else _SEM).replace('DDOF', str(ddof)), _NUMERIC, False
    if name in ('idxmax', 'idxmin'):
        if kwargs:
            return None
        return ((_IDXMAX if name == 'idxmax' else _IDXMIN),
                _NUMERIC | _TEMPORAL | _GUID | _SYMBOL, True)
    simple = {
        'mean': ('avg', _NUMERIC),
        'median': (_MEDIAN, _NUMERIC),
        'skew': (_SKEW, _NUMERIC),
        'kurt': (_KURT, _NUMERIC),
        'count': (_COUNT, None),
        'all': (_ALL, _NUMERIC | _SYMBOL),
        'any': (_ANY if skipna else _ANY_NO_SKIP, _NUMERIC | _SYMBOL),
    }
    if kwargs or name not in simple:
        return None
    if not skipna and name in ('mean', 'median', 'skew', 'kurt', 'count'):
        return None
    return (*simple[name], False)


def _column_types(tab):
    return q('{[tab] cls!abs type each tab cls:cols tab}', tab).py()


def native_agg(name, tab, kwargs):
    """Aggregate each column of an ungrouped table, returning `(values, columns)` or `None`."""
    spec = _resolve(name, kwargs)
    if spec is None:
        return None
    func, allowed, needs_rows = spec
    labels = None
    if 'Keyed' in str(type(tab)):
        if needs_rows:
            if len(q('{cols key x}', tab)) != 1:
                return None
            labels = q('{first value flip key x}', tab)
        tab = tab.values()
    types = _column_types(tab)
    cols = list(types)
    if kwargs.get('bool_only'):
        cols = [c for c in cols if types[c] == 1]
    elif kwargs.get('numeric_only'):
        cols = [c for c in cols if types[c] in _NUMERIC]
    if allowed is not None and any(types[c] not in allowed for c in cols):
        return None
    cols = SymbolVector(cols)
    if needs_rows:
        return q(
            '{[tab; cls; func; lbls]'
            '  pos:{[tab; func; col] func[tab col; til count tab]}[tab; func] each cls;'
            '  $[lbls~(::); pos; {[lbls; idx] $[null idx; 0N; lbls idx]}[lbls] each pos]}',
            tab, cols, q(func), labels if labels is not None else q('(::)')
        ), cols
    return q('{[tab; cls; func] func each tab cls}', tab, cols, q(func)), cols


def native_groupby_agg(name, gtab, args, kwargs):
    """Run the aggregation `name` on a `GroupbyTable` using a single grouped select.

    Returns a `(keys, values)` pair with the same meaning as the key and value halves of the
    generic path, or `None` if the aggregation or the grouped table cannot be lowered.
    """
    if args or gtab._source is None:
        return None
    spec = _resolve(name, kwargs)
    if spec is None or kwargs.get('bool_only'):
        return None
    func, allowed, needs_rows = spec
    src, by = gtab._source, gtab._by
    types = _column_types(src)
    keys = by.py()
    cols = gtab._agg_cols if gtab._agg_cols is not None else [c for c in types if c not in keys]
    if kwargs.get('numeric_only'):
        cols = [c for c in cols if types.get(c) in _NUMERIC]
    if len(cols) == 0 or any(c not in types or c in keys for c in cols):
        return None
    # `i` is the row index in a select, even if the table also has a column called `i`
    if 'i' in cols or 'i' in keys:
        return None
    if allowed is not None and any(types[c] not in allowed for c in cols):
        return None
    res = q(
        '{[tab; grp; cls; func; rows; srt; dropna]'
        '  aggs:cls!{[func; rows; col]'
        '    $[rows; (func; col; `i); (func; col)]}[func; rows] each cls;'
        '  if[not srt; aggs[`pykx__first]:(first; `i)];'
        '  res:?[tab; (); grp!grp; aggs];'
        '  if[dropna; res:delete from res where any null value flip key res];'
        '  $[srt; grp xasc res;'
        '    (count grp)!delete pykx__first from `pykx__first xasc 0!res]}',
        src,
        by,
        SymbolVector(list(cols)),
        q(func),
        needs_rows,
        gtab._sort,
        gtab._dropna,
    )
    return q('key', res), q('value', res)

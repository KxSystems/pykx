from ..wrappers import BooleanVector, IntVector, K, List, LongVector, ShortVector, SymbolAtom, SymbolVector, _idx_to_k # noqa
from ..exceptions import QError
from . import api_return, MetaAtomic

import inspect


def _init(_q):
    global q
    q = _q


def _get(tab, key, default=None, cols_check=True):
    idxs = None
    _init_tab = None
    single_col = isinstance(key, (SymbolAtom, str))

    if 'Keyed' in str(type(tab)):
        keys, idxs = key
        _init_tab = tab
        tab = q('value', tab)
        if 0 in idxs:
            keys = keys[1:]
        key = keys

    if cols_check and q('{not all x in cols y}', key, tab) and default is None:
        colstr = str(q('{((),x) except cols y}', key, tab).py())
        raise QError(f'Attempted to retrieve inaccessible columns: {colstr}')

    if isinstance(key, (list, SymbolVector)):
        if not all(x in tab._keys for x in key):
            return default
        tab = q('{?[x; (); 0b; y!y]}', tab, SymbolVector(key))
        if idxs is not None and 0 in idxs:
            tab = q('{(key x)!(y)}', _init_tab, tab)
        return tab

    if isinstance(key, SymbolAtom):
        key = key.py()

    if key in q('cols', tab).py():
        return (q('{[t;k](0!t)k}', tab, key) if single_col
                else q('{((),y)#0!x}', tab, key))

    return default


def _parse_cols_slice_keyed(tab, cols):
    is_numeric = not any([isinstance(x, str) for x in [cols.start, cols.stop]])
    if is_numeric:
        keys = q('{(keys x),(key flip 0#value x)}', tab)
        step = cols.step if cols.step is not None else 1
        start = cols.start if cols.start is not None else (0 if step > 0 else len(keys) - 1)
        stop = cols.stop if cols.stop is not None else (len(keys) if step > 0 else -1)
        if step > 0 and stop < 0:
            stop = stop + len(keys)
        idxs = []
        idx = start
        while True:
            idxs.append(idx)
            idx += step
            if (start < stop and idx >= stop) or (start > stop and idx <= stop) or start == stop:
                break
        return ([keys[x] for x in idxs], idxs)
    else:
        keys = q('{(keys x),(key flip 0#value x)}', tab)
        start = cols.start if cols.start is not None else keys[0]
        stop = cols.stop if cols.stop is not None else keys[-1]
        idx = 0
        idxs = []
        new_keys = []
        adding = False
        for k in keys:
            if k == start:
                adding = True
            if adding:
                new_keys.append(k)
                idxs.append(idx)
            if k == stop:
                adding = False
            idx += 1
        return (new_keys, idxs)


def _parse_cols_slice(tab, cols):
    if 'Keyed' in str(type(tab)):
        return _parse_cols_slice_keyed(tab, cols)
    is_numeric = not any([isinstance(x, str) for x in [cols.start, cols.stop]])
    if is_numeric:
        step = cols.step if cols.step is not None else 1
        start = cols.start if cols.start is not None else (0 if step > 0 else len(tab._keys) - 1)
        stop = cols.stop if cols.stop is not None else (len(tab._keys) if step > 0 else -1)
        if step > 0 and stop < 0:
            stop = stop + len(tab._keys)
        idxs = []
        idx = start
        while True:
            idxs.append(idx)
            idx += step
            if (start < stop and idx >= stop) or (start > stop and idx <= stop) or start == stop:
                break
        return SymbolVector([tab._keys[i] for i in idxs])
    else:
        keys = q('{key flip 0#x}', tab).py()
        start = cols.start if cols.start is not None else keys[0]
        stop = cols.stop if cols.stop is not None else keys[-1]
        new_keys = []
        adding = False
        for k in keys:
            if k == start:
                adding = True
            if adding:
                new_keys.append(k)
            if k == stop:
                adding = False
        return SymbolVector(new_keys)


def _parse_cols(tab, cols):
    if callable(cols):
        cols = cols(tab)
    types = [list, ShortVector, IntVector, LongVector]
    if isinstance(cols, slice):
        return _parse_cols_slice(tab, cols)
    if not any([isinstance(cols, x) for x in types]):
        cols = [cols]
    if isinstance(cols, list) and isinstance(cols[0], str):
        col_names = []
        if 'Keyed' in str(type(tab)):
            [col_names.append(x) for x in q('{(keys x),(key flip 0#value x)}', tab).py()]
        else:
            [col_names.append(x) for x in q('{key flip 0#x}', tab).py()]
        new_colls = []
        idxs = []
        for i in range(len(col_names)):
            if col_names[i] in cols:
                idxs.append(i)
                new_colls.append(col_names[i])
        if 'Keyed' in str(type(tab)):
            return (new_colls, idxs)
        cols = idxs
    cols = SymbolVector([tab._keys[i] for i in cols])
    return cols


def _parse_indexes(tab, loc):
    if callable(loc):
        loc = loc(tab)
    types = [list, ShortVector, IntVector, LongVector, range]
    if isinstance(loc, slice):
        loc = range(len(tab))[loc]
    if ((isinstance(loc, list) and isinstance(loc[0], bool)) or isinstance(loc, BooleanVector))\
            and len(loc) == len(tab):
        idxs = []
        for idx in range(len(tab)):
            if loc[idx]:
                idxs.append(idx)
        loc = idxs
    if not any([isinstance(loc, x) for x in types]):
        loc = [loc]
    return loc


def _iloc(tab, loc):
    cols = None
    if isinstance(loc, tuple):
        cols = _parse_cols(tab, loc[1])
        loc = loc[0]
        tab = _get(tab, cols, None)
    if isinstance(loc, BooleanVector):
        if len(loc) != len(tab):
            new_loc = q('{x where y}', loc, tab.prev_locs[str(len(loc))])
            tab.prev_locs[str(len(loc))] = tab.prev_locs[str(len(loc))] & loc
            loc = new_loc
            tab.prev_locs[str(len(loc))] = loc
        else:
            tab.prev_locs[str(len(tab))] = loc
        if 'Keyed' in str(type(tab)):
            return q('{(count keys x)!((0!x) each where y)}', tab, loc)
        return q('{x where y}', tab, loc)
    loc = _parse_indexes(tab, loc)
    if "Keyed" in str(type(tab)):
        return q('{((flip (keys x)!(enlist y))!x each y)}', tab, loc)
    return q('{x y}', tab, loc)


def _loc(tab, loc): # noqa
    cols = None
    if isinstance(loc, int):
        loc = _idx_to_k(loc, len(tab))
    if isinstance(loc, tuple):
        if 'Keyed' in str(type(tab)) and len(loc) == len(q.keys(tab)):
            keys = q.keys(tab).py()
            where_clause = ''
            for i, key in enumerate(keys):
                val = loc[i]
                if isinstance(val, str):
                    val = '`' + val
                elif isinstance(val, bytes):
                    val = '"' + str(val) + '"'
                where_clause += f'({key}={val}) and '
            # drop last ' and '
            where_clause = where_clause[:-5]
            return q(f'{{[t] flip value select from t where {where_clause}}}', tab)
        cols = _parse_cols(tab, loc[1])
        loc = loc[0]
        tab = _get(tab, cols, None)
    if isinstance(loc, slice):
        return _iloc(tab, loc)
    if (((isinstance(loc, list) and (isinstance(loc[0], str) or isinstance(loc[0], SymbolAtom)))
        or isinstance(loc, SymbolVector)
        or (isinstance(loc, List) and q('{-11h~type x 0}', loc)))
        or ('Keyed' in str(type(tab)) and type(loc) is str)
    ):
        if 'Keyed' in str(type(tab)):
            keys = q.keys(tab).py()
            if not isinstance(keys, list):
                keys = [keys]
            if type(loc) is str or isinstance(loc, SymbolAtom):
                if loc in keys:
                    raise KeyError(f"['{loc}'] is the index of a keyed column")
                if loc not in q.cols(tab).py():
                    raise KeyError(f"['{loc}'] is not an index")
                return q('{[x; y] key[x]!?[x;();0b;{x!x} enlist y]}', tab, loc)
            if any([x in keys for x in loc]):
                raise KeyError(f"['{loc}'] is not an index")
            return q(
                '{[x; y] (key x)!y}',
                tab,
                _get(q('{0!x}', tab), loc, None)
            )
        return _get(tab, loc, None)
    if isinstance(loc, BooleanVector):
        if len(loc) != len(tab):
            new_loc = q('{x where y}', loc, tab.prev_locs[str(len(loc))])
            tab.prev_locs[str(len(loc))] = tab.prev_locs[str(len(loc))] & loc
            loc = new_loc
            tab.prev_locs[str(len(loc))] = loc
        else:
            tab.prev_locs[str(len(tab))] = loc
        if 'Keyed' in str(type(tab)):
            return q('{(count keys x)!((0!x) each where y)}', tab, loc)
        return q('{x where y}', tab, loc)
    if isinstance(loc, str) or isinstance(loc, SymbolAtom):
        if q('{not x in cols y}', loc, tab):
            raise QError(f'Attempted to retrieve inaccessible column: {loc}')
    return q('{x[enlist each y]}', tab, loc)


def _pop(tab, col_name):
    if isinstance(col_name, str):
        col_name = [col_name]
    res = q('{?[x; (); 0b; y!y]}', tab, SymbolVector(col_name))
    new_table = q('{y _ x}', tab, SymbolVector(col_name))
    tab.__dict__.update(new_table.__dict__)
    return res


def _drop_rows(tab, labels, level=None, errors=True):
    if "Keyed" in str(type(tab)):
        if level is None:
            return q('''{[x;y;e]
                      if[0>type y; y:enlist y];
                      if[e&0<count[ee:y where not y in flip k cols k:key x];
                         '(", " sv {"(",(", " sv string x),")"} each ee)," not found."];
                      c:(flip k cols k) in y;
                      ![x;enlist c;0b;`$()]}''',
                     tab, labels, errors)  # noqa
        else:
            return q('''{[x;y;z;e]
                      if[0>type y; y:enlist y];
                      if[0h=type y; y:raze y];
                      if[e&0<count[ee:string y where not y in k xx:cols[k:key x]z];
                         '(", " sv ee)," not found."];
                      c:(k xx) in y;
                      ![x;enlist c;0b;`$()]}''',
                     tab, labels, level, errors)  # noqa
    else:
        return q('''{[x;y;e]
                  if[0>type y; y:enlist y];
                  if[7h<>type y; y:7h$y];
                  if[e&0<count[ee:string y where not y in til count x];
                     '(", " sv ee)," not found."];
                  ![x;enlist (in;`i;`long$y);0b;`$()]}''',
                 tab, labels, errors)  # noqa


def _drop_columns(tab, labels, errors=True):
    if "Keyed" in str(type(tab)):
        return q('''{[x;y;e]
                  if[0>type y; y:enlist y];
                  if[0h=type y; y:raze y];
                  if[e&0<count[ee:string y where not y in cols value x];
                     '(", " sv ee)," not found."];
                  key[x]!((`symbol$y) _ value x)}''',
                 tab, labels, errors)  # noqa
    else:
        return q('''{[x;y;e]
                  if[0>type y; y:enlist y];
                  if[0h=type y; y:raze y];
                  if[e&0<count[ee:string y where not y in cols x];
                     '(", " sv ee)," not found."];
                  (`symbol$y) _ x}''',
                 tab, labels, errors)  # noqa


def _rename_index(tab, labels):
    if "Keyed" in str(type(tab)):
        return q('''{
                    kc:first cols x;
                    idx:key[x]kc;
                    newinds:([] ky:key y;vl:value y);
                    newinds:update ind:{.[?;(x;y);count x]}[idx] each ky from newinds;
                    newinds:select from newinds where not ind=count idx;
                    if[0~count newinds;:x];
                    idx:$[(0h~type idx) or (type[idx]~type[newinds`vl]);
                        @[idx;newinds`ind;:;newinds`vl];
                        1_ @[(::),idx;1+newinds`ind;:;newinds`vl]];
                    (@[key x;kc;:;idx])!value x}''',
                 tab, labels)  # noqa
    else:
        return ValueError(f"""Only pykx.KeyedTable objects can
                          have indexes renamed. Received: {type(tab)}""")


def _rename_columns(tab, labels):
    for x in list(labels.keys()):
        if type(labels[x]) is not str:
            raise ValueError('pykx.Table column names can only be of type pykx.SymbolAtom')
        if type(x) is not str:
            labels.pop(x)
        if x not in tab.columns:
            labels.pop(x)
    if "Keyed" in str(type(tab)):
        return q('''{
                  c:cols value x;
                  c:@[c;c?key y;y];
                  key[x]!c xcol value x}''',
                 tab, labels)  # noqa
    else:
        return q('{c:cols x; c:@[c;c?key y;y]; c xcol x}', tab, labels)


class PandasIndexing:
    @api_return
    def head(self, n: int = 5):
        """Return the first `n` rows."""
        return q(f'{{{n}#x}}', self)

    @api_return
    def tail(self, n: int = 5):
        """Return the last `n` rows."""
        return q(f'{{neg[{n}]#x}}', self)

    @api_return
    def pop(self, col_name: str):
        """Pop a column / columns from a table by name and return it."""
        return _pop(self, col_name)

    @api_return
    def get(self, key, default=None):
        """Get items from table based on key, if key is not found default is returned."""
        return _get(self, key, default)

    @property
    def at(self):
        """Return the value at row, column"""
        return TableAt(self)

    @property
    def loc(self):
        """Return the value at row, column"""
        return TableLoc(self)

    @property
    def iloc(self):
        """Return the value at index."""
        return TableILoc(self)


class PandasReindexing:

    def drop(self, labels=None, axis=0, index=None, columns=None,  # noqa: C901
             level=None, inplace=False, errors='raise'):

        if labels is None and index is None and columns is None:
            raise ValueError("Need to specify at least one of 'labels', 'index' or 'columns'")
        elif labels is not None and (index is not None or columns is not None):
            raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")

        if (columns is not None or axis==1) and level is not None:
            raise ValueError('q/kdb+ tables only support symbols as column labels (no multi index on the column axis).')  # noqa

        if errors == 'raise':
            errors = True
        elif errors == 'ignore':
            errors = False
        else:
            raise ValueError('Errors should be "raise" (default) or "ignore".')

        if inplace:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is not implemented for inplace=True.") # noqa: E501

        if type(labels) is tuple:
            labels = [labels]
        if type(index) is tuple:
            index = [index]

        t = self
        if labels is not None:
            if axis == 0:
                t = _drop_rows(t, labels, level=level, errors=errors)
            elif axis == 1:
                t = _drop_columns(t, labels, errors=errors)
            else:
                raise ValueError(f'No axis named {axis}')
        else:
            if index is not None:
                t = _drop_rows(t, index, level, errors)
            if columns is not None:
                t = _drop_columns(t, columns, errors)

        return t

    def drop_duplicates(self, subset=None, keep='first', inplace=False, ignore_index=False):

        if subset is not None or keep != 'first' or inplace or ignore_index:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is only implemented for keep='first', inplace=False, ignore_index=False.") # noqa: E501

        t = self
        if "Keyed" in str(type(self)):
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is not implemented for KeyedTable objects.") # noqa: E501
        else:
            t = q('distinct', self)

        return t

    def rename(self, mapper=None, index=None, columns=None, axis=0,
               copy=None, inplace=False, level=None, errors='ignore'):
        if ("Keyed" not in str(type(self)) and columns is None
                and ((axis == 'index' or axis == 0) or (index is not None))):
            raise ValueError("Can only rename index of a KeyedTable")
        if (not isinstance(mapper, dict) and mapper is not None):
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() mapper parameter requires a dictionary mapper object.") # noqa: E501
        if (columns is None and ((axis == 'index' or axis == 0) or (index is not None))):
            if len(self.index.columns)!=1:
                raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() index renaming is only supported for single key column KeyedTables.") # noqa: E501
        if mapper is None and index is None and columns is None:
            raise ValueError("must pass an index to rename")
        elif axis !=0 and (index is not None or columns is not None):
            raise ValueError("Cannot specify both 'axis' and any of 'index' or 'columns'")

        if (columns is not None or axis==1) and level is not None:
            raise ValueError('q/kdb+ tables only support symbols as column mapper (no multi index on the column axis).')  # noqa

        if copy is not None or inplace or level is not None or errors != 'ignore':
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is only implemented for copy=None, inplace=False, level=None, errors='ignore'.") # noqa: E501

        t = self
        if mapper is not None:
            if axis == 1 or axis == 'columns':
                t = _rename_columns(t, mapper)
            elif axis == 0 or axis == 'index':
                t = _rename_index(t, mapper)
            else:
                raise ValueError(f'No axis named {axis}')
        else:
            if (index or columns) is None:
                raise ValueError("No columns or indices specified or set")
            if index is not None:
                t = _rename_index(t, index)
            if columns is not None:
                t = _rename_columns(t, columns)

        return t

    def add_suffix(self, suffix, axis=0):
        t = self
        if axis == 1:
            t = q('''{[s;t]
                  c:$[99h~type t;cols value@;cols] t;
                  (c!`$string[c],\\:string s) xcol t
                  }''', suffix, t)
        elif axis == 0:
            if 'Keyed' in str(type(t)):
                raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is \
                                            not implemented for KeyedTable objects.") # noqa: E501
            else:
                return q('{[s;t] c:cols t; (c!`$string[c],\\:string s) xcol t}', suffix, t)
        else:
            raise ValueError(f'No axis named {axis}')
        return t

    def add_prefix(self, prefix, axis=0):
        t = self
        if axis == 1:
            t = q('''{[s;t]
                  c:$[99h~type t;cols value@;cols] t;
                  (c!`$string[s],/:string[c]) xcol t
                  }''', prefix, t)
        elif axis == 0:
            if 'Keyed' in str(type(t)):
                raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is not implemented for KeyedTable objects.") # noqa: E501
            else:
                return q('{[s;t] c:cols t; (c!`$string[s],/:string[c]) xcol t}', prefix, t)
        else:
            raise ValueError(f'No axis named {axis}')
        return t

    def sample(self, n=None, frac=None, replace=False, weights=None,
               random_state=None, axis=None, ignore_index=False):
        if n is None and frac is None:
            n = 1
        elif frac is not None:
            n = int(frac * len(self))

        if weights is not None or random_state is not None \
           or axis is not None or ignore_index:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() is only implemented \
                                        for weights=None, random_state=None, axis=None, ignore_index=False.") # noqa: E501

        if replace:
            if "Keyed" in str(type(self)):
                return q('{idx:y?count x; (key[x]idx)!(value[x]idx)}', self, n)
            else:
                return q('{y?x}', self, n)
        else:
            if n > len(self):
                raise ValueError("Cannot take a larger sample than population when 'replace=False'")

            if "Keyed" in str(type(self)):
                return q('{idx:neg[y]?count x; (key[x]idx)!(value[x]idx)}', self, n)
            else:
                return q('{neg[y]?x}', self, n)


class TableILoc(MetaAtomic):
    def __init__(self, tab):
        self.tab = tab

    @api_return
    def __getitem__(self, loc):
        return _iloc(self.tab, loc)

    @api_return
    def __setitem__(self, loc, val):
        if not isinstance(loc, tuple) or len(loc) != 2:
            raise ValueError('Expected 2 values for call to Table.iloc[] = x')
        col = loc[1]
        if not (isinstance(col, str) or isinstance(col, SymbolAtom)):
            raise ValueError(
                'Expected a column name as the second value for call to Table.iloc[] = x'
            )
        if not ('Table' in str(type(loc[0])) and len(q('{cols x}', loc[0])) == 1):
            loc = _parse_indexes(self.tab, loc[0])
        else:
            loc = q('{raze value flip x}', loc[0])
        if not isinstance(loc, BooleanVector):
            loc = BooleanVector([x in loc for x in range(len(self.tab))])
        res = q(
            f'{{[pykxtable; pykxloc; pykxval] update {col}: pykxval from pykxtable where pykxloc}}',
            self.tab,
            loc,
            val
        )
        self.tab.__dict__.update(res.__dict__)
        return self.tab


class TableLoc(MetaAtomic):
    def __init__(self, tab):
        self.tab = tab

    @api_return
    def __getitem__(self, loc):
        return _loc(self.tab, loc)

    @api_return
    def __setitem__(self, loc, val):
        val = K(val)
        if isinstance(loc, str):
            res = q(f'{{[x; pykxval] update {loc}: pykxval from x}}', self.tab, val)
            self.tab.__dict__.update(res.__dict__)
            return self.tab
        if isinstance(loc, list):
            if len(val) != len(loc):
                raise RuntimeError('The length of values and columns to update must match.')
            update_str = ''
            for i, l in enumerate(loc):
                update_str += f'{l}: pykxval[{i}], '
            update_str = update_str[:-2] + ' ' # strip the last ,
            res = q(f'{{[x; pykxval] update {update_str} from x}}', self.tab, val)
            self.tab.__dict__.update(res.__dict__)
            return self.tab
        if not isinstance(loc, tuple) or len(loc) != 2:
            raise ValueError('Expected 2 values for call to Table.loc[] = x')
        col = loc[1]
        if not (isinstance(col, str) or isinstance(col, SymbolAtom)):
            raise ValueError(
                'Expected a column name as the second value for call to Table.loc[] = x'
            )
        loc = _parse_indexes(self.tab, loc[0])
        if not isinstance(loc, BooleanVector):
            loc = BooleanVector([x in loc for x in range(len(self.tab))])
        res = q(
            f'{{[pykxtable; pykxloc; pykxval] update {col}: pykxval from pykxtable where pykxloc}}',
            self.tab,
            loc,
            val
        )
        self.tab.__dict__.update(res.__dict__)
        return self.tab


class TableAt(MetaAtomic):
    def __init__(self, tab):
        self.tab = tab

    @api_return
    def __getitem__(self, loc):
        if not isinstance(loc, tuple) or len(loc) != 2:
            raise ValueError('Expected 2 values for call to Table.at[]')
        if q('{y in keys x}', self.tab, loc[1]):
            raise QError('Can\'t get the value of a key in a KeyedTable using at.')
        return q('{x[y][z]}', self.tab, loc[0], loc[1])

    @api_return
    def __setitem__(self, loc, val):
        if not isinstance(loc, tuple) or len(loc) != 2:
            raise ValueError('Expected 2 values for call to Table.at[]')
        if q('{y in keys x}', self.tab, loc[1]):
            raise QError('Can\'t reassign the value of a key in a KeyedTable using at.')
        res = q('{[x; y; z; v] x[y; z]: v; x}', self.tab, loc[0], loc[1], val)
        self.tab.__dict__.update(res.__dict__)
        return self.tab

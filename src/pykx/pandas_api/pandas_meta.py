from . import api_return
from ..exceptions import QError

import inspect


def _init(_q):
    global q
    q = _q


def _type_num_is_numeric(typenum):
    if typenum >= 4 and typenum <= 9:
        return True
    return False


def _type_num_is_numeric_or_bool(typenum):
    if typenum >= 4 and typenum <= 9 or typenum == 1:
        return True
    return False


def _get_numeric_only_subtable(tab):
    t = q('0#0!', tab)
    cols = q.cols(t).py()
    numeric_cols = []
    for c in cols:
        if _type_num_is_numeric(t[c].t):
            numeric_cols.append(c)
    return tab[numeric_cols]


def _get_numeric_only_subtable_with_bools(tab):
    t = q('0#0!', tab)
    cols = q.cols(t).py()
    numeric_cols = []
    for c in cols:
        if _type_num_is_numeric_or_bool(t[c].t):
            numeric_cols.append(c)
    return (tab[numeric_cols], numeric_cols)


def _get_bool_only_subtable(tab):
    t = q('0#0!', tab)
    cols = q.cols(t).py()
    bool_cols = []
    for c in cols:
        if t[c].t == 1 or t[c].t == -1:
            bool_cols.append(c)
    return (tab[bool_cols], bool_cols)


def preparse_computations(tab, axis=0, skipna=True, numeric_only=False, bool_only=False):
    if 'Keyed' in str(type(tab)):
        tab = tab.values()
    cols = tab.columns
    if numeric_only:
        (tab, cols) = _get_numeric_only_subtable_with_bools(tab)
    if bool_only:
        (tab, cols) = _get_bool_only_subtable(tab)
    res = q(
        '''
        {[tab;skipna;axis]
          r:value flip tab;
          if[not axis~0;r:flip r];
          if[skipna;r:{x where not null x} each r];
          r}
        ''',
        tab,
        skipna,
        axis
    )
    return (res, cols if axis == 0 else q.til(len(res)), cols)


# The simple computation functions all return a tuple of the results and the col names the results
# were created from, this decorator is used to remove some code duplication to convert all of those
# back into a dictionary
def convert_result(func):
    @api_return
    def inner(*args, **kwargs):
        res, cols = func(*args, **kwargs)
        return q('{[x; y] y!x}', res, cols)
    return inner


# Define the mapping between returns of kx.q.meta and associated data type
_type_mapping = {'c': b'kx.Char',
                 's': b'kx.Symbol',
                 'g': b'kx.GUID',
                 'c': b'kx.Char',
                 'b': b'kx.Boolean',
                 'x': b'kx.Byte',
                 'h': b'kx.Short',
                 'i': b'kx.Int',
                 'j': b'kx.Long',
                 'e': b'kx.Real',
                 'f': b'kx.Float',
                 'p': b'kx.Timestamp',
                 'd': b'kx.Date',
                 'z': b'kx.Datetime',
                 'n': b'kx.Timespan',
                 'u': b'kx.Minute',
                 'v': b'kx.Second',
                 't': b'kx.Time',
                 'm': b'kx.Month',
                 '': b'kx.List'}


class PandasMeta:
    # Dataframe properties
    @property
    def columns(self):
        return q('{if[99h~type x; x:value x]; cols x}', self)

    @property
    def dtypes(self):
        return q('''
                 {a:0!x;
                  flip `columns`datatypes!(
                    a[`c];
                    {$[x~"kx.List";x;x,$[y in .Q.a;"Atom";"Vector"]]}'[y `$/:lower a`t;a`t])
                 }
                 ''', q.meta(self), _type_mapping)

    @property
    def empty(self):
        return q('{0~count x}', self)

    @property
    def ndim(self):
        return q('2')

    @property
    def shape(self):
        return tuple(q('{if[99h~type x; x:value x]; (count x; count cols x)}', self))

    @property
    def size(self):
        return q('{count[x] * count[cols x]}', self)

    @api_return
    def mean(self, axis: int = 0, numeric_only: bool = False):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q('value', tab)
        if numeric_only:
            tab = _get_numeric_only_subtable(tab)

        return q(
            '''
            {[tab;axis]
              idx:$[axis;til count tab;cols tab];
              r:{[tab;axis;idx]
                  (
                   $[axis;`$string@;]idx;
                   avg $[axis;"f"$value@;]tab idx
                  )
                  }[tab;axis]each idx;
              {x[;0]!x[;1]} r where not (::)~/:r[;1]}
            ''', tab, axis
        )

    @api_return
    def kurt(self, axis: int = 0, numeric_only: bool = False):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q.value(tab)
        if numeric_only:
            tab = _get_numeric_only_subtable(tab)

        axis_keys = q('{[axis;tab] $[0~axis;cols;`$string til count @] tab}', axis, tab)

        return q(
            '''{[tab;axis;axis_keys]
                tab:$[0~axis;(::);flip] value flip tab;
                kurt:{[x]
                      res: x - avg x;
                      n: count x;
                      m2: sum rsq: res xexp 2;
                      m4: sum rsq xexp 2;
                      adj: 3 * xexp[n - 1;2] % (n - 2) * (n - 3);
                      num: n * (n + 1) * (n - 1) * m4;
                      den: (n - 2) * (n - 3) * m2 xexp 2;
                      (num % den) - adj};
                axis_keys!kurt each tab}
            ''', tab, axis, axis_keys
        )

    @api_return
    def std(self, axis: int = 0, ddof: int = 1, numeric_only: bool = False):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q.value(tab)
        if numeric_only:
            tab = _get_numeric_only_subtable(tab)

        axis_keys = q('{[axis;tab] $[0~axis;cols;`$string til count @] tab}', axis, tab)

        if ddof == len(tab):
            return q('{x!count[x]#0n}', axis_keys)

        return q(
            '''{[tab;axis;ddof;axis_keys]
                tab:$[0~axis;(::);flip] value flip 9h$tab;
                d:$[0~ddof;dev;
                    1~ddof;sdev;
                    {sqrt (n*var y*c>0)%c:0|(neg x)+n:sum not null y}ddof];
                axis_keys!d each tab
            }''', tab, axis, ddof, axis_keys
        )

    @api_return
    def median(self, axis: int = 0, numeric_only: bool = False):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q('value', tab)
        if numeric_only:
            tab = _get_numeric_only_subtable(tab)

        return q('''
                 {[tab;axis]
                  idx:$[axis;til count tab;cols tab];
                  r:{[tab;axis;idx]
                    (
                     $[axis;`$string@;]idx;
                     med $[axis;"f"$value@;]tab idx
                    )
                    }[tab;axis]each idx;
                  raze{(enlist x 0)!enlist x 1}each r where not (::)~/:r[;1]}
                 ''', tab, axis)

    @convert_result
    def skew(self, axis=0, skipna=True, numeric_only=False):
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q('''
                  {[row]
                    m:{(sum (x - avg x) xexp y) % count x};
                    g1:{[m;x]m:m[x]; m[3] % m[2] xexp 3%2}[m];
                    (g1 each row) * {sqrt[n * n-1] % neg[2] + n:count x} each row
                    }''', res), cols)

    @api_return
    def mode(self, axis: int = 0, numeric_only: bool = False, dropna: bool = True):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q('value', tab)
        if numeric_only:
            tab = _get_numeric_only_subtable(tab)

        return q('''
            {[tab; axis; numeric; drop]
              idx:$[axis;til count tab;cols tab];
              modeQuery:$[numeric;
                {x[l] where d=max d:1_deltas (l:where differ x),count x:asc x};
                {x where f=max f:@[0*i;i:x?x;+;1]}
                ];
              r:{[tab; axis; modeQuery; drop; x]
                  (x; modeQuery $[drop;{x where not null x};] $[axis;value;]tab x)
                  }[tab;axis;modeQuery;drop]each idx;
              maxc: max{count x y}[$[axis;{raze x _ 0};{x 1}]]each r;
              r:{[x; y]
                  $[not y=t:count x 1;
                    [qq: x 1; (x 0;(y - t){[z; t]z,z[t]}[;t]/qq)];
                    (x 0; x 1)]}[;maxc] each r;
              cs:$[axis;`idx,`$string each til count r[0][1];cols tab];
              m:$[axis;{x: raze x; x iasc null x};{1 _ raze x}] each r;
              cs!/:$[axis;;flip]m
            }''', tab, axis, numeric_only, dropna
        )

    @api_return
    def sem(self, axis: int = 0, ddof: int = 1, numeric_only: bool = False):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q.value(tab)
        if numeric_only:
            tab = _get_numeric_only_subtable(tab)

        axis_keys = q('{[axis;tab] $[0~axis;cols;`$string til count @] tab}', axis, tab)

        if ddof == len(tab):
            return q('{x!count[x]#0n}', axis_keys)

        return q(
            '''{[tab;axis;ddof;axis_keys]
                tab:$[0~axis;(::);flip] value flip tab;
                d:{dev[x] % sqrt count[x] - y}[;ddof];
                axis_keys!d each tab}
            ''', tab, axis, ddof, axis_keys
        )

    @api_return
    def abs(self, numeric_only=False):
        tab = self
        if numeric_only:
            tab = _get_numeric_only_subtable(self)
        return q.abs(tab)

    @convert_result
    def all(self, axis=0, bool_only=False, skipna=True):
        res, cols, _ = preparse_computations(self, axis, skipna, bool_only=bool_only)
        return (q('{"b"$x}', [all(x) for x in res]), cols)

    @convert_result
    def any(self, axis=0, bool_only=False, skipna=True):
        res, cols, _ = preparse_computations(self, axis, skipna, bool_only=bool_only)
        return (q('{"b"$x}', [any(x) for x in res]), cols)

    @convert_result
    def max(self, axis=0, skipna=True, numeric_only=False):
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q(
            '{[row] {$[11h=type x; {[x1; y1] $[x1 > y1; x1; y1]} over x; max x]} each row}',
            res
        ), cols)

    @convert_result
    def min(self, axis=0, skipna=True, numeric_only=False):
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q(
            '{[row] {$[11h=type x; {[x1; y1] $[x1 < y1; x1; y1]} over x; min x]} each row}',
            res
        ), cols)

    @convert_result
    def idxmax(self, axis=0, skipna=True, numeric_only=False):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q('value', tab)
        axis = q('{$[11h~type x; `index`columns?x; x]}', axis)
        res, cols, ix = preparse_computations(tab, axis, skipna, numeric_only)
        return (q(
            '''{[row;tab;axis]
                row:{$[11h~type x; {[x1; y1] $[x1 > y1; x1; y1]} over x; max x]} each row;
                m:$[0~axis; (::); flip] value flip tab;
                $[0~axis; (::); cols tab] m {$[abs type y;x]?y}' row}
            ''', res, tab[ix], axis), cols)

    @convert_result
    def idxmin(self, axis=0, skipna=True, numeric_only=False):
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q('value', tab)
        axis = q('{$[11h~type x; `index`columns?x; x]}', axis)
        res, cols, ix = preparse_computations(tab, axis, skipna, numeric_only)
        return (q(
            '''{[row;tab;axis]
                row:{$[11h~type x; {[x1; y1] $[x1 < y1; x1; y1]} over x; min x]} each row;
                m:$[0~axis; (::); flip] value flip tab;
                $[0~axis; (::); cols tab] m {$[abs type y;x]?y}' row}
            ''', res, tab[ix], axis), cols)

    @convert_result
    def prod(self, axis=0, skipna=True, numeric_only=False, min_count=0):
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q('''
                  {[row; minc]
                    {$[y > 0; $[y>count[x]; 0N; prd x]; prd x]}[;minc] each row
                    }
                  ''', res, min_count),
                cols)

    @convert_result
    def sum(self, axis=0, skipna=True, numeric_only=False, min_count=0):
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q('''
                 {[row;minc]
                  {$[y > 0;
                    $[y>count[x]; 0N; $[11h=type x; `$"" sv string x;sum x]];
                    $[11h=type x; `$"" sv string x;sum x]
                    ]}[;minc] each row}
                  ''', res, min_count), cols)

    def agg(self, func, axis=0, *args, **kwargs): # noqa: C901
        if 'KeyedTable' in str(type(self)):
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() 'agg' method is not supported for KeyedTable.") # noqa: E501
        if 'GroupbyTable' not in str(type(self)):
            if 0 == len(self):
                raise QError("Application of 'agg' method not supported for on tabular data with 0 rows") # noqa: E501
        keyname = q('()')
        data = q('()')
        if axis != 0:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() 'axis' parameter is only supported for axis=0.") # noqa: E501
        if isinstance(func, str):
            return getattr(self, func)()
        elif callable(func):
            return self.apply(func, *args, **kwargs)
        elif isinstance(func, list):
            for i in func:
                if isinstance(i, str):
                    keyname = q('{x,y}', keyname, i)
                    data = q('{x, enlist y}', data, getattr(self, i)())
                elif callable(i):
                    keyname = q('{x,y}', keyname, i.__name__)
                    data = q('{x, enlist y}', data, self.apply(i, *args, **kwargs))
            if 'GroupbyTable' in str(type(self)):
                return q('{x!y}', keyname, data)
        elif isinstance(func, dict):
            if 'GroupbyTable' in str(type(self)):
                raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() dictionary input '{func}' is not presently supported for GroupbyTable") # noqa: E501
            data = q('{(flip enlist[`function]!enlist ())!'
                     'flip ($[1~count x;enlist;]x)!'
                     '$[1~count x;enlist;]count[x]#()}', self.keys())
            for key, value in func.items():
                data_name = [key]
                if isinstance(value, str):
                    valname = value
                    keyname = q('{x, y}', keyname, value)
                    exec_data = getattr(self[data_name], value)()
                elif callable(value):
                    valname = value.__name__
                    keyname = q('{x, y}', keyname, valname)
                    exec_data = self[data_name].apply(value, *args, **kwargs)
                else:
                    raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() unsupported type '{type(value)}' was supplied as dictionary value.") # noqa: E501
                data = q('{[x;y;z;k]x upsert(enlist enlist[`function]!enlist[k])!enlist z}',
                         data,
                         self.keys(),
                         exec_data,
                         valname)
            return data
        else:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() func type: {type(func)} is not supported.") # noqa: E501
        if 'GroupbyTable' in str(type(self)):
            return data
        else:
            return (q('{(flip enlist[`function]!enlist x)!y}', keyname, data))

    @convert_result
    def count(self, axis=0, numeric_only=False):
        res, cols, _ = preparse_computations(self, axis, True, numeric_only)
        return (q('count each', res), cols)

    @api_return
    def isna(self):
        return q.null(self)

    @api_return
    def isnull(self):
        return self.isna()

    @api_return
    def notna(self):
        return q('not', self.isna())

    @api_return
    def notnull(self):
        return self.notna()

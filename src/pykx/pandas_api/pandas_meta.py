from . import api_return
from .pandas_native_agg import native_agg
from ..exceptions import QError

import functools
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
    return (tab[numeric_cols] if numeric_cols else None, numeric_cols)


def _get_bool_only_subtable(tab):
    t = q('0#0!', tab)
    cols = q.cols(t).py()
    bool_cols = []
    for c in cols:
        if t[c].t == 1 or t[c].t == -1:
            bool_cols.append(c)
    return (tab[bool_cols] if bool_cols else None, bool_cols)


def preparse_computations(tab, axis=0, skipna=True, numeric_only=False, bool_only=False):
    if 'Keyed' in str(type(tab)):
        tab = tab.values()
    cols = tab.columns
    if numeric_only:
        (tab, cols) = _get_numeric_only_subtable_with_bools(tab)
    if bool_only and tab is not None:
        (tab, cols) = _get_bool_only_subtable(tab)
    if tab is None:
        # `numeric_only` or `bool_only` left no columns, so the result is empty
        cols = q('`symbol$()')
        return (q('()'), cols, cols)
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


def _native(name, tab, axis, **kwargs):
    return native_agg(name, tab, kwargs) if axis == 0 else None


def _native_dict(name, tab, axis, **kwargs):
    native = _native(name, tab, axis, **kwargs)
    return None if native is None else q('{[vals; cls] cls!vals}', *native)


# The simple computation functions all return a tuple of the results and the col names the results
# were created from, this decorator is used to remove some code duplication to convert all of those
# back into a dictionary
def convert_result(func):
    @api_return
    @functools.wraps(func)
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
        """The names of the columns of the table, not including the key columns.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.columns
            pykx.SymbolVector(pykx.q('`a`b'))
        """
        return q('{if[99h~type x; x:value x]; cols x}', self)

    @property
    def dtypes(self):
        """The data type of each column, as a table of column names and `pykx` types.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.dtypes
            pykx.Table(pykx.q('
            columns datatypes
            ---------------------
            a       "kx.LongAtom"
            b       "kx.LongAtom"
            '))
        """
        return q('''
                 {a:0!x;
                  flip `columns`datatypes!(
                    a[`c];
                    {$[x~"kx.List";x;x,$[y in .Q.a;"Atom";"Vector"]]}'[y `$/:lower a`t;a`t])
                 }
                 ''', q.meta(self), _type_mapping)

    @property
    def empty(self):
        """Whether the table has no rows.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.empty
            pykx.BooleanAtom(pykx.q('0b'))
            >>> kx.q('0#([] a:1 2 3)').empty
            pykx.BooleanAtom(pykx.q('1b'))
        """
        return q('{0~count x}', self)

    @property
    def ndim(self):
        """The number of dimensions of the table, which is always 2.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.ndim
            pykx.LongAtom(pykx.q('2'))
        """
        return q('2')

    @property
    def shape(self):
        """The shape of the table as a tuple of the number of rows and the number of columns.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.shape
            (pykx.LongAtom(pykx.q('3')), pykx.LongAtom(pykx.q('2')))
        """
        return tuple(q('{if[99h~type x; x:value x]; (count x; count cols x)}', self))

    @property
    def size(self):
        """The number of values in the table, which is rows multiplied by columns.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.size
            pykx.LongAtom(pykx.q('6'))
        """
        return q('{count[x] * count[cols x]}', self)

    @api_return
    def mean(self, axis: int = 0, numeric_only: bool = False):
        """Returns the mean of values across the given axis.

        Nulls are skipped, as they are by the q function `avg`.

        Parameters:
            axis: The axis to calculate the mean across. 0 is columns, 1 is rows.
            numeric_only: Include only columns or rows with numeric data.

        Returns:
            (Dictionary): A dictionary of the mean across each row or column, keyed by row number or
                column name.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.mean()
            pykx.Dictionary(pykx.q('
            a| 2
            b| 5
            '))
        """
        native = _native_dict('mean', self, axis, numeric_only=numeric_only)
        if native is not None:
            return native
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
        """Returns the unbiased kurtosis across the given axis, using Fisher's definition.

        The kurtosis of a normal distribution is 0.0. Nulls are skipped. A column with fewer than 4
        non null values or with no variance gives null.

        Parameters:
            axis: The axis to calculate the kurtosis across. 0 is columns, 1 is rows.
            numeric_only: Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary of the kurtosis across each row or column, keyed by row
                number or column name.

        Examples:
            >>> tab = kx.q('([] a:1 2 4 8 16f; b:3 1 4 1 5f)')
            >>> tab.kurt()
            pykx.Dictionary(pykx.q('
            a| 1.303763
            b| -2.324219
            '))
        """
        native = _native_dict('kurt', self, axis, numeric_only=numeric_only)
        if native is not None:
            return native
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
        """Returns the sample standard deviation across the given axis.

        This is normalized by N - 1 by default, as in pandas. The q function `dev` is normalized by
        N and `sdev` by N - 1. Nulls are skipped and a column with N <= ddof gives null.

        Parameters:
            axis: The axis to calculate the standard deviation across. 0 is columns, 1 is rows.
            ddof: Delta degrees of freedom. The divisor used is N - ddof, where N is the number of
                non null values.
            numeric_only: Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary of the standard deviation across each row or column, keyed by
                row number or column name.

        Examples:
            >>> tab = kx.q('([] v:1 2 3 4f)')
            >>> tab.std()
            pykx.Dictionary(pykx.q('
            v| 1.290994
            '))
            >>> tab.std(ddof=0)
            pykx.Dictionary(pykx.q('
            v| 1.118034
            '))
        """
        native = _native_dict('std', self, axis, ddof=ddof, numeric_only=numeric_only)
        if native is not None:
            return native
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
        """Returns the median of values across the given axis.

        Nulls are skipped. The q function `med` counts a null as the smallest value, so the two can
        differ when a column contains nulls.

        Parameters:
            axis: The axis to calculate the median across. 0 is columns, 1 is rows.
            numeric_only: Include only columns or rows with numeric data.

        Returns:
            (Dictionary): A dictionary of the median across each row or column, keyed by row number
                or column name.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.median()
            pykx.Dictionary(pykx.q('
            a| 2
            b| 5
            '))
        """
        native = _native_dict('median', self, axis, numeric_only=numeric_only)
        if native is not None:
            return native
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
                     {med $[(type x) within 1 19h; x where not null x; x]}$[axis;"f"$value@;]tab idx
                    )
                    }[tab;axis]each idx;
                  raze{(enlist x 0)!enlist x 1}each r where not (::)~/:r[;1]}
                 ''', tab, axis)

    @convert_result
    def skew(self, axis=0, skipna=True, numeric_only=False):
        """Returns the unbiased skewness across the given axis.

        A column with fewer than 3 non null values or with no variance gives null.

        Parameters:
            axis (int): The axis to calculate the skewness across. 0 is columns, 1 is rows.
            skipna (bool): Ignore any null values along the axis.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `skew` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:1 2 4 8 16f; b:3 1 4 1 5f)')
            >>> tab.skew()
            pykx.Dictionary(pykx.q('
            a| 1.325315
            b| 0.05240784
            '))
        """
        native = _native('skew', self, axis, skipna=skipna, numeric_only=numeric_only)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q('''
                  {[row]
                    m:{(sum (x - avg x) xexp y) % count x};
                    g1:{[m;x]m:m[x]; m[3] % m[2] xexp 3%2}[m];
                    (g1 each row) * {sqrt[n * n-1] % neg[2] + n:count x} each row
                    }''', res), cols)

    @api_return
    def mode(self, axis: int = 0, numeric_only: bool = False, dropna: bool = True):
        """Returns the mode of values across the given axis.

        Parameters:
            axis: The axis to calculate the mode across. 0 is columns, 1 is rows.
            numeric_only: Include only columns or rows with numeric data.
            dropna: Remove null values from the data before calculating the mode.

        Returns:
            (Table): A table of the modes of each row or column. Columns with fewer modes are padded
                with nulls.

        Examples:
            >>> tab = kx.q('([] a:1 1 2; b:3 4 4)')
            >>> tab.mode()
            pykx.Table(pykx.q('
            a b
            ---
            1 4
            '))
        """
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
        """Returns the unbiased standard error of the mean across the given axis.

        This is normalized by N - 1 by default. Nulls are skipped.

        Parameters:
            axis: The axis to calculate the standard error across. 0 is columns, 1 is rows.
            ddof: Delta degrees of freedom. The divisor used is N - ddof, where N is the number of
                non null values.
            numeric_only: Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary of the standard error of the mean across each row or column,
                keyed by row number or column name.

        Examples:
            >>> tab = kx.q('([] a:1 2 4 8 16f; b:3 1 4 1 5f)')
            >>> tab.sem()
            pykx.Dictionary(pykx.q('
            a| 2.727636
            b| 0.8
            '))
        """
        native = _native_dict('sem', self, axis, ddof=ddof, numeric_only=numeric_only)
        if native is not None:
            return native
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
        """Take the absolute value of each element of the table.

        Parameters:
            numeric_only (bool): Only use columns whose data type has an absolute value. Otherwise
                any other column raises an error.

        Returns:
            (Table or KeyedTable): A table or keyed table of the same shape with only positive
                values.

        Examples:
            >>> tab = kx.q('([] a:-1 2 -3)')
            >>> tab.abs()
            pykx.Table(pykx.q('
            a
            -
            1
            2
            3
            '))
        """
        tab = self
        if numeric_only:
            tab = _get_numeric_only_subtable(self)
        return q.abs(tab)

    @convert_result
    def all(self, axis=0, bool_only=False, skipna=True):
        """Returns whether all values across the given axis are `truthy`.

        This follows the behavior of the pandas method of the same name, not the q function `all`.
        Zero is false and any other number is true. Nulls are skipped, so they never make `all`
        false. A null symbol is an empty string, which is false.

        Parameters:
            axis (int): The axis to calculate `all` across. 0 is columns, 1 is rows.
            bool_only (bool): Only use columns of the table that are boolean types.
            skipna (bool): Ignore any null values along the axis. With `skipna=False` a null counts
                as true.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `all` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:0 1 0; c:`x``z; d:101b)')
            >>> tab.all()
            pykx.Dictionary(pykx.q('
            a| 1
            b| 0
            c| 0
            d| 0
            '))
        """
        native = _native('all', self, axis, bool_only=bool_only, skipna=skipna)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, skipna, bool_only=bool_only)
        return (q('{"b"$x}', [all(x) for x in res]), cols)

    @convert_result
    def any(self, axis=0, bool_only=False, skipna=True):
        """Returns whether any value across the given axis is `truthy`.

        This follows the behavior of the pandas method of the same name, not the q function `any`.
        Zero is false and any other number is true. A null is skipped by default, whereas q treats a
        null as nonzero. A null symbol is an empty string, which is false.

        Parameters:
            axis (int): The axis to calculate `any` across. 0 is columns, 1 is rows.
            bool_only (bool): Only use columns of the table that are boolean types.
            skipna (bool): Ignore any null values along the axis. With `skipna=False` a null counts
                as true.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `any` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:0 0n 0f; b:0 1 0)')
            >>> tab.any()
            pykx.Dictionary(pykx.q('
            a| 0
            b| 1
            '))
            >>> tab.any(skipna=False)
            pykx.Dictionary(pykx.q('
            a| 1
            b| 1
            '))
        """
        native = _native('any', self, axis, bool_only=bool_only, skipna=skipna)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, skipna, bool_only=bool_only)
        return (q('{"b"$x}', [any(x) for x in res]), cols)

    @convert_result
    def max(self, axis=0, skipna=True, numeric_only=False):
        """Returns the maximum value across the given axis.

        This follows the behavior of the pandas method of the same name, not the q function `max`.
        The q function always skips nulls.

        Parameters:
            axis (int): The axis to calculate the maximum across. 0 is columns, 1 is rows.
            skipna (bool): Ignore any null values along the axis. With `skipna=False` a column
                containing a null returns a null.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `max` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.max()
            pykx.Dictionary(pykx.q('
            a| 3
            b| 6
            '))
            >>> tab.max(skipna=False)
            pykx.Dictionary(pykx.q('
            a|
            b| 6
            '))
        """
        native = _native('max', self, axis, skipna=skipna, numeric_only=numeric_only)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q(
            '{[row] {$[11h=type x; {[x1; y1] $[x1 > y1; x1; y1]} over x; max x]} each row}',
            res
        ), cols)

    @convert_result
    def min(self, axis=0, skipna=True, numeric_only=False):
        """Returns the minimum value across the given axis.

        This follows the behavior of the pandas method of the same name, not the q function `min`.
        The q function always skips nulls.

        Parameters:
            axis (int): The axis to calculate the minimum across. 0 is columns, 1 is rows.
            skipna (bool): Ignore any null values along the axis. With `skipna=False` a column
                containing a null returns a null.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `min` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.min()
            pykx.Dictionary(pykx.q('
            a| 1
            b| 4
            '))
            >>> tab.min(skipna=False)
            pykx.Dictionary(pykx.q('
            a|
            b| 4
            '))
        """
        native = _native('min', self, axis, skipna=skipna, numeric_only=numeric_only)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q(
            '{[row] {$[11h=type x; {[x1; y1] $[x1 < y1; x1; y1]} over x; min x]} each row}',
            res
        ), cols)

    @convert_result
    def idxmax(self, axis=0, skipna=True, numeric_only=False):
        """Returns the index of the first occurrence of the maximum over the given axis.

        The result is an index label, as in pandas, and not a position. For a table with no key it
        is the row number and for a keyed table it is the value of the key column. Under `groupby`
        the labels are those of the original table. A column of only nulls gives null.

        Parameters:
            axis (int): The axis to calculate `idxmax` across. 0 is columns, 1 is rows.
            skipna (bool): Ignore any null values along the axis.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `idxmax` on that column or row.

        Examples:
            >>> tab = kx.q('1!([] id:10 20 30; v:5 9 2)')
            >>> tab.idxmax()
            pykx.Dictionary(pykx.q('
            v| 20
            '))
        """
        native = _native('idxmax', self, 0 if axis in (0, 'index') else None, skipna=skipna,
                         numeric_only=numeric_only)
        if native is not None:
            return native
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q('value', tab)
        axis = q('{$[11h~type x; `index`columns?x; x]}', axis)
        res, cols, ix = preparse_computations(tab, axis, skipna, numeric_only)
        if len(cols) == 0:
            return (res, cols)
        return (q(
            '''{[row;tab;axis]
                row:{$[11h~type x; {[x1; y1] $[x1 > y1; x1; y1]} over x; max x]} each row;
                m:$[0~axis; (::); flip] value flip tab;
                $[0~axis; (::); cols tab] m {$[abs type y;x]?y}' row}
            ''', res, tab[ix], axis), cols)

    @convert_result
    def idxmin(self, axis=0, skipna=True, numeric_only=False):
        """Returns the index of the first occurrence of the minimum over the given axis.

        The result is an index label, as in pandas, and not a position. For a table with no key it
        is the row number and for a keyed table it is the value of the key column. Under `groupby`
        the labels are those of the original table. A column of only nulls gives null.

        Parameters:
            axis (int): The axis to calculate `idxmin` across. 0 is columns, 1 is rows.
            skipna (bool): Ignore any null values along the axis.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `idxmin` on that column or row.

        Examples:
            >>> tab = kx.q('1!([] id:10 20 30; v:5 9 2)')
            >>> tab.idxmin()
            pykx.Dictionary(pykx.q('
            v| 30
            '))
        """
        native = _native('idxmin', self, 0 if axis in (0, 'index') else None, skipna=skipna,
                         numeric_only=numeric_only)
        if native is not None:
            return native
        tab = self
        if 'Keyed' in str(type(tab)):
            tab = q('value', tab)
        axis = q('{$[11h~type x; `index`columns?x; x]}', axis)
        res, cols, ix = preparse_computations(tab, axis, skipna, numeric_only)
        if len(cols) == 0:
            return (res, cols)
        return (q(
            '''{[row;tab;axis]
                row:{$[11h~type x; {[x1; y1] $[x1 < y1; x1; y1]} over x; min x]} each row;
                m:$[0~axis; (::); flip] value flip tab;
                $[0~axis; (::); cols tab] m {$[abs type y;x]?y}' row}
            ''', res, tab[ix], axis), cols)

    @convert_result
    def prod(self, axis=0, skipna=True, numeric_only=False, min_count=0):
        """Returns the product of all values across the given axis.

        This follows the behavior of the pandas method of the same name, not the q function. Nulls
        are skipped unless `skipna=False`.

        Parameters:
            axis (int): The axis to calculate the product across. 0 is columns, 1 is rows.
            skipna (bool): Ignore any null values along the axis. With `skipna=False` a column
                containing a null returns a null.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.
            min_count (int): If not 0, a null is returned when there are fewer than `min_count` non
                null values.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `prod` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.prod()
            pykx.Dictionary(pykx.q('
            a| 3
            b| 120
            '))
            >>> tab.prod(skipna=False)
            pykx.Dictionary(pykx.q('
            a|
            b| 120
            '))
        """
        native = _native('prod', self, axis, skipna=skipna, numeric_only=numeric_only,
                         min_count=min_count)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q('''
                  {[row; minc; skipna]
                    {$[(not z) and $[0h=type x; 0b; any null x]; first 0#x;
                       y > 0; $[y>count[x]; 0N; prd x]; prd x]}[;minc;skipna] each row
                    }
                  ''', res, min_count, skipna),
                cols)

    @convert_result
    def sum(self, axis=0, skipna=True, numeric_only=False, min_count=0):
        """Returns the sum of all values across the given axis.

        This follows the behavior of the pandas method of the same name, not the q function `sum`.
        The q function always skips nulls, so use `skipna=False` or the query API if a null should
        be kept.

        Parameters:
            axis (int): The axis to calculate the sum across. 0 is columns, 1 is rows.
            skipna (bool): Ignore any null values along the axis. With `skipna=False` a column
                containing a null returns a null.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.
            min_count (int): If not 0, a null is returned when there are fewer than `min_count` non
                null values.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `sum` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f)')
            >>> tab.sum()
            pykx.Dictionary(pykx.q('
            a| 4
            '))
            >>> tab.sum(skipna=False)
            pykx.Dictionary(pykx.q('
            a|
            '))
        """
        native = _native('sum', self, axis, skipna=skipna, numeric_only=numeric_only,
                         min_count=min_count)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, skipna, numeric_only)
        return (q('''
                 {[row;minc;skipna]
                  {$[(not z) and $[0h=type x; 0b; any null x]; first 0#x;
                    y > 0;
                    $[y>count[x]; 0N; $[11h=type x; `$"" sv string x;sum x]];
                    $[11h=type x; `$"" sv string x;sum x]
                    ]}[;minc;skipna] each row}
                  ''', res, min_count, skipna), cols)

    def agg(self, func, axis=0, *args, **kwargs): # noqa: C901
        """Aggregate the table using one or more operations over an axis.

        A dictionary for `func` is not supported on a `groupby` and `agg` is not supported on a
        `KeyedTable`.

        Parameters:
            func (callable, str, list or dict): A function, the name of a function, a list of
                either, or a dictionary of column names to functions or function names.
            axis (int): Only 0, which applies the operation to each column, is supported.
            *args (Any): Positional arguments passed to `func`.
            **kwargs (Any): Keyword arguments passed to `func`.

        Returns:
            (Dictionary, Table or KeyedTable): The result of each operation as a dictionary or
                table.

        Examples:
            >>> tab = kx.q('([] a:1 2 3; b:4 5 6)')
            >>> tab.agg('sum')
            pykx.Dictionary(pykx.q('
            a| 6
            b| 15
            '))
            >>> tab.agg(['min', 'max'])
            pykx.KeyedTable(pykx.q('
            function| a b
            --------| ---
            min     | 1 4
            max     | 3 6
            '))
        """
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
        """Returns the count of non null values across the given axis.

        This counts the values which are not null. The q function `count` counts every value,
        including nulls.

        Parameters:
            axis (int): The axis to count elements across. 0 is columns, 1 is rows.
            numeric_only (bool): Only use columns of the table that are of a numeric data type.

        Returns:
            (Dictionary): A dictionary where the key is the column name or row number and the value
                is the result of `count` on that column or row.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.count()
            pykx.Dictionary(pykx.q('
            a| 2
            b| 3
            '))
        """
        native = _native('count', self, axis, numeric_only=numeric_only)
        if native is not None:
            return native
        res, cols, _ = preparse_computations(self, axis, True, numeric_only)
        return (q('count each', res), cols)

    @api_return
    def isna(self):
        """Detect null values in the table.

        Returns:
            (Table): A table of the same shape where `1b` marks a null value and `0b` a value which
                is not.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.isna()
            pykx.Table(pykx.q('
            a b
            ---
            0 0
            1 0
            0 0
            '))
        """
        return q.null(self)

    @api_return
    def fills(self):
        """Fill each null value with the last value before it which is not null.

        Returns:
            (Table): A table with the nulls filled forward.

        Examples:
            >>> tab = kx.q('([] a:1 0n 0n 4f)')
            >>> tab.fills()
            pykx.Table(pykx.q('
            a
            -
            1
            1
            1
            4
            '))
        """
        return q.fills(self)

    @api_return
    def ungroup(self):
        """Expand a table with list valued columns into one row per list item.

        Returns:
            (Table): The ungrouped table, as in the q function `ungroup`.

        Examples:
            >>> tab = kx.q('([] k:`a`b; v:(1 2;3 4 5))')
            >>> tab.ungroup()
            pykx.Table(pykx.q('
            k v
            ---
            a 1
            a 2
            b 3
            b 4
            b 5
            '))
        """
        return q.ungroup(self)

    @api_return
    def isnull(self):
        """Detect null values in the table. This is an alias of `isna`.

        Returns:
            (Table): A table of the same shape where `1b` marks a null value and `0b` a value which
                is not.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.isnull()
            pykx.Table(pykx.q('
            a b
            ---
            0 0
            1 0
            0 0
            '))
        """
        return self.isna()

    @api_return
    def notna(self):
        """Detect values in the table which are not null.

        Returns:
            (Table): A table of the same shape where `1b` marks a value which is not null and `0b` a
                null.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.notna()
            pykx.Table(pykx.q('
            a b
            ---
            1 1
            0 1
            1 1
            '))
        """
        return q('not', self.isna())

    @api_return
    def notnull(self):
        """Detect values in the table which are not null. This is an alias of `notna`.

        Returns:
            (Table): A table of the same shape where `1b` marks a value which is not null and `0b` a
                null.

        Examples:
            >>> tab = kx.q('([] a:1 0n 3f; b:4 5 6f)')
            >>> tab.notnull()
            pykx.Table(pykx.q('
            a b
            ---
            1 1
            0 1
            1 1
            '))
        """
        return self.notna()

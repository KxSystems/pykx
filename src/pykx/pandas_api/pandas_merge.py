from ..wrappers import K, SymbolVector
from . import api_return

import inspect


def _init(_q):
    global q
    q = _q


def GTable_init(gtab):
    global GTable
    GTable = gtab


def __inner_join(
    left,
    right,
    on,
    distinct=False
):
    if distinct:
        return q.ij(left, q('1!', right))
    if 'KeyedTable' in str(type(left)):
        left = q('0!', left)
    return q.ej(on, q('{z iasc y?x}', left[on], q.distinct(left[on]), left), right)


def __left_join(
    left,
    right,
    on,
    distinct
):
    if distinct:
        return q.ej(on, left, right)
    return q(
        '{[k; t; tt] a:ej[k; t; tt]; a uj t except ((count[k] _ cols tt)_a)}',
        on,
        left,
        right
    )


def __outer_join(
    left,
    right,
    on,
    distinct
):
    if isinstance(on, SymbolVector):
        on = on.py()[0].replace(",", "").replace("`", "")
    if distinct:
        return __inner_join(q('0!', left), q('0!', right), on)
    if 'KeyedTable' not in str(type(left)):
        left = q('1!', left)
    if 'KeyedTable' not in str(type(right)):
        right = q('1!', right)
    return q(
        '{1!distinct (ej[z; x; y]) uj (distinct (uj). 0!/:lj\'[(x;y);(y;x)])}',
        left,
        right,
        on
    )


def _parse_input(
    left,
    right,
    suffixes,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    how='inner'
):
    (left, right, on, added_idx) = __parse_on(
        left,
        right,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how
    )
    (left, right) = __fix_cols(left, right, suffixes, on)
    if how == 'cross':
        if 'Keyed' in str(type(left)):
            left = q('{y _ 0!x}', left, q.keys(left))
        if 'Keyed' in str(type(right)):
            right = q('{y _ 0!x}', right, q.keys(right))
    return (left, right, on, added_idx)


def __fix_cols(left, right, suffixes, on):
    if not suffixes[0] and not suffixes[1]:
        overlap = [
            str(x)
            for x in q.cols(left)
            if str(x) in [str(y) for y in q.cols(right)] and str(x) not in on
        ]
        if len(overlap) != 0:
            raise ValueError(f'Columns overlap but no suffix specified: {overlap}')
    old_l_cols = q.cols(left)
    if suffixes[0]:
        new_l_cols = SymbolVector([
            str(x) + suffixes[0]
            if str(x) not in on and str(x) in q.cols(right)
            else x
            for x in q.cols(left)
        ])
        left = q.xcol(new_l_cols, left)
    if suffixes[1]:
        new_r_cols = SymbolVector([
            str(x) + suffixes[1]
            if str(x) not in on and str(x) in old_l_cols
            else x
            for x in q.cols(right)
        ])
        right = q.xcol(new_r_cols, right)
    return (left, right)


def __parse_on(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    how='inner'
):
    added_idx = False
    if (on is None
        and left_on is None
        and right_on is None
        and not left_index
        and not right_index
    ):
        c2 = q.cols(right)
        on = [x for x in q.cols(left) if x in c2]
    elif left_on is not None and right_on is not None:
        on = left_on
        if on not in q.cols(right):
            # Copy the right_key column and name it left_key
            # and move it to the left most column of the table.
            right = q(
                f'{{`{left_on} xcols update {left_on}: {right_on} from x}}',
                q('0!', right)
            )
    elif left_index and right_index:
        if 'KeyedTable' not in str(type(left)) and 'Table' in str(type(left)):
            added_idx = q(
                '{$[`idx in c:cols[x],cols[y];'
                'last cols .Q.id flip (c,`idx)!enlist each til 1+count c;`idx]}',
                left,
                right
            )
            left = q(f'{{`{added_idx} xcols update {added_idx}: til count x from x}}', left)
            left = q('1!', left)

        if 'KeyedTable' not in str(type(right)) and 'Table' in str(type(right)):
            if not added_idx:
                added_idx = q(
                    '{$[`idx in c:cols[x],cols[y];'
                    'last cols .Q.id flip (c,`idx)!enlist each til 1+count c;`idx]}',
                    left,
                    right
                )
            right = q(f'{{`{added_idx} xcols update {added_idx}: til count x from x}}', right)
        on = q.keys(left)
        left = q('0!', left)
        right = q('0!', right)
    if how == 'cross':
        on = []
    return (left, right, on, added_idx)


def _merge_tables(left, right, on, how, added_idx, left_index, right_index, distinct):
    res = left
    if how == 'inner':
        if left_index and right_index and distinct:
            res = __inner_join(left, right, on, True)
            if added_idx:
                res.pop(added_idx)
            else:
                res = q('1!', res)
        else:
            res = __inner_join(left, right, on)
        return res
    elif how == 'left':
        res = __left_join(left, right, on, distinct)
    elif how == 'right':
        cols = q.cols(left).py()
        cols.extend([x for x in q.cols(right).py() if x not in on])
        res = q.xcols(SymbolVector(cols), __left_join(right, left, on, distinct))
    elif how == 'outer':
        res = __outer_join(left, right, on, distinct)
    elif how == 'cross':
        res = q.cross(left, right)
    if added_idx:
        res.pop(added_idx)
    return res


def _q_merge_tables(left, right, how, added_idx):
    res = left
    if how == 'inner':
        if 'KeyedTable' not in str(type(right)):
            raise ValueError("Inner Join requires a keyed table"
                             " for the right dataset.")
        else:
            res = q.ij(left, right)
    elif how == 'left':
        if 'KeyedTable' not in str(type(right)):
            raise ValueError("Left Join requires a keyed table"
                             " for the right dataset.")
        else:
            res = q.ij(left, right)
    elif how == 'right':

        if 'KeyedTable' not in str(type(left)):
            raise ValueError("Right Join requires a keyed table"
                             " for the left dataset.")
        else:
            res = _q_merge_tables(right, left, 'left', added_idx)
    if added_idx:
        res.pop(added_idx)
    return res


def _validate_merge(left, right, on, validate):
    if validate is not None:
        if validate == '1:1' or validate == 'one_to_one':
            if (not q('{(count x)~count distinct x}', left[on])
                or not q('{(count x)~count distinct x}', right[on])
            ):
                raise ValueError(
                    'Merge keys are not unique in either left or right dataset; '
                    'not a one-to-one merge'
                )
        elif validate == '1:m' or validate == 'one_to_many':
            if not q('{(count x)~count distinct x}', left[on]):
                raise ValueError(
                    'Merge keys are not unique in left dataset; not a one-to-many merge'
                )
        elif validate == 'm:1' or validate == 'many_to_one':
            if not q('{(count x)~count distinct x}', right[on]):
                raise ValueError(
                    'Merge keys are not unique in right dataset; not a many-to-one merge'
                )


def _clean_result(self, res, how, left_index, right_index, added_idx, left, right): # noqa
    if 'Keyed' in str(type(self)) and 'Keyed' not in str(type(res)):
        if how == 'cross':
            res = q('{1!`idx xcols update idx: til count x from x}', res)
        else:
            res = q('1!', res)
    elif 'Keyed' in str(type(res)) and 'Keyed' not in str(type(self)):
        res = q('0!', res)
    if left_index and right_index and not added_idx:
        if how == 'outer':
            key = q.keys(res)[0].py()
            res = q('1!', q('xasc', key, q('0!', res)))
        elif how == 'left':
            if 'Keyed' not in str(type(left)):
                left = q('1!', left)
            res = q('1!', q('{(0!x) iasc ((0!y) keys[y] 0) ((0!x) keys[x] 0)}', res, left))
        elif how == 'right':
            if 'Keyed' not in str(type(right)):
                right = q('1!', right)
            res = q('1!', q('{(0!x) iasc ((0!y) keys[y] 0) ((0!x) keys[x] 0)}', res, right))
    if how == 'outer' and not (left_index or right_index):
        is_keyed = 'KeyedTable' in str(type(res))
        if not is_keyed:
            res = q('1!', res)
        if 'Keyed' not in str(type(left)):
            left = q('1!', left)
        if 'Keyed' not in str(type(right)):
            right = q('1!', right)
        res = q(
            '{(0!x) iasc (distinct[((0!y) keys[y] 0),((0!z) keys[z] 0)])?((0!x) keys[x] 0)}',
            res,
            left,
            right
        )
        if is_keyed:
            res = q('1!', res)
    return res


class PandasMerge:

    @api_return
    def merge(
            self,
            right,
            how='inner',
            on=None,
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            sort=False,
            suffixes=('_x', '_y'),
            copy=True,
            validate=None,
            q_join=False
    ):
        """Join two tables with a database style join.

        A `cross` join cannot be given any columns or indexes to join on.

        Parameters:
            right (Table or KeyedTable): The `Table` or `KeyedTable` to join with.
            how (str): The type of join, one of `'inner'`, `'left'`, `'right'`, `'outer'` or
                `'cross'`.
            on (str): The name of the column to join on.
            left_on (str): The name of the column in the left table to join on.
            right_on (str): The name of the column in the right table to join on.
            left_index (bool): Join on the index of the left table.
            right_index (bool): Join on the index of the right table.
            sort (bool): Sort the result on the join keys.
            suffixes (tuple): The suffixes added to overlapping column names from the left and right
                tables.
            copy (bool): Not yet implemented.
            validate (str): If given, check the join is of this type: `one_to_one` or `1:1`,
                `one_to_many` or `1:m`, `many_to_one` or `m:1`, or `many_to_many` or `m:m`.
            q_join (bool): Use the native q joins rather than the pandas style joins.

        Returns:
            (Table or KeyedTable): The joined table or keyed table.

        Examples:
            >>> left = kx.q('([] k:`a`b`c; v:1 2 3)')
            >>> right = kx.q('([] k:`a`b`d; w:4 5 6)')
            >>> left.merge(right, on='k')
            pykx.Table(pykx.q('
            k v w
            -----
            a 1 4
            b 2 5
            '))
            >>> left.merge(right, on='k', how='outer')
            pykx.Table(pykx.q('
            k v w
            -----
            a 1 4
            b 2 5
            c 3
            d   6
            '))
        """
        if (
            how == 'cross'
            and (
                on is not None
                or left_on is not None
                or right_on is not None
                or left_index
                or right_index
            )
        ):
            raise ValueError(
                'Can not pass on, right_on, '
                'left_on or set right_index=True or left_index=True'
            )
        (left, right, on, added_idx) = _parse_input(
            self,
            right,
            suffixes,
            on,
            left_on,
            right_on,
            left_index,
            right_index,
            how
        )
        _validate_merge(left, right, on, validate)

        distinct = q('{a:0!x; b:0!y; (asc distinct a[z])~(asc distinct b[z])}', left, right, on)
        res = self
        if q_join and (how == 'inner' or how == 'left' or how == 'right'):
            res = _q_merge_tables(
                left,
                right,
                how,
                added_idx
            )
        else:
            res = _merge_tables(left, right, on, how, added_idx, left_index, right_index, distinct)
        res = _clean_result(self, res, how, left_index, right_index, added_idx, left, right)
        if sort and not added_idx:
            if on is None:
                res = q.asc(res)
            else:
                res = q.xasc(on, res)
        if not copy:
            replace_self = self.replace_self
            self.__dict__.update(res.__dict__)
            self.replace_self = replace_self
        return res

    @api_return
    def merge_asof(
            self,
            right,
            on=None,
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            by=None,
            left_by=None,
            right_by=None,
            suffixes=('_x', '_y'),
            tolerance=None,
            allow_exact_matches=True,
            direction='backward'
    ):
        """Join two tables on the nearest earlier value of a key, as the q function `aj` does.

        Parameters:
            right (Table or KeyedTable): The `Table` or `KeyedTable` to join with.
            on (str): The name of the column to join on, which must be sorted.
            left_on (str): The name of the column in the left table to join on.
            right_on (str): The name of the column in the right table to join on.
            left_index (bool): Join on the index of the left table.
            right_index (bool): Join on the index of the right table.
            by (str or list): The columns to match exactly before the nearest value is found.
            left_by (str or list): The names of the columns in the left table to match on.
            right_by (str or list): The names of the columns in the right table to match on.
            suffixes (tuple): The suffixes added to overlapping column names from the left and right
                tables.
            tolerance (None): Not yet implemented.
            allow_exact_matches (bool): Not yet implemented.
            direction (str): Only `'backward'` is implemented.

        Returns:
            (Table or KeyedTable): The joined table or keyed table.

        Examples:
            >>> trades = kx.q('([] time:10:00:01 10:00:03 10:00:05; sym:`a`a`a; px:1 2 3f)')
            >>> quotes = kx.q('([] time:10:00:00 10:00:02 10:00:04; sym:`a`a`a; bid:9 8 7f)')
            >>> trades.merge_asof(quotes, on='time', by='sym')
            pykx.Table(pykx.q('
            time     sym px bid
            -------------------
            10:00:01 a   1  9
            10:00:03 a   2  8
            10:00:05 a   3  7
            '))
        """
        if (
            direction != 'backward'
            or not allow_exact_matches
            or tolerance is not None
            or left_by is not None
            or right_by is not None
        ):
            if direction == 'forward':
                raise ValueError(
                    'nyi: To do an asof join in the opposite direction you can change the order of'
                    'the data within the table. (https://code.kx.com/pykx/api/q/q.html#xdesc) or ('
                    'https://code.kx.com/pykx/api/q/q.html#xasc).'
                )
            else:
                raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() only implemented for direction='backward', \
                                            allow_exact_matches=True, tolerance=None, left_by=None, right_by=None.") # noqa: E501
        if by is not None:
            on = ([by] if isinstance(by, str) else by) + [on]
        (left, right, on, added_idx) = _parse_input(
            self,
            right,
            suffixes,
            on,
            left_on,
            right_on,
            left_index,
            right_index,
            'left'
        )
        res = q.aj(on, left, right)
        if added_idx:
            res = q(f'{len(q.keys(self))}!', res)
        return res


def _parse_group_by_cols(tab, by):
    t = str(type(by)).lower()
    if not ('list' in t or 'vector' in t):
        by = [by]
    keys = q('keys', tab).py()
    cols = []
    for x in by:
        if isinstance(x, int):
            if x > len(keys):
                raise KeyError('Index out of range for groupby column.')
            cols.append(keys[x])
        else:
            if issubclass(type(x), K):
                cols.append(x.py())
            else:
                cols.append(x)
    return SymbolVector(cols)


class PandasGroupBy:

    @api_return
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        observed=False,
        dropna=True
    ):
        """Group the table by the values of one or more columns.

        Parameters:
            by (str, int or list): The name of the column, or a list of names, to group by.
            axis (int): Not yet implemented, must be 0.
            level (str, int or list): The name or number of the key column, or a list of them, to
                group by. A key number can only be used on a `KeyedTable`. `by` and `level` cannot
                both be given.
            as_index (bool): Return the groups as the key of the result.
            sort (bool): Sort the result by the group keys.
            group_keys (bool): Not yet implemented, must be `True`.
            observed (bool): Not yet implemented, must be `False`.
            dropna (bool): Drop the groups where any key is null.

        Returns:
            (GroupbyTable): A `GroupbyTable`, on which aggregations such as `sum` can be called.

        Examples:
            >>> tab = kx.q('([] k:`a`b`a; v:1 2 3)')
            >>> tab.groupby('k').sum()
            pykx.KeyedTable(pykx.q('
            k| v
            -| -
            a| 4
            b| 2
            '))
            >>> tab.groupby('k', as_index=False).sum()
            pykx.KeyedTable(pykx.q('
            idx| k v
            ---| ---
            0  | a 4
            1  | b 2
            '))
        """
        if observed:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() 'observed' parameter is not implemented, please set to False.") # noqa: E501
        if axis != 0:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() non 0 value for the 'axis' parameter is not implemented, please set to 0.") # noqa: E501
        if not group_keys:
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() 'group_keys' parameter is not implemented, please set to True.") # noqa: E501
        if callable(by):
            raise NotImplementedError(f"pykx.{type(self).__name__}.{inspect.stack()[0][3]}() using a callable function for the 'by' parameter is not implemented.") # noqa: E501
        if by is not None and level is not None:
            raise RuntimeError('Cannot use both by and level keyword arguments.')
        pre_keys = q('keys', self)
        grouped = _parse_group_by_cols(self, by if by is not None else level)

        def build():
            res = q('{y xgroup x}', self, grouped)
            post_keys = q('keys', res)
            if len(pre_keys) > 0 and len(pre_keys) != len(post_keys):
                to_remove = SymbolVector([x for x in pre_keys if x not in post_keys])
                res = q(f'{{{len(post_keys)}!(y _ (0!x))}}', res, to_remove)
            if dropna:
                res = q(
                    '{[t] delete from t where any null value flip key t}',
                    res
                )
            if sort:
                res = q('{[t; b] b xasc t}', res, grouped)
            return res

        if len(pre_keys) == 0 and q('{[tab; grp] all grp in cols tab}', self, grouped):
            return GTable(None, as_index, False, build=build, source=self, by=grouped,
                          sort=sort, dropna=dropna)
        return GTable(build(), as_index, len(pre_keys) > 0)

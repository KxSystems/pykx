"""Tests for the Pandas API."""

import sys

import numpy as np
import pandas as pd
import pytest


def check_result_and_type(kx, tab, result):
    if ((isinstance(tab, kx.Table) or isinstance(tab, kx.KeyedTable))
            and tab.py() == result.py() if isinstance(result, kx.K) else result):
        return True
    return False


def test_api_meta_error(kx):
    with pytest.raises(Exception):
        kx.PandasAPI()


def test_df_columns(q):
    df = q('([] til 10; 10?10)')
    assert all(df.columns == df.pd().columns)


def test_df_dtypes(q):
    df = q('([] til 10; 10?0Ng; 10?1f;0f,til 9;10?("abc";"def");10?1e)')
    assert all(df.dtypes.columns == ['columns', 'type'])
    assert q('{x~y}',
             q('("kx.LongAtom";"kx.GUIDAtom";"kx.FloatAtom";"kx.List";"kx.CharVector";"kx.RealAtom")'), # noqa: E501
             df.dtypes['type'])


def test_df_empty(q):
    df = q('([] til 10; 10?10)')
    assert df.empty == df.pd().empty
    df = q('([] `long$(); `long$())')
    assert df.empty == df.pd().empty


def test_df_ndim(q):
    df = q('([] til 10; 10?10)')
    assert(df.ndim == df.pd().ndim)


def test_df_ndim_multicol(q):
    df = q('([] til 10; 10?10; 10?1f)')
    assert(df.ndim == df.pd().ndim)


def test_df_shape(q):
    df = q('([] til 10; 10?10)')
    assert (df.shape == df.pd().shape)


def test_df_size(q):
    df = q('([] til 10; 10?10)')
    assert (df.size == df.pd().size)


def test_df_head(kx, q):
    df = q('([] til 10; 10 - til 10)')
    assert check_result_and_type(kx, df.head(), q('5 # ([] til 10; 10 - til 10)'))
    assert check_result_and_type(kx, df.head(2), q('2 # ([] til 10; 10 - til 10)'))
    df = q('([til 10] 10 - til 10)')
    assert check_result_and_type(kx, df.head(), q('5 # ([til 10] 10 - til 10)'))
    assert check_result_and_type(kx, df.head(2), q('2 # ([til 10] 10 - til 10)'))


def test_df_tail(kx, q):
    df = q('([] til 10; 10 - til 10)')
    assert check_result_and_type(kx, df.tail(), q('5 _ ([] til 10; 10 - til 10)'))
    assert check_result_and_type(kx, df.tail(2), q('8 _ ([] til 10; 10 - til 10)'))
    df = q('([til 10] 10 - til 10)')
    assert check_result_and_type(kx, df.tail(), q('5 _ ([til 10] 10 - til 10)'))
    assert check_result_and_type(kx, df.tail(2), q('8 _ ([til 10] 10 - til 10)'))


def test_df_pop(kx, q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c`d)')
    assert check_result_and_type(kx, df.pop('x'), {'x': [x for x in range(10)]})
    assert check_result_and_type(kx, df.pop('y'), {'y': [10 - x for x in range(10)]})
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c`d)')
    df.pop('z')
    assert check_result_and_type(
        kx,
        df.head(),
        {
            'x': [x for x in range(5)],
            'y': [10 - x for x in range(5)]
        }
    )
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c`d)')
    df.pop(['y', 'z'])
    assert check_result_and_type(kx, df, {'x': [x for x in range(10)]})


def test_df_get(kx, q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    assert check_result_and_type(kx, df.get('x'), {'x': [x for x in range(10)]})
    assert check_result_and_type(kx, df.get(kx.SymbolAtom('y')), {'y': [10 - x for x in range(10)]})
    assert check_result_and_type(kx, df.get(['x', 'y']), {
        'x': [x for x in range(10)],
        'y': [10 - x for x in range(10)]
    })
    assert df.get(['y', 'z']).py() == df[['y', 'z']].py()
    assert df.get(['x', 'y']).py() == df[['x', 'y']].py()
    assert df.get('r') is None
    assert df.get('r', default=5) == 5
    assert df.get(['x', 'r']) is None
    assert df.get(['x', 'r'], default=5) == 5


def test_df_get_keyed(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: 10?`a`b`c)')
    assert check_result_and_type(kx, df.get('x'), {'x': [x for x in range(10)]})
    assert check_result_and_type(kx, df.get(kx.SymbolAtom('y')), {'y': [10 - x for x in range(10)]})
    assert check_result_and_type(kx, df.get(['x', 'y']), {
        'x': [x for x in range(10)],
        'y': [10 - x for x in range(10)]
    })
    assert df.get(['y', 'z']).py() == q.value(df[['y', 'z']]).py()
    assert df.get('r') is None
    assert df.get('r', default=5) == 5
    assert df.get(['x', 'r']) is None
    assert df.get(['x', 'r'], default=5) == 5


def test_df_at(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    for i in range(10):
        assert df.at[i, 'y'].py() == 10 - i
        df.at[i, 'y'] = 2
        assert df.at[i, 'y'].py() == 2
    assert not df.replace_self
    with pytest.raises(ValueError):
        df.at[0]
    with pytest.raises(ValueError):
        df.at[0] = 5


def test_df_at_keyed(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: 10?`a`b`c)')
    for i in range(10):
        assert df.at[i, 'y'].py() == 10 - i
        df.at[i, 'y'] = 2
        assert df.at[i, 'y'].py() == 2
    assert not df.replace_self
    with pytest.raises(ValueError):
        df.at[0]
    with pytest.raises(ValueError):
        df.at[0] = 5
    with pytest.raises(kx.QError):
        df.at[0, 'x']
    with pytest.raises(kx.QError):
        df.at[0, 'x'] = 5


def test_df_replace_self(q):
    df = q('([x: 0, til 10] y: 0, 10 - til 10; z: 11?`a`b`c)')
    df.replace_self = True
    df.tail(10)
    for i in range(10):
        assert df.at[i, 'y'].py() == 10 - i
        df.at[i, 'y'] = 2
        assert df.at[i, 'y'].py() == 2
    assert df.replace_self


def test_df_loc(kx, q):
    df = q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(kx, df.loc[0], {'y': 10, 'z': 'a'})
    assert check_result_and_type(kx, df.loc[[1]], {'y': 9, 'z': 'a'})
    assert check_result_and_type(kx, df.loc[[0, 1]], {'y': [10, 9], 'z': ['a', 'a']})
    assert check_result_and_type(kx, df.loc[0, :], {'y': [10, 9], 'z': ['a', 'a']})


def test_df_loc_keyed(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(kx, df.loc[0], {'y': 10, 'z': 'a'})
    assert check_result_and_type(kx, df.loc[[1]], {'y': 9, 'z': 'a'})
    assert check_result_and_type(kx, df.loc[[0, 1]], {'y': [10, 9], 'z': ['a', 'a']})
    assert check_result_and_type(kx, df.loc[df['y'] < 100], df.py())


def test_df_loc_cols(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(kx, df.loc[[0, 1], 'z':], {'z': ['a', 'a']})
    assert check_result_and_type(kx, df[[0, 1], :'y'], {'y': [10, 9]})
    assert check_result_and_type(kx, df[[0, 1], 'y':'y'], {'y': [10, 9]})
    assert check_result_and_type(kx, df[[0, 1], :2], {'y': [10, 9]})


def test_df_getitem(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(kx, df[0], {'y': 10, 'z': 'a'})
    assert check_result_and_type(kx, df[[1]], {'y': 9, 'z': 'a'})
    assert check_result_and_type(kx, df[[0, 1]], {'y': [10, 9], 'z': ['a', 'a']})
    assert check_result_and_type(kx, df[:], df.py())
    assert check_result_and_type(kx, df[:, ['x', 'y']], q('([x: til 10] y: 10 - til 10)').py())
    df = q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(
        kx,
        df[df['z'] == 'a'],
        {
            'x': [0, 1],
            'y': [10, 9],
            'z': ['a', 'a']
        }
    )


def test_df_loc_set(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    df.loc[df.loc['z'] == 'a', 'y'] = 99
    assert check_result_and_type(
        kx,
        df,
        q('([x: til 10] y: (99 99),8 - til 8; z: `a`a`b`b`c`c`d`d`e`e)').py()
    )
    df = q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    df.loc[df['z'] == 'a', 'y'] = 99
    assert check_result_and_type(
        kx,
        df,
        q('([x: til 10] y: (99 99),8 - til 8; z: `a`a`b`b`c`c`d`d`e`e)').py()
    )
    with pytest.raises(ValueError):
        df.loc[df['z'] == 'a'] = 99
    with pytest.raises(ValueError):
        df.loc[df['z'] == 'a', 3] = 99
    with pytest.raises(ValueError):
        df.loc[df['z'] == 'a', 'y', 'z'] = 99


def test_df_set_cols(kx, q):
    qtab = q('([]til 10;10?1f;10?100)')
    df = qtab
    df['x3'] = 99
    assert check_result_and_type(
        kx,
        df,
        q('{update x3:99 from x}', qtab).py()
    )
    df = qtab
    df['x'] = q('reverse til 10')
    assert check_result_and_type(
        kx,
        df,
        q('{update x:reverse til 10 from x}', qtab).py()
    )
    df = qtab
    df['x'] = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e']
    assert check_result_and_type(
        kx,
        df,
        q('{update x:`a`a`b`b`c`c`d`d`e`e from x}', qtab).py()
    )
    df = qtab
    df[['x', 'x3']] = [q('reverse til 10'), 99]
    assert check_result_and_type(
        kx,
        df,
        q('{update x:reverse til 10, x3:99 from x}', qtab).py()
    )
    df = qtab
    df[['x', 'x3']] = [q('reverse til 10'), ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e']]
    assert check_result_and_type(
        kx,
        df,
        q('{update x:reverse til 10, x3:`a`a`b`b`c`c`d`d`e`e from x}', qtab).py()
    )


def test_df_iloc_set(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    df.iloc[df.loc['z'] == 'a', 'y'] = 99
    assert check_result_and_type(
        kx,
        df,
        q('([x: til 10] y: (99 99),8 - til 8; z: `a`a`b`b`c`c`d`d`e`e)').py()
    )
    df = q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    df.iloc[df['z'] == 'a', 'y'] = 99
    assert check_result_and_type(
        kx,
        df,
        q('([x: til 10] y: (99 99),8 - til 8; z: `a`a`b`b`c`c`d`d`e`e)').py()
    )
    with pytest.raises(ValueError):
        df.iloc[df['z'] == 'a'] = 99
    with pytest.raises(ValueError):
        df.iloc[df['z'] == 'a', 3] = 99
    with pytest.raises(ValueError):
        df.iloc[df['z'] == 'a', 'y', 'z'] = 99


def test_df_iloc(kx, q):
    df = q('([x: til 10] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(kx, df.iloc[:], df.py())
    assert check_result_and_type(kx, df.iloc[:, :-1], q('([x: til 10] y: 10 - til 10)').py())
    assert check_result_and_type(kx, df.iloc[df['y'] < 100], df.py())
    df = q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(kx, df.iloc[:-2], df.head(8).py())
    assert check_result_and_type(kx, df.iloc[0], {'x': 0, 'y': 10, 'z': 'a'})
    assert check_result_and_type(kx, df.iloc[[0]], {'x': 0, 'y': 10, 'z': 'a'})
    assert check_result_and_type(
        kx,
        df.iloc[::-1],
        {
            'x': [10 - x for x in range(10)],
            'y': [x for x in range(10)],
            'z': ['e', 'e', 'd', 'd', 'c', 'c', 'b', 'b', 'a', 'a']
        }
    )
    assert check_result_and_type(
        kx,
        df.head(4).iloc[[True, False, True, False]],
        {
            'x': [0, 2],
            'y': [10, 8],
            'z': ['a', 'b']
        })
    assert check_result_and_type(
        kx,
        df.iloc[lambda x: [x % 2 == 0 for x in range(len(x))]],
        {
            'x': [0, 2, 4, 6, 8],
            'y': [10, 8, 6, 4, 2],
            'z': ['a', 'b', 'c', 'd', 'e']
        }
    )
    assert check_result_and_type(
        kx,
        df.iloc[df['y'] > 5],
        {
            'x': [0, 1, 2, 3, 4],
            'y': [10, 9, 8, 7, 6],
            'z': ['a', 'a', 'b', 'b', 'c']
        }
    )


def test_df_iloc_with_cols(kx, q):
    df = q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    assert check_result_and_type(kx, df.iloc[0, 0], {'x': 0, 'z': 'a'})
    assert check_result_and_type(kx, df.iloc[[0], [2]], {'z': 'a'})
    assert check_result_and_type(
        kx,
        df.iloc[::-1, ::-1],
        {
            'z': ['e', 'e', 'd', 'd', 'c', 'c', 'b', 'b', 'a', 'a'],
            'y': [x for x in range(10)],
            'x': [10 - x for x in range(10)]
        }
    )
    assert check_result_and_type(
        kx,
        df.head(4).iloc[[True, False, True, False], [False, True, False]],
        {
            'y': [10, 8]
        }
    )
    assert check_result_and_type(
        kx,
        df.iloc[lambda x: [x % 2 == 0 for x in range(len(x))], lambda x: [0, 2]],
        {
            'x': [0, 2, 4, 6, 8],
            'z': ['a', 'b', 'c', 'd', 'e']
        }
    )
    assert check_result_and_type(
        kx,
        df.iloc[:, :],
        q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)').py()
    )
    assert check_result_and_type(
        kx,
        df.iloc[:, 'y':],
        q('([] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)').py()
    )
    assert check_result_and_type(
        kx,
        df.iloc[:, :'y'],
        q('([] x: til 10; y: 10 - til 10)').py()
    )
    assert check_result_and_type(
        kx,
        df.iloc[:, 1:],
        q('([] y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)').py()
    )
    assert check_result_and_type(
        kx,
        df.iloc[:, :2],
        q('([] x: til 10; y: 10 - til 10)').py()
    )
    assert check_result_and_type(
        kx,
        df.iloc[:, :-2],
        q('([] x: til 10; y: 10 - til 10)').py()
    )
    assert check_result_and_type(kx, df.loc[df['z']=='a', ['x', 'y']], {'x': [0, 1], 'y': [10, 9]})


def test_table_validate(kx):
    # Copy kwarg
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    with pytest.raises(ValueError):
        tab1.merge(tab2, left_on='lkey', right_on='rkey', validate='1:1')
    with pytest.raises(ValueError):
        tab1.merge(tab2, left_on='lkey', right_on='rkey', validate='m:1')
    with pytest.raises(ValueError):
        tab1.merge(tab2, left_on='lkey', right_on='rkey', validate='1:m')


def test_table_merge_copy(kx, q):
    # Copy kwarg
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    tab1.merge(tab2, left_on='lkey', right_on='rkey', copy=False)
    assert df1.merge(df2, left_on='lkey', right_on='rkey').equals(tab1.pd())

    # Replace_self property
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab1.replace_self = True
    tab2 = kx.toq(df2)
    tab1.merge(tab2, left_on='lkey', right_on='rkey')
    assert df1.merge(df2, left_on='lkey', right_on='rkey').equals(tab1.pd())


def test_table_inner_merge(kx, q):
    # Merge on keys
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    assert df1.merge(
        df2,
        left_on='lkey',
        right_on='rkey'
    ).equals(
        tab1.merge(
            tab2,
            left_on='lkey',
            right_on='rkey'
        ).pd()
    )

    # Merge on keys KeyedTable
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = q('{1!x}', kx.toq(df1))
    tab2 = q('{1!x}', kx.toq(df2))
    assert df1.merge(
        df2,
        left_on='lkey',
        right_on='rkey'
    ).equals(
        q('{0!x}', tab1.merge(
            tab2,
            left_on='lkey',
            right_on='rkey'
        )).pd()
    )

    # Merge on differing keys
    df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
    df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    assert df1.merge(df2, on='a').equals(tab1.merge(tab2, on='a').pd())

    # Merge on same indexes
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    assert df1.merge(
        df2,
        left_index=True,
        right_index=True
    ).equals(
        tab1.merge(
            tab2,
            left_index=True,
            right_index=True
        ).pd()
    )

    # Merge on different indexes
    df1 = pd.DataFrame(
        {
            'lkey': ['foo', 'bar', 'baz', 'foo'],
            'value': [1, 2, 3, 5]
        },
        index=[4, 3, 2, 1]
    )
    df2 = pd.DataFrame(
        {
            'rkey': ['foo', 'bar', 'baz', 'foo'],
            'value': [5, 6, 7, 8]
        },
        index=[0, 1, 2, 3]
    )
    tab1 = q('{`idx xcols update idx: reverse 1 + til count x from x}', tab1)
    tab1 = q('{1!x}', tab1)
    tab2 = q('{`idx xcols update idx: til count x from x}', tab2)
    tab2 = q('{1!x}', tab2)
    res = tab1.merge(tab2, left_index=True, right_index=True)
    assert q(
        '{x~y}',
        tab1.merge(tab2, left_index=True, right_index=True, q_join=True),
        q(
            '{1!x}',
            q.ij(
                q('{0!x}', q.xcol(kx.SymbolVector(['idx', 'lkey', 'value_x']), tab1)),
                q.xcol(kx.SymbolVector(['idx', 'rkey', 'value_y']), tab2)
            )
        )
    )
    assert isinstance(res, kx.KeyedTable)
    df_res = df1.merge(df2, left_index=True, right_index=True)
    # assert our index does match properly before removing it
    assert q('0!', res)['idx'].py() == list(df_res.index)
    # We have idx as a column so we have to remove it to be equal as it won't convert
    # to the pandas index column automatically
    res = q('{(enlist `idx)_(0!x)}', res)
    df_res = df_res.reset_index() # Reset pandas index to default, we already checked it
    df_res.pop('index')
    assert df_res.equals(res.pd())


def test_table_left_merge(kx, q):
    if sys.version_info.minor > 7:
        # Merge on keys
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        assert df1.merge(
            df2,
            left_on='lkey',
            right_on='rkey',
            how='left'
        ).equals(
            tab1.merge(
                tab2,
                left_on='lkey',
                right_on='rkey',
                how='left'
            ).pd()
        )

        # Merge on keys KeyedTable
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = q('{1!x}', kx.toq(df1))
        tab2 = q('{1!x}', kx.toq(df2))
        assert df1.merge(
            df2,
            left_on='lkey',
            right_on='rkey',
            how='left'
        ).equals(
            q('{0!x}', tab1.merge(
                tab2,
                left_on='lkey',
                right_on='rkey',
                how='left'
            )).pd()
        )

        # Merge on differing keys
        df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        tab_res = tab1.merge(tab2, on='a', how='left').pd()
        assert str(tab_res.at[1, 'c']) == '--'
        tab_res.at[1, 'c'] = np.NaN
        assert df1.merge(df2, on='a', how='left').equals(tab_res)

        # Merge on same indexes
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        assert df1.merge(
            df2,
            left_index=True,
            right_index=True,
            how='left'
        ).equals(
            tab1.merge(
                tab2,
                left_index=True,
                right_index=True,
                how='left'
            ).pd()
        )

        # Merge on different indexes
        df1 = pd.DataFrame(
            {
                'lkey': ['foo', 'bar', 'baz', 'foo'],
                'value': [1, 2, 3, 5]
            },
            index=[4, 3, 2, 1]
        )
        df2 = pd.DataFrame(
            {
                'rkey': ['foo', 'bar', 'baz', 'foo'],
                'value': [5, 6, 7, 8]
            },
            index=[0, 1, 2, 3]
        )
        tab1 = q('{`idx xcols update idx: reverse 1 + til count x from x}', tab1)
        tab1 = q('{1!x}', tab1)
        tab2 = q('{`idx xcols update idx: til count x from x}', tab2)
        tab2 = q('{1!x}', tab2)
        res = tab1.merge(tab2, left_index=True, right_index=True, how='left')
        assert q(
            '{x~y}',
            tab1.merge(tab2, left_index=True, right_index=True, how='left', q_join=True),
            q(
                '{1!x}',
                q.lj(
                    q('{0!x}', q.xcol(kx.SymbolVector(['idx', 'lkey', 'value_x']), tab1)),
                    q.xcol(kx.SymbolVector(['idx', 'rkey', 'value_y']), tab2)
                )
            )
        )
        assert isinstance(res, kx.KeyedTable)
        df_res = df1.merge(df2, left_index=True, right_index=True, how='left')
        # assert our index does match properly before removing it
        assert q('0!', res)['idx'].py() == list(df_res.index)
        # We have idx as a column so we have to remove it to be equal as it won't convert
        # to the pandas index column automatically
        res = q('{(enlist `idx)_(0!x)}', res).pd()
        df_res = df_res.reset_index() # Reset pandas index to default, we already checked it
        df_res.pop('index')
        res.at[0, 'rkey'] = np.NaN
        res.at[0, 'value_y'] = np.NaN
        assert df_res.equals(res)

        df1 = pd.DataFrame(
            {'key': ['foo', 'bar', 'baz', 'foo', 'quz'], 'value': [1, 2, 3, 5, None]}
        )
        df2 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo', None], 'value': [5, 6, 7, 8, 99]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)

        df_res = df1.merge(df2, on='key', how='left')
        res = tab1.merge(tab2, on='key', how='left').pd()
        assert str(res.at[6, 'value_y']) == '--'
        res.at[6, 'value_y'] = np.NaN
        assert res.equals(df_res)


def test_table_right_merge(kx, q):
    if sys.version_info.minor > 7:
        # Merge on keys
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        assert df1.merge(
            df2,
            left_on='lkey',
            right_on='rkey',
            how='right'
        ).equals(
            tab1.merge(
                tab2,
                left_on='lkey',
                right_on='rkey',
                how='right'
            ).pd()
        )

        # Merge on keys KeyedTable
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = q('{1!x}', kx.toq(df1))
        tab2 = q('{1!x}', kx.toq(df2))
        assert df1.merge(
            df2,
            left_on='lkey',
            right_on='rkey',
            how='right'
        ).equals(
            q('{0!x}', tab1.merge(
                tab2,
                left_on='lkey',
                right_on='rkey',
                how='right'
            )).pd()
        )

        # Merge on differing keys
        df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        tab_res = tab1.merge(tab2, on='a', how='right').pd()
        assert str(tab_res.at[1, 'b']) == '--'
        tab_res.at[1, 'b'] = np.NaN
        assert df1.merge(df2, on='a', how='right').equals(tab_res)

        # Merge on same indexes
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        assert df1.merge(
            df2,
            left_index=True,
            right_index=True,
            how='right'
        ).equals(
            tab1.merge(
                tab2,
                left_index=True,
                right_index=True,
                how='right'
            ).pd()
        )

        # Merge on different indexes
        df1 = pd.DataFrame(
            {
                'lkey': ['foo', 'bar', 'baz', 'foo'],
                'value': [1, 2, 3, 5]
            },
            index=[4, 3, 2, 1]
        )
        df2 = pd.DataFrame(
            {
                'rkey': ['foo', 'bar', 'baz', 'foo'],
                'value': [5, 6, 7, 8]
            },
            index=[0, 1, 2, 3]
        )
        tab1 = q('{`idx xcols update idx: reverse 1 + til count x from x}', tab1)
        tab1 = q('{1!x}', tab1)
        tab2 = q('{`idx xcols update idx: til count x from x}', tab2)
        tab2 = q('{1!x}', tab2)
        res = tab1.merge(tab2, left_index=True, right_index=True, how='right')
        assert q(
            '{x~y}',
            tab1.merge(tab2, left_index=True, right_index=True, how='right', q_join=True),
            q(
                '{1!x}',
                q.lj(
                    q('{0!x}', q.xcol(kx.SymbolVector(['idx', 'rkey', 'value_y']), tab2)),
                    q.xcol(kx.SymbolVector(['idx', 'lkey', 'value_x']), tab1)
                )
            )
        )
        assert isinstance(res, kx.KeyedTable)
        df_res = df1.merge(df2, left_index=True, right_index=True, how='right')
        # assert our index does match properly before removing it
        assert q('0!', res)['idx'].py() == list(df_res.index)
        # We have idx as a column so we have to remove it to be equal as it won't convert
        # to the pandas index column automatically
        res = q('{(enlist `idx)_(0!x)}', res).pd()
        df_res = df_res.reset_index() # Reset pandas index to default, we already checked it
        df_res.pop('index')
        res.at[0, 'lkey'] = np.NaN
        res.at[0, 'value_x'] = np.NaN
        assert df_res.equals(res)

        df1 = pd.DataFrame(
            {'key': ['foo', 'bar', 'baz', 'foo', 'quz'], 'value': [1, 2, 3, 5, None]}
        )
        df2 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo', None], 'value': [5, 6, 7, 8, 99]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)

        df_res = df1.merge(df2, on='key', how='right')
        res = tab1.merge(tab2, on='key', how='right').pd()
        assert str(res.at[6, 'key']) == ''
        res.at[6, 'key'] = None
        assert res.equals(df_res)


def test_table_outer_merge(kx, q):
    if sys.version_info.minor > 7:
        # Merge on keys
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        assert df1.merge(
            df2,
            left_on='lkey',
            right_on='rkey',
            how='outer'
        ).equals(
            tab1.merge(
                tab2,
                left_on='lkey',
                right_on='rkey',
                how='outer'
            ).pd()
        )
        assert df1.merge(
            df2,
            left_on='lkey',
            right_on='rkey',
            how='outer',
            sort=True
        ).equals(
            tab1.merge(
                tab2,
                left_on='lkey',
                right_on='rkey',
                how='outer',
                sort=True
            ).pd()
        )

        # Merge on keys KeyedTable
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = q('{1!x}', kx.toq(df1))
        tab2 = q('{1!x}', kx.toq(df2))
        assert df1.merge(
            df2,
            left_on='lkey',
            right_on='rkey',
            how='outer',
            sort=True
        ).equals(
            q('{0!x}', tab1.merge(
                tab2,
                left_on='lkey',
                right_on='rkey',
                how='outer',
                sort=True
            )).pd()
        )

        # Merge on differing keys
        df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        tab_res = tab1.merge(tab2, on='a', how='outer').pd()
        assert str(tab_res.at[1, 'c']) == '--'
        tab_res.at[1, 'c'] = np.NaN
        assert str(tab_res.at[2, 'b']) == '--'
        tab_res.at[2, 'b'] = np.NaN
        assert df1.merge(df2, on='a', how='outer').equals(tab_res)

        # Merge on same indexes
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        assert df1.merge(
            df2,
            left_index=True,
            right_index=True,
            how='outer'
        ).equals(
            tab1.merge(
                tab2,
                left_index=True,
                right_index=True,
                how='outer'
            ).pd()
        )
        assert df1.merge(
            df2,
            left_index=True,
            right_index=True,
            how='outer',
            sort=True
        ).equals(
            tab1.merge(
                tab2,
                left_index=True,
                right_index=True,
                how='outer',
                sort=True
            ).pd()
        )

        # Merge on different indexes
        df1 = pd.DataFrame(
            {
                'lkey': ['foo', 'bar', 'baz', 'foo'],
                'value': [1, 2, 3, 5]
            },
            index=[4, 3, 2, 1]
        )
        df2 = pd.DataFrame(
            {
                'rkey': ['foo', 'bar', 'baz', 'foo'],
                'value': [5, 6, 7, 8]
            },
            index=[0, 1, 2, 3]
        )
        tab1 = q('{`idx xcols update idx: reverse 1 + til count x from x}', tab1)
        tab1 = q('{1!x}', tab1)
        tab2 = q('{`idx xcols update idx: til count x from x}', tab2)
        tab2 = q('{1!x}', tab2)
        res = tab1.merge(tab2, left_index=True, right_index=True, how='outer')
        assert isinstance(res, kx.KeyedTable)
        df_res = df1.merge(df2, left_index=True, right_index=True, how='outer')
        # assert our index does match properly before removing it
        assert q('0!', res)['idx'].py() == list(df_res.index)
        # We have idx as a column so we have to remove it to be equal as it won't convert
        # to the pandas index column automatically
        res = q('{(enlist `idx)_(0!x)}', res).pd()
        df_res = df_res.reset_index() # Reset pandas index to default, we already checked it
        df_res.pop('index')
        res.at[0, 'lkey'] = np.NaN
        res.at[0, 'value_x'] = np.NaN
        res.at[4, 'rkey'] = np.NaN
        res.at[4, 'value_y'] = np.NaN
        assert df_res.equals(res)

        df1 = pd.DataFrame(
            {'key': ['foo', 'bar', 'baz', 'foo', 'quz'], 'value': [1, 2, 3, 5, None]}
        )
        df2 = pd.DataFrame(
            {
                'key': ['foo', 'bar', 'baz', 'foo', None],
                'value': [5.0, 6.0, 7.0, 8.0, 99.0]
            }
        )
        tab1 = kx.toq(df1)
        tab2 = kx.toq(df2)
        df_res = df1.merge(df2, on='key', how='outer')
        res = tab1.merge(tab2, on='key', how='outer').pd()
        assert res.at[7, 'key'] == ''
        res.at[7, 'key'] = None
        assert df_res.equals(res)


def test_cross_merge(kx, q):
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    assert df1.merge(df2, how='cross').equals(tab1.merge(tab2, how='cross').pd())
    tab1 = kx.q('{`idx xcols update idx: reverse 1 + til count x from x}', tab1)
    tab1 = kx.q('{1!x}', tab1)
    tab2 = kx.q('{`idx xcols update idx: til count x from x}', tab2)
    tab2 = kx.q('{1!x}', tab2)
    df_res = df1.merge(df2, how='cross')
    res = tab1.merge(tab2, how='cross')
    assert q('0!', res)['idx'].py() == list(df_res.index)
    # We have idx as a column so we have to remove it to be equal as it won't convert
    # to the pandas index column automatically
    res = q('{(enlist `idx)_(0!x)}', res).pd()
    assert df_res.equals(res)


def test_merge_errors(kx):
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    with pytest.raises(ValueError):
        tab1.merge(
            tab2,
            left_on='lkey',
            right_on='rkey',
            how='outer',
            suffixes=(False, False)
        )


def test_cross_merge_errors(kx, q):
    df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]})
    df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]})
    tab1 = kx.toq(df1)
    tab2 = kx.toq(df2)
    with pytest.raises(ValueError) as e1:
        tab1.merge(tab2, how='cross', on='lkey')
    assert (
        'Can not pass on, right_on, left_on or set right_index=True or left_index=True'
        in str(e1.value)
    )
    with pytest.raises(ValueError) as e2:
        tab1.merge(tab2, how='cross', left_on='lkey', right_on='rkey')
    assert (
        'Can not pass on, right_on, left_on or set right_index=True or left_index=True'
        in str(e2.value)
    )
    with pytest.raises(ValueError) as e3:
        tab1.merge(tab2, how='cross', left_index=True, right_index=True)
    assert (
        'Can not pass on, right_on, left_on or set right_index=True or left_index=True'
        in str(e3.value)
    )


def test_api_vs_pandas(kx, q):
    tab = q('([] x: til 10; y: 10 - til 10; z: `a`a`b`b`c`c`d`d`e`e)')
    df = tab.pd()
    assert q(
        '{x ~ y}',
        tab[(tab['z'] == 'b') | (tab['z'] == 'c') | (tab['z'] == 'd')],
        q('{value x}', kx.toq(df[(df['z'] == 'b') | (df['z'] == 'c') | (df['z'] == 'd')]))
    )
    assert q(
        '{x ~ y}',
        tab[(tab['z'] == 'b') | (tab['z'] == 'c') | (tab['z'] == 'd')][tab['x'] > 5],
        q(
            '{value x}',
            kx.toq(df[(df['z'] == 'b') | (df['z'] == 'c') | (df['z'] == 'd')][df['x'] > 5])
        )
    )
    assert q(
        '{x ~ y}',
        tab.iloc[(tab['z'] == 'b') | (tab['z'] == 'c') | (tab['z'] == 'd')].iloc[tab['x'] > 5],
        q(
            '{value x}',
            kx.toq(df[(df['z'] == 'b') | (df['z'] == 'c') | (df['z'] == 'd')][df['x'] > 5])
        )
    )


def test_df_astype_vanilla_checks(kx, q):
    df = q('([] c1:1 2 3i; c2:1 2 3j; c3:1 2 3h; c4:1 2 3i)')
    assert check_result_and_type(
        kx,
        df.astype(kx.LongVector).py(),
        q('([] c1:1 2 3j; c2:1 2 3j; c3:1 2 3j; c4:1 2 3j)').py()
    )
    assert check_result_and_type(
        kx,
        df.astype({'c1': kx.LongVector, 'c2': 'kx.ShortVector'}).py(),
        q('([] c1:1 2 3j; c2:1 2 3h; c3:1 2 3h; c4:1 2 3i)').py()
    )


def test_df_astype_string_to_sym(kx, q):
    df = q('''([] c1:3#.z.p; c2:`abc`def`ghi; c3:1 2 3j;
            c4:("abc";"def";"ghi");c5:"abc";c6:(1 2 3;4 5 6;7 8 9))''')
    assert check_result_and_type(
        kx,
        df.astype({'c4': kx.SymbolVector, 'c5': kx.SymbolVector}).py(),
        q('''([] c1:3#.z.p; c2:`abc`def`ghi; c3:1 2 3j;
            c4:`abc`def`ghi;c5:`a`b`c;c6:(1 2 3;4 5 6;7 8 9))''').py()
    )
    assert check_result_and_type(
        kx,
        df.astype({'c4': kx.SymbolVector}).py(),
        q('''([] c1:3#.z.p; c2:`abc`def`ghi; c3:1 2 3j;
            c4:`abc`def`ghi;c5:"abc";c6:(1 2 3;4 5 6;7 8 9))''').py()
    )


def test_df_astype_value_errors(kx, q):
    df = q('''([] c1:3#.z.p; c2:`abc`def`ghi; c3:1 2 3j;
            c4:("abc";"def";"ghi");c5:"abc";c6:(1 2 3;4 5 6;7 8 9))''')
    # Check errors parameter set to 'ignore'
    assert check_result_and_type(
        kx,
        df.astype({'c6': kx.CharVector}, errors='ignore').py(),
        q('''([] c1:3#.z.p; c2:`abc`def`ghi; c3:1 2 3j;
            c4:("abc";"def";"ghi");c5:"abc";c6:(1 2 3;4 5 6;7 8 9))''').py()
    )
    with pytest.raises(ValueError,
                       match=r"This method can only handle casting string complex columns to "
                       "symbols.  Other complex column data or"
                       " casting to other data is not supported."):
        raise df.astype({'c4': kx.ShortVector})
    with pytest.raises(ValueError,
                       match=r"This method can only handle casting string complex columns to "
                       "symbols.  Other complex column data or"
                       " casting to other data is not supported."):
        raise df.astype({'c6': kx.CharVector})
    with pytest.raises(kx.QError,
                       match=r"Not supported: "
                       "Error casting LongVector to GUIDVector with q error: type"):
        raise df.astype({'c3': kx.GUIDVector})
    with pytest.raises(NotImplementedError,
                       match=r"Currently only the default value of True is accepted for copy"):
        raise df.astype({'c3': kx.ShortVector}, copy='False')
    with pytest.raises(ValueError,
                       match=r"Column name passed in dictionary not present in df table"):
        raise df.astype({'c100': kx.ShortVector})
    with pytest.raises(kx.QError,
                       match=r'Value passed does not match PyKX wrapper type'):
        raise df.astype({'c1': 'nomatchvalue'})
    with pytest.raises(kx.QError,
                       match=r'Value passed does not match PyKX wrapper type'):
        raise df.astype('nomatchvalue')
    df = q('''([] c1:("abc";"def";"ghi");c2:(1 2 3;4 5 6;7 8 9))''')
    with pytest.raises(ValueError,
                       match=r"This method can only handle casting string complex"
                       " columns to symbols.  Other complex column data or"
                       " casting to other data is not supported."):
        raise df.astype(kx.SymbolVector)
    df = q('''([] a:1 2 3 4 5 6;b:`b`n`h``v`;
            c:("ll";"ll";"ll";"ll";"ll";"ll");
            d:("ll";"ll";"ll";"ll";"ll";1 2 3))''')
    with pytest.raises(ValueError,
                       match=r"This method can only handle casting string complex"
                       " columns to symbols.  Other complex column data or"
                       " casting to other data is not supported."):
        raise df.astype({'d': kx.SymbolVector})


def test_df_select_dtypes(kx, q):
    df = q('([] c1:`a`b`c; c2:1 2 3h; c3:1 2 3j; c4:1 2 3i)')
    assert check_result_and_type(
        kx,
        df.select_dtypes(include=[kx.ShortVector, kx.LongVector]).py(),
        q('([] c2:1 2 3h; c3:1 2 3j)').py()
    )
    assert check_result_and_type(
        kx,
        df.select_dtypes(exclude='kx.LongVector').py(),
        q('([] c1:`a`b`c; c2:1 2 3h; c4:1 2 3i)').py()
    )
    assert check_result_and_type(
        kx,
        df.select_dtypes(include=['ShortVector', kx.LongVector],
                         exclude=[kx.SymbolVector]).py(),
        q('([] c2:1 2 3h; c3:1 2 3j;  c4:1 2 3i)').py()
    )


def test_df_select_dtypes_errors(kx, q):
    df = q('([] c1:`a`b`c; c2:1 2 3h; c3:1 2 3j; c4:1 2 3i)')
    with pytest.raises(ValueError, match=r"Expecting either include or"
                       " exclude param to be passed"):
        raise df.select_dtypes()
    with pytest.raises(ValueError, match=r"Include and Exclude lists"
                       " have overlapping elements"):
        df.select_dtypes(include='kx.LongVector',
                         exclude='kx.LongVector')


def test_df_drop(kx, q):
    t = q('([] til 10; 10?10; 10?1f; (10 10)#100?" ")')

    # Test dropping rows from table

    rez = t.drop(5)
    assert(len(rez)==9)
    assert(5 not in rez['x'])

    rez = t.drop([3, 5, 7])
    assert(len(rez)==7)
    assert(all([x not in rez['x'] for x in [3, 5, 7]]))

    rez = t.drop(index=5)
    assert(len(rez)==9)
    assert(5 not in rez['x'])

    rez = t.drop(index=[3, 5, 7])
    assert(len(rez)==7)
    assert(all([x not in rez['x'] for x in [3, 5, 7]]))

    rez = t.drop(-1, errors='ignore')
    assert(q('{x~y}', t, rez).py())

    rez = t.drop([-1, 10], errors='ignore')
    assert(q('{x~y}', t, rez).py())

    # Test dropping columns from table

    rez = t.drop('x1', axis=1)
    assert(len(rez.columns) == 3)
    assert('x1' not in rez.columns)

    rez = t.drop(['x1', 'x3'], axis=1)

    assert(len(rez.columns) == 2)
    assert(all([x not in rez.columns for x in ['x1', 'x3']]))

    rez = t.drop(columns='x1')
    assert(len(rez.columns) == 3)
    assert('x1' not in rez.columns)

    rez = t.drop(columns=['x1', 'x3'])
    assert(len(rez.columns) == 2)
    assert(all([x not in rez.columns for x in ['x1', 'x3']]))

    rez = t.drop('x72', axis=1, errors='ignore')
    assert(q('{x~y}', t, rez).py())

    rez = t.drop(['x42', 'x72'], axis=1, errors='ignore')
    assert(q('{x~y}', t, rez).py())

    # Test dropping rows from keyed table

    q('sym:`aaa`bbb`ccc')
    kt = q('([sym,7?sym; til 10] 10?10; 10?1f; (10 10)#100?" ")')

    key = q('{value exec from key[x] where sym=`aaa}', kt).py()
    rez = kt.drop([key])
    assert(len(rez)==q('{[a;b;c] count delete from key[a] where (sym=b) and x=c}',
                       kt, key[0], key[1]).py())
    assert(key not in q('key', rez))

    keys = q('{(2 2)#raze flip value flip 2#select from key[x] where sym=`aaa}', kt).py()
    rez = kt.drop(keys)
    rez2 = q('{c:{(count[x];2)#raze flip value flip key[x]}[x] in y; count delete from x where c}',
             kt, keys).py()
    assert(len(rez)==rez2)
    assert(not any(q('{{(count[x];2)#raze flip value flip key[x]}[x] in y}', rez, keys).py()))

    rez = kt.drop(index=[key])
    assert(len(rez)==q('{[a;b;c] count delete from key[a] where (sym=b) and x=c}',
                       kt, key[0], key[1]).py())
    assert(key not in q('key', rez))

    rez = kt.drop(index=keys)
    rez2 = q('{c:{(count[x];2)#raze flip value flip key[x]}[x] in y; count delete from x where c}',
             kt, keys).py()
    assert(len(rez)==rez2)
    assert(not any(q('{{(count[x];2)#raze flip value flip key[x]}[x] in y}', rez, keys).py()))

    key = 'aaa'
    rez = kt.drop(key, level=0)
    rez2 = q('{c:y=key[x]`sym; delete from x where c}', kt, key)
    assert(len(rez)==len(rez2))
    assert(not q('{y in key[x]`sym}', rez, key).py())

    keys = ['aaa', 'bbb']
    rez = kt.drop(keys, level=0)
    rez2 = q('{c:(key[x]`sym) in y; delete from x where c}', kt, keys)
    assert(len(rez)==len(rez2))
    assert(not any(q('{(key[x]`sym) in y}', rez, keys).py()))

    keys = [0, 1, 2, 3, 4]
    rez = kt.drop(keys, level=1)
    rez2 = q('{c:(key[x]`x) in y; delete from x where c}', kt, keys)
    assert(len(rez)==len(rez2))
    assert(not any(q('{(key[x]`x) in y}', rez, keys).py()))

    rez = kt.drop([('a', -1), ('zzz', 99)], errors='ignore')
    assert(q('{x~y}', kt, rez).py())

    rez = kt.drop('zzz', level=0, errors='ignore')
    assert(q('{x~y}', kt, rez).py())

    rez = kt.drop(['a', 'zzz'], level=0, errors='ignore')
    assert(q('{x~y}', kt, rez).py())

    # Test dropping columns from keyed table

    rez = kt.drop('x1', axis=1)
    assert(len(rez.columns) == 2)
    assert('x1' not in rez.columns)

    rez = kt.drop(['x', 'x2'], axis=1)
    assert(len(rez.columns) == 1)
    assert(all([x not in rez.columns for x in ['x', 'x2']]))

    rez = kt.drop(columns='x1')
    assert(len(rez.columns) == 2)
    assert('x1' not in rez.columns)

    rez = kt.drop(columns=['x', 'x2'])
    assert(len(rez.columns) == 1)
    assert(all([x not in rez.columns for x in ['x', 'x2']]))

    rez = kt.drop('x72', axis=1, errors='ignore')
    assert(q('{x~y}', kt, rez).py())

    rez = kt.drop(['x42', 'x72'], axis=1, errors='ignore')
    assert(q('{x~y}', kt, rez).py())

    # Test error cases

    with pytest.raises(ValueError):
        t.drop()

    with pytest.raises(ValueError):
        t.drop(4, index=5, columns='x1')

    with pytest.raises(kx.QError):
        t.drop('x1')

    with pytest.raises(kx.QError):
        t.drop(2, axis=1)

    with pytest.raises(ValueError):
        t.drop(0, axis=1, level=0)

    with pytest.raises(kx.QError) as e:
        t.drop(-1)
    assert(str(e.value) == '-1 not found.')

    with pytest.raises(kx.QError) as e:
        t.drop([-1, 10])
    assert(str(e.value) == '-1, 10 not found.')

    with pytest.raises(kx.QError) as e:
        kt.drop([('a', -1), ('zzz', 99)])
    assert(str(e.value) == '(a, -1), (zzz, 99) not found.')

    with pytest.raises(kx.QError) as e:
        kt.drop('zzz', level=0)
    assert(str(e.value) == 'zzz not found.')

    with pytest.raises(kx.QError) as e:
        kt.drop(['a', 'zzz'], level=0)
    assert(str(e.value) == 'a, zzz not found.')

    with pytest.raises(kx.QError) as e:
        t.drop('x42', axis=1)
    assert(str(e.value) == 'x42 not found.')

    with pytest.raises(kx.QError) as e:
        t.drop(['x42', 'x72'], axis=1)
    assert(str(e.value) == 'x42, x72 not found.')

    with pytest.raises(kx.QError) as e:
        kt.drop('x42', axis=1)
    assert(str(e.value) == 'x42 not found.')

    with pytest.raises(kx.QError) as e:
        kt.drop(['x42', 'x72'], axis=1)
    assert(str(e.value) == 'x42, x72 not found.')


def test_df_drop_duplicates(kx, q):
    N = 100
    q['N'] = N
    q('sym:`aaa`bbb`ccc')
    t = q('([] N?sym; N?3)')

    rez = t.drop_duplicates()
    rez2 = t.pd().drop_duplicates().reset_index(drop=True)
    assert(q('{x~y}', rez, rez2))

    with pytest.raises(ValueError):
        t.drop_duplicates(subset=['x', 'x1'])

    with pytest.raises(ValueError):
        t.drop_duplicates(keep='last')

    with pytest.raises(ValueError):
        t.drop_duplicates(inplace=True)

    with pytest.raises(ValueError):
        t.drop_duplicates(ignore_index=True)


def test_df_rename(kx, q):
    q('sym:`aaa`bbb`ccc')
    t = q('([] 10?sym; til 10; 10?10; 10?1f)')

    cols = {'sym': 'Symbol', 'x2': 'xsquare'}
    rez = t.rename(cols, axis=1)
    assert(q('{x~y}', rez, t.pd().rename(cols, axis=1)))

    cols = {'sym': 'Symbol', 'x2': 'xsquare'}
    rez = t.rename(columns=cols)
    assert(q('{x~y}', rez, t.pd().rename(columns=cols)))

    kt = kx.q('([idx:til 10] til 10; 10?10; 10?1f; (10;10)#100?" ")')

    idx = {0: 'foo', 5: 'bar'}
    rez = kt.rename(idx)
    # assert(q('{x~y}', rez, kt.pd().rename(idx)))  # {x~y}=1b because of some q attribute
    assert(all(rez.pd().eq(kt.pd().rename(idx))))

    idx = {0: 'foo', 5: 'bar'}
    rez = kt.rename(index=idx)
    # assert(q('{x~y}', rez, kt.pd().rename(index=idx)))  # {x~y}=1b because of some q attribute
    assert(all(rez.pd().eq(kt.pd().rename(index=idx))))

    with pytest.raises(ValueError):
        t.rename()

    with pytest.raises(ValueError):
        t.rename(index={5: 'foo'}, axis=1)

    with pytest.raises(ValueError):
        t.rename(columns={'x': 'xXx'}, level=0)

    with pytest.raises(ValueError):
        t.rename(columns={'x': 'xXx'}, copy=False)

    with pytest.raises(ValueError):
        t.rename(columns={'x': 'xXx'}, inplace=True)

    with pytest.raises(ValueError):
        t.rename({5: 'foo'}, level=0)

    with pytest.raises(ValueError):
        t.rename(columns={'x': 'xXx'}, errors='raise')


@pytest.mark.pandas_api
@pytest.mark.xfail(reason='Flaky randomization')
def test_df_sample(kx, q):
    q('sym:`aaa`bbb`ccc')
    t = q('([] 10?sym; til 10; 10?10; 10?1f)')
    df = t.pd()
    kt = q('([idx:til 10] til 10; 10?10; 10?1f; (10;10)#100?" ")')
    df2 = kt.pd()

    rez = t.sample()
    assert(type(rez) is kx.Table)
    assert(len(rez) == 1)
    check = df.iloc[rez['x'].py()].reset_index(drop=True)
    assert(q('{x~y}', rez, check))

    rez = t.sample(5)
    assert(type(rez) is kx.Table)
    assert(len(rez) == 5)
    check = df.iloc[rez['x'].py()].reset_index(drop=True)
    assert(q('{x~y}', rez, check))

    rez = t.sample(10)
    assert(type(rez) is kx.Table)
    assert(len(rez) == 10)
    check = df.iloc[rez['x'].py()].reset_index(drop=True)
    assert(q('{x~y}', rez, check))
    assert(q('{x~y}', rez['x'].pd().unique(), check['x'].unique()))

    rez = t.sample(100, replace=True)
    assert(type(rez) is kx.Table)
    assert(len(rez) == 100)
    check = df.iloc[rez['x'].py()].reset_index(drop=True)
    assert(q('{x~y}', rez, check))

    rez = t.sample(frac=0.5, replace=True)
    assert(type(rez) is kx.Table)
    assert(len(rez) == 5)
    check = df.iloc[rez['x'].py()].reset_index(drop=True)
    assert(q('{x~y}', rez, check))

    rez = kt.sample()
    assert(type(rez) is kx.KeyedTable)
    assert(len(rez) == 1)
    check = df2.iloc[rez['x'].py()]
    assert(q('{x~y}', rez, check))

    rez = kt.sample(5)
    assert(type(rez) is kx.KeyedTable)
    assert(len(rez) == 5)
    check = df2.iloc[rez['x'].py()]
    assert(q('{x~y}', rez, check))

    rez = kt.sample(10)
    assert(type(rez) is kx.KeyedTable)
    assert(len(rez) == 10)
    check = df2.iloc[rez['x'].py()]
    assert(q('{x~y}', rez, check))
    assert(q('{x~y}', rez['x'].pd().unique(), check['x'].unique()))

    rez = kt.sample(100, replace=True)
    assert(type(rez) is kx.KeyedTable)
    assert(len(rez) == 100)
    check = df2.iloc[rez['x'].py()]
    assert(q('{x~y}', rez, check))

    rez = kt.sample(frac=0.5, replace=True)
    assert(type(rez) is kx.KeyedTable)
    assert(len(rez) == 5)
    check = df2.iloc[rez['x'].py()]
    assert(q('{x~y}', rez, check))

    with pytest.raises(ValueError):
        t.sample(100)

    with pytest.raises(ValueError):
        t.sample(weights=np.ones(10))

    with pytest.raises(ValueError):
        t.sample(random_state=42)

    with pytest.raises(ValueError):
        t.sample(axis=1)

    with pytest.raises(ValueError):
        t.sample(ignore_index=True)


def test_mean(kx, q):
    df = pd.DataFrame(
        {
            'a': [1, 2, 2, 4],
            'b': [1, 2, 6, 7],
            'c': [7, 8, 9, 10],
            'd': [7, 11, 14, 14]
        }
    )
    tab = kx.toq(df)
    p_m = df.mean()
    q_m = tab.mean()
    for c in q.key(q_m).py():
        assert p_m[c] == q_m[c].py()
    p_m = df.mean(axis=1)
    q_m = tab.mean(axis=1)
    for c in range(len(q.cols(tab))):
        assert p_m[c] == q_m[q('{`$string x}', c)].py()

    q['tab'] = kx.toq(df)
    tab = q('1!`idx xcols update idx: til count tab from tab')
    p_m = df.mean()
    q_m = tab.mean()
    for c in q.key(q_m).py():
        assert p_m[c] == q_m[c].py()
    p_m = df.mean(axis=1)
    q_m = tab.mean(axis=1)
    for c in range(len(q.cols(tab)) - 1):
        assert p_m[c] == q_m[q('{`$string x}', c)].py()

    df = pd.DataFrame(
        {
            'a': [1, 2, 2, 4],
            'b': [1, 2, 6, 7],
            'c': [7, 8, 9, 10],
            'd': ['foo', 'bar', 'baz', 'qux']
        }
    )
    tab = kx.toq(df)
    p_m = df.mean(numeric_only=True)
    q_m = tab.mean(numeric_only=True)
    for c in q.key(q_m).py():
        assert p_m[c] == q_m[c].py()
    p_m = df.mean(axis=1, numeric_only=True)
    q_m = tab.mean(axis=1, numeric_only=True)
    for c in range(len(q.cols(tab))):
        assert p_m[c] == q_m[q('{`$string x}', c)].py()

    with pytest.raises(kx.QError):
        q_m = tab.mean()
    with pytest.raises(kx.QError):
        q_m = tab.mean(axis=1)


def test_median(kx, q):
    df = pd.DataFrame(
        {
            'a': [1, 2, 2, 4],
            'b': [1, 2, 6, 7],
            'c': [7, 8, 9, 10],
            'd': [7, 11, 14, 14]
        }
    )
    tab = kx.toq(df)
    p_m = df.median()
    q_m = tab.median()
    for c in q.key(q_m).py():
        assert p_m[c] == q_m[c].py()
    p_m = df.median(axis=1)
    q_m = tab.median(axis=1)
    for c in range(len(q.cols(tab))):
        assert p_m[c] == q_m[q('{`$string x}', c)].py()

    q['tab'] = kx.toq(df)
    tab = q('1!`idx xcols update idx: til count tab from tab')
    p_m = df.median()
    q_m = tab.median()
    for c in q.key(q_m).py():
        assert p_m[c] == q_m[c].py()
    p_m = df.median(axis=1)
    q_m = tab.median(axis=1)
    for c in range(len(q.cols(tab)) - 1):
        assert p_m[c] == q_m[q('{`$string x}', c)].py()

    df = pd.DataFrame(
        {
            'a': [1, 2, 2, 4],
            'b': [1, 2, 6, 7],
            'c': [7, 8, 9, 10],
            'd': ['foo', 'bar', 'baz', 'qux']
        }
    )
    tab = kx.toq(df)
    p_m = df.median(numeric_only=True)
    q_m = tab.median(numeric_only=True)
    for c in q.key(q_m).py():
        assert p_m[c] == q_m[c].py()
    p_m = df.median(axis=1, numeric_only=True)
    q_m = tab.median(axis=1, numeric_only=True)
    for c in range(len(q.cols(tab))):
        assert p_m[c] == q_m[q('{`$string x}', c)].py()

    with pytest.raises(kx.QError):
        q_m = tab.median()
    with pytest.raises(kx.QError):
        q_m = tab.median(axis=1)


def test_mode(kx, q): # noqa
    if sys.version_info.minor > 7:
        def compare_q_to_pd(tab, df):
            if 'idx' in q.cols(tab):
                tab.pop('idx')
            tab = tab.pd()
            for i in range(len(tab)):
                for c in tab.columns:
                    df_c = c
                    try:
                        df_c = int(c)
                    except BaseException:
                        pass
                    if str(tab.at[i, c]) == '--':
                        tab.at[i, c] = np.NaN
                    if str(tab.at[i, c]) == '':
                        tab.at[i, c] = 'nan'
                    if str(tab.at[i, c]) == 'nan' and str(df.at[i, df_c]) == 'nan':
                        continue
                    if tab.at[i, c] != df.at[i, df_c]:
                        return False
            return True

        df = pd.DataFrame(
            {
                'a': [1, 2, 2, 4],
                'b': [1, 2, 6, 7],
                'c': [7, 8, 9, 10],
                'd': [7, 11, 14, 14]
            }
        )
        tab = kx.toq(df)
        p_m = df.mode()
        q_m = tab.mode()
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(axis=1)
        q_m = tab.mode(axis=1)
        assert compare_q_to_pd(q_m, p_m)

        q['tab'] = kx.toq(df)
        tab = q('1!`idx xcols update idx: til count tab from tab')

        p_m = df.mode()
        q_m = tab.mode()
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(axis=1)
        q_m = tab.mode(axis=1)
        assert compare_q_to_pd(q_m, p_m)

        df = pd.DataFrame(
            {
                'a': [1, 2, 2, 4],
                'b': [1, 2, 6, 7],
                'c': [7, 8, 9, 10],
                'd': ['foo', 'bar', 'baz', 'foo']
            }
        )
        tab = kx.toq(df)
        p_m = df.mode()
        q_m = tab.mode()
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(axis=1, numeric_only=True)
        q_m = tab.mode(axis=1, numeric_only=True)
        assert compare_q_to_pd(q_m, p_m)

        df = pd.DataFrame({
            'x': [0, 1, 2, 3, 4, 5, 6, 7, np.NaN, np.NaN],
            'y': [10, 11, 12, 13, 14, 15, 16, 17, 18, np.NaN],
            'z': ['a', 'b', 'c', 'd', 'd', 'e', 'e', 'f', 'g', 'h']
        })
        tab = kx.toq(df)

        p_m = df.mode()
        q_m = tab.mode()
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(axis=1, numeric_only=True)
        q_m = tab.mode(axis=1, numeric_only=True)
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(numeric_only=True)
        q_m = tab.mode(numeric_only=True)
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(axis=1, numeric_only=True)
        q_m = tab.mode(axis=1, numeric_only=True)
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(dropna=False)
        q_m = tab.mode(dropna=False)
        assert compare_q_to_pd(q_m, p_m)

        p_m = df.mode(axis=1, dropna=False, numeric_only=True)
        q_m = tab.mode(axis=1, dropna=False, numeric_only=True)
        assert compare_q_to_pd(q_m, p_m)


def test_table_merge_asof(kx, q):
    left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
    right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})
    qleft = kx.toq(left)
    qright = kx.toq(right)

    assert (pd.merge_asof(left, right, on='a')
            == kx.merge_asof(qleft, qright, on='a').pd()).all().all()
    assert (pd.merge_asof(left, right, on='a')
            == qleft.merge_asof(qright, on='a').pd()).all().all()
    assert (pd.merge_asof(left, right, on='a')
            == q('0!', q('1!', qleft).merge_asof(qright, on='a')).pd()).all().all()
    assert (pd.merge_asof(left, right, on='a')
            == qleft.merge_asof(q('1!', qright), on='a').pd()).all().all()
    assert (pd.merge_asof(left, right, on='a')
            == q('0!', q('1!', qleft).merge_asof(q('1!', qright), on='a')).pd()).all().all()
    left = pd.DataFrame({
        "time": [
            pd.Timestamp("2016-05-25 13:30:00.023"),
            pd.Timestamp("2016-05-25 13:30:00.023"),
            pd.Timestamp("2016-05-25 13:30:00.030"),
            pd.Timestamp("2016-05-25 13:30:00.041"),
            pd.Timestamp("2016-05-25 13:30:00.048"),
            pd.Timestamp("2016-05-25 13:30:00.049"),
            pd.Timestamp("2016-05-25 13:30:00.072"),
            pd.Timestamp("2016-05-25 13:30:00.075")
        ],
        "ticker": [
            "GOOG",
            "MSFT",
            "MSFT",
            "MSFT",
            "GOOG",
            "AAPL",
            "GOOG",
            "MSFT"
        ],
        "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
        "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03]
    })
    right = pd.DataFrame({
        "time": [
            pd.Timestamp("2016-05-25 13:30:00.023"),
            pd.Timestamp("2016-05-25 13:30:00.038"),
            pd.Timestamp("2016-05-25 13:30:00.048"),
            pd.Timestamp("2016-05-25 13:30:00.048"),
            pd.Timestamp("2016-05-25 13:30:00.048")
        ],
        "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
        "price": [51.95, 51.95, 720.77, 720.92, 98.0],
        "quantity": [75, 155, 100, 100, 100]
    })

    qleft = kx.toq(left)
    qright = kx.toq(right)

    assert (pd.merge_asof(left, right, on='time')
            == kx.merge_asof(qleft, qright, on='time').pd()).all().all()
    assert (pd.merge_asof(left, right, on='time')
            == qleft.merge_asof(qright, on='time').pd()).all().all()
    assert (pd.merge_asof(left, right, on='time')
            == q('0!', q('1!', qleft).merge_asof(qright, on='time')).pd()).all().all()
    assert (pd.merge_asof(left, right, on='time')
            == qleft.merge_asof(q('1!', qright), on='time').pd()).all().all()
    assert (pd.merge_asof(left, right, on='time')
            == q('0!', q('1!', qleft).merge_asof(q('1!', qright), on='time')).pd()).all().all()


def test_pandas_abs(kx, q):
    tab = q('([] sym: 100?`foo`bar`baz`qux; price: 250.0f - 100?500.0f; ints: 100 - 100?200)')
    ntab = tab[['price', 'ints']]

    assert ntab.abs().py() == tab.abs(numeric_only=True).py()

    with pytest.raises(kx.QError):
        tab.abs()


def test_pandas_min(q):
    tab = q('([] sym: 100?`foo`bar`baz`qux; price: 250.0f - 100?500.0f; ints: 100 - 100?200)')
    df = tab.pd()

    qmin = tab.min().py()
    pmin = df.min()

    assert str(pmin['sym']) == qmin['sym']
    assert float(pmin['price']) == qmin['price']
    assert float(pmin['ints']) == qmin['ints']

    qmin = tab.min(axis=1, numeric_only=True, skipna=True).py()
    pmin = df.min(axis=1, numeric_only=True, skipna=True)

    for i in range(100):
        assert float(qmin[i]) == float(pmin[i])


def test_pandas_max(q):
    tab = q('([] sym: 100?`foo`bar`baz`qux; price: 250.0f - 100?500.0f; ints: 100 - 100?200)')
    df = tab.pd()

    qmax = tab.max().py()
    pmax = df.max()

    assert str(pmax['sym']) == qmax['sym']
    assert float(pmax['price']) == qmax['price']
    assert float(pmax['ints']) == qmax['ints']

    qmax = tab.max(axis=1, numeric_only=True, skipna=True).py()
    pmax = df.max(axis=1, numeric_only=True, skipna=True)

    for i in range(100):
        assert float(qmax[i]) == float(pmax[i])


def test_pandas_all(q):
    tab = q(
        '([] sym: 100?`foo`bar`baz`qux; price: 250.0f - 100?500.0f; ints: 100 - 100?200;'
        ' bools: 100?0b)'
    )
    df = tab.pd()

    qall = tab.all().py()
    pall = df.all()
    assert qall['sym'] == pall['sym']
    assert qall['ints'] == pall['ints']
    assert qall['price'] == pall['price']
    assert qall['bools'] == pall['bools']

    qall = tab.all(bool_only=True).py()
    pall = df.all(bool_only=True)
    assert qall['bools'] == pall['bools']

    qall = tab.all(axis=1).py()
    pall = df.all(axis=1)
    for i in range(100):
        assert qall[i] == pall[i]


def test_pandas_any(q):
    tab = q(
        '([] sym: 100?`foo`bar`baz`qux; price: 250.0f - 100?500.0f; ints: 100 - 100?200;'
        ' bools: 100?0b)'
    )
    df = tab.pd()

    qany = tab.any().py()
    pany = df.any()
    assert qany['sym'] == pany['sym']
    assert qany['ints'] == pany['ints']
    assert qany['price'] == pany['price']
    assert qany['bools'] == pany['bools']

    qany = tab.any(bool_only=True).py()
    pany = df.any(bool_only=True)
    assert qany['bools'] == pany['bools']

    qany = tab.any(axis=1).py()
    pany = df.any(axis=1)
    for i in range(100):
        assert qany[i] == pany[i]


def test_pandas_prod(q):
    tab = q('([] sym: 10?`a`b`c; price: 12.25f - 10?25.0f; ints: 10 - 10?20)')
    df = tab.pd()

    qprod = tab.prod(numeric_only=True).py()
    pprod = df.prod(numeric_only=True)
    assert float(qprod['price']) == float(pprod['price'])
    assert float(qprod['ints']) == float(pprod['ints'])

    qprod = tab.prod(numeric_only=True, skipna=True, axis=1).py()
    pprod = df.prod(numeric_only=True, skipna=True, axis=1)
    for i in range(10):
        assert float(qprod[i]) == float(pprod[i])

    qprod = tab.prod(numeric_only=True, skipna=True, axis=1, min_count=5).py()
    pprod = df.prod(numeric_only=True, skipna=True, axis=1, min_count=5)
    for i in range(10):
        assert qprod[i] == q('0N')
        assert str(pprod[i]) == 'nan'


def test_pandas_sum(q):
    tab = q('([] sym: 100?`foo`bar`baz`qux; price: 250.0f - 100?500.0f; ints: 100 - 100?200)')
    df = tab.pd()

    qsum = tab.sum().py()
    psum = df.sum()
    assert float(qsum['price']) == float(psum['price'])
    assert float(qsum['ints']) == float(psum['ints'])
    assert str(qsum['sym']) == str(psum['sym'])

    qsum = tab.sum(numeric_only=True, skipna=True, axis=1).py()
    psum = df.sum(numeric_only=True, skipna=True, axis=1)
    for i in range(10):
        assert float(qsum[i]) == float(psum[i])

    qsum = tab.sum(numeric_only=True, skipna=True, axis=1, min_count=5).py()
    psum = df.sum(numeric_only=True, skipna=True, axis=1, min_count=5)
    for i in range(10):
        assert qsum[i] == q('0N')
        assert str(psum[i]) == 'nan'


def test_pandas_groupby_errors(kx, q):
    tab = q('([] sym: 100?`foo`bar`baz`qux; price: 250.0f - 100?500.0f; ints: 100 - 100?200)')

    with pytest.raises(RuntimeError):
        tab.groupby(by='sym', level=[1])

    with pytest.raises(NotImplementedError):
        tab.groupby(by=lambda x: x)
    with pytest.raises(NotImplementedError):
        tab.groupby(by='sym', observed=True)
    with pytest.raises(NotImplementedError):
        tab.groupby(by='sym', group_keys=False)
    with pytest.raises(NotImplementedError):
        tab.groupby(by='sym', axis=1)

    arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot'],
              ['Captive', 'Wild', 'Captive', 'Wild', 'Wild']]
    index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
    df = pd.DataFrame({'Max Speed': [390., 350., 30., 20., 25.]},
                      index=index)
    tab = kx.toq(df)

    with pytest.raises(KeyError):
        tab.groupby(level=[0, 4])


def test_pandas_groupby(kx, q):
    df = pd.DataFrame(
        {
            'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
            'Max Speed': [380., 370., 24., 26.],
            'Max Altitude': [570., 555., 275., 300.]
        }
    )

    tab = kx.toq(df)

    assert all(
        df.groupby(['Animal']).mean() == tab.groupby(kx.SymbolVector(['Animal'])).mean().pd()
    )
    assert df.groupby(['Animal']).ndim == tab.groupby(kx.SymbolVector(['Animal'])).ndim
    assert all(
        df.groupby(['Animal'], as_index=False).mean()
        == tab.groupby(kx.SymbolVector(['Animal']), as_index=False).mean().pd()
    )
    assert all(
        df.groupby(['Animal']).tail(1).reset_index(drop=True)
        == tab.groupby(kx.SymbolVector(['Animal'])).tail(1).pd()
    )
    assert all(
        df.groupby(['Animal']).tail(2)
        == tab.groupby(kx.SymbolVector(['Animal'])).tail(2).pd()
    )

    df = pd.DataFrame(
        [
            ["a", 12, 12],
            [None, 12.3, 33.],
            ["b", 12.3, 123],
            ["a", 1, 1]
        ],
        columns=["a", "b", "c"]
    )
    tab = kx.toq(df)

    # NaN in column is filled when converted to q this unfills it and re-sorts it
    assert q(
        '{[x; y] x:update a:` from x where i=2; x: `a xasc x; x~y}',
        df.groupby('a', dropna=False).sum(),
        tab.groupby('a', dropna=False).sum()
    )
    assert q(
        '{[x; y] x:update a:` from x where i=1; x~y}',
        df.groupby('a', dropna=False, sort=False).sum(),
        tab.groupby('a', dropna=False, sort=False).sum()
    )
    assert all(
        df.groupby('a', dropna=False, as_index=False).sum()
        == tab.groupby('a', dropna=False, as_index=False).sum().pd()
    )

    arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot'],
              ['Captive', 'Wild', 'Captive', 'Wild', 'Wild']]
    index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
    df = pd.DataFrame({'Max Speed': [390., 350., 30., 20., 25.]},
                      index=index)
    tab = kx.toq(df)

    assert all(
        df.groupby(['Animal']).mean()
        == tab.groupby(['Animal']).mean().pd()
    )
    assert all(
        df.groupby(['Animal'], as_index=False).mean()
        == tab.groupby(['Animal'], as_index=False).mean().pd()
    )

    assert all(
        df.groupby(level=[1]).mean()
        == tab.groupby(level=[1]).mean().pd()
    )
    assert all(
        df.groupby(level=1, as_index=False).mean()
        == tab.groupby(level=1, as_index=False).mean().pd()
    )

    assert all(
        df.groupby(level=[0, 1]).mean()
        == tab.groupby(level=[0, 1]).mean().pd()
    )
    assert all(
        df.groupby(level=[0, 1], as_index=False).mean()
        == tab.groupby(level=[0, 1], as_index=False).mean().pd()
    )


def test_keyed_loc_fixes(q):
    mkt = q('([k1:`a`b`a;k2:100+til 3] x:til 3; y:`multi`keyed`table)')
    assert q.keys(mkt['x']).py() == ['k1', 'k2']
    assert q.value(mkt['x']).py() == {'x': [0, 1, 2]}
    assert mkt[['x', 'y']].pd().equals(mkt.pd()[['x', 'y']])
    assert mkt['a', 100].py() == {'x': [0], 'y': ['multi']}

    with pytest.raises(KeyError):
        mkt[['k1', 'y']]
    with pytest.raises(KeyError):
        mkt['k1']


def test_isnull(q):
    tab = q('''([]
        g:1#0Ng;    h:1#0Nh;    i1:1#0Ni; j:1#0Nj;
        e:1#0Ne;    f:1#0Nf;    s:1#`  ;  p:1#0Np;
        m:1#0Nm;    d:1#0Nd;    n:1#0Nn;  u:1#0Nu;
        v:1#0Nv;    t:1#0Nt;    c:1#" ";
        g2:1?0Ng;   h2:1?0Wh;   i2:1?10i; j2:1?10j;
        e2:1?10e;   f2:1?10f;   s2:1#`foo;p2:1?10p;
        m2:1?"m"$10;d2:1?"d"$10;n2:1?10n; u2:1?10u;
        v2:1?10v;   t2:1?10t;   c2:1?" ")
        ''')

    cols = ["g", "h", "i1", "j",
            "e", "f", "s", "p",
            "m", "d", "n", "u",
            "v", "t", "c",
            "g2", "h2", "i2", "j2",
            "e2", "f2", "s2", "p2",
            "m2", "d2", "n2", "u2",
            "v2", "t2", "c2"]

    expected = pd.DataFrame.from_dict({c: [True] if i < 15 else [False]
                                       for i, c in enumerate(cols)})
    expected_inv = ~expected

    pd.testing.assert_frame_equal(tab.isna().pd(), expected)
    pd.testing.assert_frame_equal(tab.isnull().pd(), expected)
    pd.testing.assert_frame_equal(tab.notna().pd(), expected_inv)
    pd.testing.assert_frame_equal(tab.notnull().pd(), expected_inv)

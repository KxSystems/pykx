import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)

NATIVE_AGGS = ['sum', 'prod', 'min', 'max', 'mean', 'median', 'count']


def make_df(rows=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'k': rng.choice(['a', 'b', 'c', 'd'], rows),
        'k2': rng.choice(['x', 'y'], rows),
        'f': np.where(rng.random(rows) < 0.1, np.nan, rng.random(rows) * 10 - 5),
        'n': rng.integers(-3, 4, rows),
        'b': rng.random(rows) < 0.5,
    })


@pytest.fixture
def spy(monkeypatch):
    """Record the aggregations that were lowered to native q."""
    import pykx.pandas_api as pandas_api
    import pykx.pandas_api.pandas_meta as pandas_meta
    calls = []

    def wrap(module, name):
        orig = getattr(module, name)

        def inner(*args, **kwargs):
            res = orig(*args, **kwargs)
            calls.append((name, args[0], res is not None))
            return res
        monkeypatch.setattr(module, name, inner)

    wrap(pandas_api, 'native_groupby_agg')
    wrap(pandas_meta, 'native_agg')
    return calls


def assert_series_matches(res, expected):
    res = res.py()
    assert list(res.keys()) == list(expected.index)
    np.testing.assert_allclose(
        [np.nan if res[k] is None or res[k] is pd.NA else float(res[k]) for k in expected.index],
        expected.to_numpy(dtype=float),
        equal_nan=True
    )


@pytest.mark.parametrize('name', ['sum', 'prod', 'min', 'max', 'count', 'any', 'all'])
def test_ungrouped_matches_pandas(kx, name):
    df = make_df()[['f', 'n', 'b']]
    tab = kx.toq(df)
    assert_series_matches(getattr(tab, name)(), getattr(df, name)())


@pytest.mark.parametrize('name', ['sum', 'prod', 'min', 'max', 'count'])
def test_ungrouped_numeric_only(kx, name):
    df = make_df()
    tab = kx.toq(df)
    expected = getattr(df[['f', 'n', 'b']], name)()
    assert_series_matches(getattr(tab, name)(numeric_only=True), expected)


def test_ungrouped_native_path_used(kx, spy):
    tab = kx.toq(make_df()[['f', 'n']])
    for name in ['sum', 'prod', 'min', 'max', 'count', 'any', 'all']:
        getattr(tab, name)()
    assert [c[0] for c in spy] == ['native_agg'] * 7
    assert all(c[2] for c in spy)


def test_ungrouped_native_agg_keyed_table(kx):
    df = make_df()[['n', 'f', 'b']]
    tab = kx.q('1!', kx.toq(df))
    assert_series_matches(tab.sum(), df.set_index('n').sum())


@pytest.mark.parametrize('kwargs', [
    {'min_count': 400},
])
def test_ungrouped_fallback_matches_pandas(kx, spy, kwargs):
    df = make_df()[['f', 'n']]
    tab = kx.toq(df)
    res = tab.sum(**kwargs)
    expected = df.sum(**kwargs)
    assert_series_matches(res, expected)
    # None of these can run natively but they must not be reported as native
    assert not any(c[2] for c in spy)


def test_ungrouped_symbol_columns_fall_back(kx, spy):
    tab = kx.q('([] k:`a`b`a; f:1 2 3f)')
    assert tab.sum().py() == {'k': 'aba', 'f': 6.0}
    assert not any(c[2] for c in spy)


@pytest.mark.parametrize('name', ['min', 'max', 'idxmax', 'idxmin'])
def test_symbol_columns_match_pandas(kx, spy, name):
    tab = kx.q('([] s:`b`a``c`a; f:1 2 3 4 5f)')
    df = pd.DataFrame({'s': ['b', 'a', np.nan, 'c', 'a'], 'f': [1, 2, 3, 4, 5.0]})
    assert getattr(tab, name)().py() == getattr(df, name)().to_dict()
    assert [c[2] for c in spy if c[0] == 'native_agg'] == [True]
    res = getattr(tab.groupby('s'), name)().pd()
    pd.testing.assert_frame_equal(res, getattr(df.groupby('s'), name)(), check_dtype=False,
                                  check_index_type=False)


def test_symbol_min_max_skipna_false(kx):
    tab = kx.q('([] s:`b`a``c)')
    assert tab.min(skipna=False).py() == {'s': ''}
    assert tab.max(skipna=False).py() == {'s': ''}
    assert tab.max().py() == {'s': 'c'}


def test_any_all_semantics(kx):
    tab = kx.q('([] z:0 0 0; n:1 2 3; m:0 1 0; f:0.5 0n 0f; s:`a``b; e:3#0b; t:3#1b)')
    assert tab.all().py() == {'z': False, 'n': True, 'm': False, 'f': False, 's': True,
                              'e': False, 't': True}
    assert kx.q('([] s:`a``b)').all(skipna=False).py() == {'s': True}
    assert kx.q('([] k:`a`a`b; s:`x``y)').groupby('k')['s'].all().py() == {
        ('a',): {'s': True}, ('b',): {'s': True}}
    assert tab.any().py() == {'z': False, 'n': True, 'm': True, 'f': True, 's': True,
                              'e': False, 't': True}
    # A null is true unless nulls are skipped
    nulls = kx.q('([] f:1 0n 2f)')
    assert nulls.all().py() == {'f': True}
    assert nulls.all(skipna=False).py() == {'f': True}
    assert kx.q('([] f:0 0n 0f)').any().py() == {'f': False}
    assert kx.q('([] f:0 0n 0f)').any(skipna=False).py() == {'f': True}
    empty = kx.q('0#([] f:1 2f)')
    assert empty.all().py() == {'f': True}
    assert empty.any().py() == {'f': False}


def test_any_all_large_table_is_fast(kx):
    tab = kx.q('([] a:1000000?1f; b:1000000?1f)')
    assert tab.all().py() == {'a': True, 'b': True}
    assert tab.any().py() == {'a': True, 'b': True}


def test_any_all_bool_only(kx):
    tab = kx.q('([] a:101b; b:1 2 3; c:111b)')
    assert tab.all(bool_only=True).py() == {'a': False, 'c': True}
    assert tab.any(bool_only=True).py() == {'a': True, 'c': True}
    assert tab[['b']].all(bool_only=True).py() == {}


def grouped_expected(df, by, name, cols, **kwargs):
    return getattr(df.groupby(by, **kwargs)[cols], name)()


@pytest.mark.parametrize('name', NATIVE_AGGS)
@pytest.mark.parametrize('by', ['k', ['k', 'k2']])
def test_grouped_matches_pandas(kx, spy, name, by):
    df = make_df()
    tab = kx.toq(df)
    cols = ['f', 'n']
    res = getattr(tab.groupby(by)[cols], name)().pd()
    expected = grouped_expected(df, by, name, cols)
    pd.testing.assert_frame_equal(res, expected, check_dtype=False, check_exact=False)
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True]


@pytest.mark.parametrize('name, kwargs', [
    ('mean', {'skipna': False}),
    ('std', {'ddof': -1}),
    ('sem', {'ddof': 1.5}),
    ('sum', {'min_count': 2}),
])
def test_unsupported_arguments_are_not_lowered(name, kwargs):
    from pykx.pandas_api.pandas_native_agg import _resolve
    assert _resolve(name, kwargs) is None


@pytest.mark.parametrize('name', ['skew', 'kurt', 'sem', 'idxmax', 'idxmin'])
def test_grouped_statistics_match_pandas(kx, spy, name):
    df = make_df()
    tab = kx.toq(df)
    res = getattr(tab.groupby('k')[['f', 'n']], name)().pd()
    expected = getattr(df.groupby('k')[['f', 'n']], name)()
    pd.testing.assert_frame_equal(res, expected, check_dtype=False, check_exact=False)
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True]


@pytest.mark.parametrize('name', ['any', 'all'])
def test_grouped_any_all_matches_pandas(kx, spy, name):
    df = make_df()
    tab = kx.toq(df)
    res = getattr(tab.groupby('k')[['f', 'n', 'b']], name)().pd()
    expected = grouped_expected(df, 'k', name, ['f', 'n', 'b'])
    pd.testing.assert_frame_equal(res, expected, check_dtype=False)
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True]


@pytest.mark.parametrize('ddof', [0, 1])
def test_grouped_std_matches_pandas(kx, spy, ddof):
    df = make_df()
    tab = kx.toq(df)
    res = tab.groupby('k')[['f', 'n']].std(ddof=ddof).pd()
    expected = df.groupby('k')[['f', 'n']].std(ddof=ddof)
    pd.testing.assert_frame_equal(res, expected, check_dtype=False, check_exact=False)
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True]


def test_grouped_single_column_selection(kx):
    df = make_df()
    tab = kx.toq(df)
    res = tab.groupby('k')['f'].sum().pd()
    pd.testing.assert_frame_equal(res, df.groupby('k')[['f']].sum(), check_exact=False)


def test_grouped_default_columns_exclude_keys(kx):
    df = make_df()[['k', 'f', 'n']]
    tab = kx.toq(df)
    pd.testing.assert_frame_equal(
        tab.groupby('k').sum().pd(), df.groupby('k').sum(), check_dtype=False, check_exact=False
    )


@pytest.mark.parametrize('name', ['sum', 'min', 'count'])
def test_grouped_as_index_false(kx, name):
    df = make_df()
    tab = kx.toq(df)
    res = getattr(tab.groupby('k', as_index=False)[['f', 'n']], name)().pd()
    expected = getattr(df.groupby('k', as_index=False)[['f', 'n']], name)()
    pd.testing.assert_frame_equal(
        res.reset_index(drop=True), expected, check_dtype=False, check_exact=False
    )


@pytest.mark.parametrize('cols', [['v'], 'v'])
def test_grouped_keyed_table_as_index_false_keeps_keys(kx, cols):
    tab = kx.q('([k:`a`b`a] v:1 2 3; w:4 5 6)')
    res = tab.groupby(level=0, as_index=False)[cols].sum().pd()
    expected = tab.pd().groupby(level=0, as_index=False)[cols].sum()
    pd.testing.assert_frame_equal(res.reset_index(drop=True), expected, check_dtype=False)


def test_grouped_sort_false_keeps_first_appearance_order(kx):
    tab = kx.q('([] k:`c`a`b`a`c; v:1 2 3 4 5)')
    res = tab.groupby('k', sort=False)['v'].sum().pd()
    assert list(res.index) == ['c', 'a', 'b']
    assert list(res['v']) == [6, 6, 3]
    res = tab.groupby('k', sort=True)['v'].sum().pd()
    assert list(res.index) == ['a', 'b', 'c']


def test_grouped_null_keys(kx):
    tab = kx.q('([] k:`a``b`a`; v:1 2 3 4 5)')
    dropped = tab.groupby('k')['v'].sum().pd()
    assert list(dropped.index) == ['a', 'b']
    assert list(dropped['v']) == [5, 3]
    kept = tab.groupby('k', dropna=False)['v'].sum().pd()
    assert list(kept['v']) == [7, 5, 3]


@pytest.mark.parametrize('kwargs', [{}, {'min_count': 1}])
@pytest.mark.parametrize('by', [['a', 'b'], ['b', 'a']])
def test_grouped_null_in_any_key_is_dropped(kx, kwargs, by):
    tab = kx.q('([] a:1 1 0n 2 2f; b:10 0n 30 40 40f; v:1 2 3 4 5)')
    res = tab.groupby(by)['v'].sum(**kwargs).pd()
    expected = tab.pd().groupby(by)[['v']].sum(**kwargs)
    pd.testing.assert_frame_equal(res, expected, check_dtype=False)


def test_grouped_null_values(kx):
    tab = kx.q('([] k:`a`a`b`b`c; v:1 0n 3 4 0n)')
    assert list(tab.groupby('k')['v'].sum().pd()['v']) == [1, 7, 0]
    assert list(tab.groupby('k')['v'].count().pd()['v']) == [1, 2, 0]
    assert list(tab.groupby('k')['v'].max().pd()['v'][:2]) == [1, 4]


def test_grouped_native_path_skips_grouping(kx):
    tab = kx.q('([] k:`a`b`a; v:1 2 3)')
    grouped = tab.groupby('k')
    assert grouped._tab is None
    grouped['v'].sum()
    assert grouped._tab is None
    # Anything which needs the groups builds them on demand
    assert len(grouped.tail(1)) == 2
    assert grouped._tab is not None


def test_grouped_bad_column_raises_on_groupby(kx):
    tab = kx.q('([] k:`a`b`a; v:1 2 3)')
    with pytest.raises(kx.QError):
        tab.groupby('missing')


@pytest.mark.parametrize('kwargs', [
    {'min_count': 3},
])
def test_grouped_fallback_matches_pandas(kx, spy, kwargs):
    df = make_df()
    tab = kx.toq(df)
    res = tab.groupby('k')[['f', 'n']].sum(**kwargs).pd()
    expected = df.groupby('k')[['f', 'n']].sum(**kwargs)
    pd.testing.assert_frame_equal(res, expected, check_dtype=False, check_exact=False)
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [False]


def test_grouped_symbol_columns_fall_back(kx, spy):
    tab = kx.q('([] k:`a`b`a; s:`x`y`z; v:1 2 3)')
    res = tab.groupby('k')[['v']].sum().pd()
    assert list(res['v']) == [4, 2]
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True]
    tab.groupby('k')[['s']].sum()
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True, False]


def test_grouped_keyed_table_falls_back(kx, spy):
    tab = kx.q('1!([] k:`a`b`c; l:`x`x`y; v:1 2 3)')
    assert list(tab.groupby('l')[['v']].sum().pd()['v']) == [3, 3]
    assert not any(c[2] for c in spy if c[0] == 'native_groupby_agg')


def test_function_names_survive_decorators(kx):
    from pykx.pandas_api.pandas_meta import PandasMeta
    for name in ['sum', 'min', 'max', 'prod', 'count', 'any', 'all', 'mean']:
        assert getattr(PandasMeta, name).__name__ == name


def test_grouped_column_named_i_is_not_lowered(kx):
    # `i` is the row index inside a q select, so it cannot be aggregated natively
    from pykx.pandas_api.pandas_native_agg import native_groupby_agg
    tab = kx.q('([] k:`a`b`a; i:10 20 30; v:1 2 3)')
    assert native_groupby_agg('sum', tab.groupby('k')[['i']], (), {}) is None
    assert native_groupby_agg('sum', tab.groupby('k')[['v']], (), {}) is not None


@pytest.mark.parametrize('name', ['sum', 'prod', 'min', 'max'])
def test_skipna_false_matches_pandas(kx, spy, name):
    df = make_df()[['k', 'f', 'n']]
    tab = kx.toq(df)
    cols = ['f', 'n']
    assert_series_matches(getattr(tab[cols], name)(skipna=False),
                          getattr(df[cols], name)(skipna=False))
    res = getattr(tab.groupby('k')[cols], name)(skipna=False).pd()
    expected = getattr(df.groupby('k')[cols], name)(skipna=False)
    pd.testing.assert_frame_equal(res, expected, check_dtype=False, check_exact=False)
    assert all(c[2] for c in spy)


@pytest.mark.parametrize('name', ['mean', 'median', 'std', 'sem', 'skew', 'kurt',
                                  'idxmax', 'idxmin'])
def test_ungrouped_statistics_match_pandas(kx, spy, name):
    df = make_df()[['f', 'n']]
    tab = kx.toq(df)
    assert_series_matches(getattr(tab, name)(), getattr(df, name)())
    assert [c[2] for c in spy] == [True]


@pytest.mark.parametrize('ddof', [0, 1, 2, 5])
@pytest.mark.parametrize('name', ['std', 'sem'])
def test_ddof_matches_pandas(kx, name, ddof):
    df = make_df()[['f', 'n']]
    tab = kx.toq(df)
    assert_series_matches(getattr(tab, name)(ddof=ddof), getattr(df, name)(ddof=ddof))
    # Fewer values than ddof gives null
    short = kx.toq(df.head(3))
    assert_series_matches(getattr(short, name)(ddof=3), getattr(df.head(3), name)(ddof=3))


@pytest.mark.parametrize('name', ['skew', 'kurt'])
def test_skew_kurt_edge_cases(kx, name):
    # Too few values is null
    df = pd.DataFrame({'few': [1.0, 2.0, np.nan], 'ok': [1.0, 2.0, 7.0]})
    tab = kx.toq(df)
    assert_series_matches(getattr(tab, name)(), getattr(df, name)())
    df = pd.DataFrame({'ok': [1.0, 2.0, 7.0, 9.0, 20.0], 'few': [1.0, 2.0, 3.0, np.nan, np.nan]})
    assert_series_matches(getattr(kx.toq(df), name)(), getattr(df, name)())


@pytest.mark.parametrize('name', ['skew', 'kurt'])
def test_skew_kurt_no_variance_is_null(kx, spy, name):
    # pandas 3.1 returns NaN here (pandas GH 62864) and earlier pandas returns 0, so the
    # result is checked directly rather than against the installed pandas
    tab = kx.q('([] k:`a`a`a`a`a`b`b`b`b`b; c:10#2f; n:2 2 0n 2 2 1 2 7 9 20f; j:10#3)')
    res = getattr(tab, name)(numeric_only=True).py()
    assert all(np.isnan(res[c]) for c in ['c', 'j'])
    grouped = kx.q.value(getattr(tab.groupby('k'), name)()).py()
    assert np.isnan(grouped['n'][0]) and not np.isnan(grouped['n'][1])
    assert all(np.isnan(v) for v in grouped['c'] + grouped['j'])
    assert [c[2] for c in spy] == [True, True]


@pytest.mark.parametrize('name', ['skew', 'kurt'])
def test_skew_kurt_no_variance_is_null_on_fallback(kx, monkeypatch, name):
    import pykx.pandas_api as pandas_api
    import pykx.pandas_api.pandas_meta as pandas_meta
    monkeypatch.setattr(pandas_meta, 'native_agg', lambda *a, **k: None)
    monkeypatch.setattr(pandas_api, 'native_groupby_agg', lambda *a, **k: None)
    tab = kx.q('([] k:`a`a`a`a`a`b`b`b`b`b; c:10#2f; n:2 2 0n 2 2 1 2 7 9 20f; j:10#3)')
    res = getattr(tab, name)(numeric_only=True).py()
    assert all(np.isnan(res[c]) for c in ['c', 'j'])
    grouped = kx.q.value(getattr(tab.groupby('k'), name)()).py()
    assert np.isnan(grouped['n'][0]) and not np.isnan(grouped['n'][1])


@pytest.mark.parametrize('name', ['idxmax', 'idxmin'])
def test_idxmax_idxmin_keyed_table_uses_key_labels(kx, name):
    df = make_df()[['n', 'f']].drop_duplicates('n').reset_index(drop=True)
    df['n'] = df['n'] * 10
    tab = kx.q('1!', kx.toq(df))
    assert_series_matches(getattr(tab, name)(), getattr(df.set_index('n'), name)())
    assert_series_matches(getattr(tab, name)(skipna=False),
                          getattr(df.set_index('n'), name)())


def test_count_string_columns(kx):
    tab = kx.q('([] s:("ab";enlist "c";"";enlist "d"))')
    assert tab.count().py() == {'s': 4}
    assert kx.q('([] s:enlist "ab")').count().py() == {'s': 1}
    assert kx.q('0#([] s:("ab";"c"))').count().py() == {'s': 0}


@pytest.mark.parametrize('name', ['sum', 'prod'])
def test_skipna_false_with_min_count(kx, name):
    tab = kx.q('([] f:1 0n 2f; s:`a``b)')
    res = getattr(tab, name)(skipna=False, min_count=1, numeric_only=True).py()
    assert pd.isna(res['f'])
    assert tab[['s']].sum(skipna=False).py() == {'s': ''}
    # A column of strings does not raise on the null check
    getattr(kx.q('([] s:enlist "ab")'), name)(skipna=False)


@pytest.mark.parametrize('name, kwargs', [
    ('sum', {'min_count': 1}),
    ('prod', {'min_count': 1}),
    ('skew', {'skipna': False}),
    ('idxmax', {}),
])
def test_no_columns_left_after_filter(kx, name, kwargs):
    # min_count, skipna=False on skew and a table with two keys use the generic path
    tab = kx.q('([k1:`a`b; k2:1 2] s:`x`y)')
    res = getattr(tab, name)(numeric_only=True, **kwargs)
    assert kx.q('~', res, kx.q('(`symbol$())!()'))


@pytest.mark.parametrize('name, best', [('idxmax', 2), ('idxmin', 0)])
def test_idxmax_idxmin_all_null_column(kx, name, best):
    tab = kx.q('([] a:3#0n; b:1 2 3f)')
    res = getattr(tab, name)().py()
    assert res['b'] == best
    assert res['a'] is None or res['a'] is pd.NA


def float_inf_df():
    return pd.DataFrame({'f': [1.0, np.inf, -np.inf, 4.0], 'g': [1.0, np.inf, 3.0, 4.0]})


@pytest.mark.parametrize('name', ['sum', 'prod', 'min', 'max', 'mean', 'median', 'count',
                                  'any', 'all', 'idxmax', 'idxmin', 'skew', 'kurt'])
def test_infinities_match_pandas(kx, name):
    df = float_inf_df()
    tab = kx.toq(df)
    assert_series_matches(getattr(tab, name)(), getattr(df, name)())


@pytest.mark.parametrize('name', ['sum', 'prod', 'min', 'max'])
def test_infinities_skipna_false_match_pandas(kx, name):
    df = pd.concat([float_inf_df(), pd.DataFrame({'f': [np.nan], 'g': [np.nan]})],
                   ignore_index=True)
    tab = kx.toq(df)
    assert_series_matches(getattr(tab, name)(skipna=False), getattr(df, name)(skipna=False))


@pytest.mark.parametrize('name', ['sum', 'prod', 'min', 'max', 'mean', 'median', 'std', 'sem',
                                  'skew', 'kurt', 'count', 'any', 'all', 'idxmax', 'idxmin'])
def test_empty_table_does_not_raise(kx, name):
    plain = kx.q('0#([] a:1 2; f:1 2f)')
    assert len(getattr(plain, name)().py()) == len(plain.columns)
    # all and any on a KeyedTable return a single boolean
    keyed = kx.q('0#1!([] k:`a`b; v:1 2)')
    res = getattr(keyed, name)()
    if name not in ('all', 'any'):
        assert len(res.py()) == len(keyed.columns)


def test_empty_table_values(kx):
    tab = kx.q('0#([] a:1 2; f:1 2f)')
    assert tab.sum().py() == {'a': 0, 'f': 0.0}
    assert tab.prod().py() == {'a': 1, 'f': 1.0}
    assert tab.count().py() == {'a': 0, 'f': 0}
    assert tab.any().py() == {'a': False, 'f': False}
    assert tab.all().py() == {'a': True, 'f': True}
    assert all(pd.isna(v) for v in tab.mean().py().values())


def test_all_null_columns(kx):
    tab = kx.q('([] f:3#0n; n:3#0N)')
    assert tab.sum().py() == {'f': 0.0, 'n': 0}
    assert tab.count().py() == {'f': 0, 'n': 0}
    assert tab.any().py() == {'f': False, 'n': False}
    assert tab.all().py() == {'f': True, 'n': True}
    assert all(pd.isna(v) for v in tab.mean().py().values())
    assert all(pd.isna(v) for v in tab.std().py().values())
    assert all(pd.isna(v) for v in tab.idxmax().py().values())


@pytest.mark.parametrize('vec', ['1 0N 3 0N 5', '1 0N 3 0N', '1.5 0n 3 0n 5', '(),1', '0#0'])
def test_count_switches_to_long_sum_above_int_limit(kx, vec):
    from pykx.pandas_api.pandas_native_agg import _COUNT
    # A real vector of 0Wi elements will not fit in a test runner's memory so
    # the limit is lowered to exercise both branches of the same expression.
    limited = _COUNT.replace('0Wi', '3i')
    assert limited != _COUNT
    expected = kx.q(f'{{sum not null x}} {vec}').py()
    for expr in (_COUNT, limited):
        res = kx.q(f'{expr} {vec}')
        assert isinstance(res, kx.LongAtom)
        assert res.py() == expected


@pytest.mark.parametrize('q_table', [
    '([] k:`a`a`b`b; f:1 0n 3 4f; n:1 2 3 4)',
    '0#([] a:1 2; f:1 2f)',
    '([] f:3#0n; n:3#0N)',
    '1!([] k:`a`b; v:1 2)',
])
def test_count_returns_long(kx, q_table):
    res = kx.q(q_table).count()
    assert len(res) > 0
    assert all(isinstance(v, kx.LongAtom) for v in res.values())


@pytest.mark.parametrize('by', ['k', ['k', 'k2']])
def test_grouped_count_dtype_matches_pandas(kx, by):
    df = make_df()
    expected = df.groupby(by).count()
    res = kx.toq(df).groupby(by).count().pd()
    assert (expected.dtypes == np.int64).all()
    assert (res.dtypes == np.int64).all()


@pytest.mark.parametrize('name', ['sum', 'prod', 'min', 'max', 'mean', 'median', 'std', 'count'])
@pytest.mark.parametrize('keys', [1, 2])
def test_keyed_table_aggregates_value_columns(kx, name, keys):
    df = make_df()[['k', 'k2', 'f', 'n']].drop_duplicates(['k', 'k2']).reset_index(drop=True)
    if keys == 1:
        df = df.drop(columns='k2').drop_duplicates('k').reset_index(drop=True)
    index = ['k'] if keys == 1 else ['k', 'k2']
    tab = kx.q(f'{keys}!', kx.toq(df))
    assert_series_matches(getattr(tab, name)(), getattr(df.set_index(index), name)())


@pytest.mark.parametrize('name', ['sum', 'min', 'max', 'mean', 'count', 'idxmax', 'skew'])
def test_grouped_numeric_only_is_native_and_matches_pandas(kx, spy, name):
    df = make_df()[['k', 'f', 'n', 'b']]
    tab = kx.toq(df)
    res = getattr(tab.groupby('k'), name)(numeric_only=True).pd()
    # pandas has no numeric_only for a grouped count and every column here is numeric
    kwargs = {} if name == 'count' else {'numeric_only': True}
    expected = getattr(df.groupby('k'), name)(**kwargs)
    pd.testing.assert_frame_equal(res, expected, check_dtype=False, check_exact=False)
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True]


@pytest.mark.parametrize('expr', [
    "([] a:(\"hello\";\"there\"))",
    "([] s:`a`b`c)",
    "([] l:(1 2;3 4 5;6))",
])
def test_unsupported_column_types_are_not_lowered(kx, spy, expr):
    tab = kx.q(expr)
    names = ['sum', 'prod', 'min', 'max', 'mean', 'median', 'std', 'sem', 'skew', 'kurt',
             'idxmax', 'idxmin']
    if '`' in expr:
        # min, max, idxmax and idxmin take symbols natively
        names = [n for n in names if n not in ('min', 'max', 'idxmax', 'idxmin')]
    for name in names:
        try:
            getattr(tab, name)()
        except kx.QError:
            pass
    assert not any(c[2] for c in spy)


def guid_df(rows=6, seed=3):
    import uuid
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'g': [uuid.UUID(int=int(x)) for x in rng.integers(1, 2**62, rows)],
                         'v': rng.integers(0, 10, rows)})


@pytest.mark.parametrize('name', ['min', 'max', 'idxmax', 'idxmin'])
def test_guid_ordering_aggregations_match_pandas(kx, spy, name):
    df = guid_df()
    tab = kx.toq(df)
    res = getattr(tab, name)().py()
    expected = getattr(df, name)()
    assert str(res['g']) == str(expected['g'])
    assert res['v'] == expected['v']
    assert all(c[2] for c in spy)


def test_guid_nulls_are_skipped(kx):
    tab = kx.q('([] g:0Ng,3?0Ng)')
    values = sorted(tab['g'].py()[1:])
    assert tab.min().py()['g'] == values[0]
    assert tab.max().py()['g'] == values[-1]
    assert tab.count().py() == {'g': 3}
    assert tab.min(skipna=False).py()['g'] is None or str(tab.min(skipna=False).py()['g']) == \
        '00000000-0000-0000-0000-000000000000'
    all_null = kx.q('([] g:3#0Ng)')
    assert all_null.idxmax().py()['g'] is None or pd.isna(all_null.idxmax().py()['g'])
    assert all_null.count().py() == {'g': 0}


@pytest.mark.parametrize('name', ['count', 'any', 'all'])
def test_guid_count_any_all(kx, name):
    df = guid_df()
    assert getattr(kx.toq(df)[['g']], name)().py() == getattr(df[['g']], name)().to_dict()


@pytest.mark.parametrize('name', ['sum', 'prod', 'mean', 'median', 'std', 'sem', 'skew', 'kurt'])
def test_guid_arithmetic_aggregations_raise(kx, name):
    # Arithmetic on GUIDs is a q type error, as it is a TypeError in pandas
    tab = kx.toq(guid_df()[['g']])
    with pytest.raises(kx.QError):
        getattr(tab, name)()


@pytest.mark.parametrize('name', ['min', 'max', 'count'])
def test_guid_values_grouped(kx, spy, name):
    import uuid
    rng = np.random.default_rng(5)
    df = pd.DataFrame({'k': ['a', 'b', 'a', 'b', 'a'],
                       'g': [uuid.UUID(int=int(x)) for x in rng.integers(1, 2**62, 5)]})
    res = getattr(kx.toq(df).groupby('k'), name)().pd()
    expected = getattr(df.groupby('k'), name)()
    assert [str(x) for x in res['g']] == [str(x) for x in expected['g']]
    assert [c[2] for c in spy if c[0] == 'native_groupby_agg'] == [True]


def test_guid_group_key_and_keyed_guid_index(kx):
    import uuid
    a, b = uuid.UUID(int=7), uuid.UUID(int=9)
    df = pd.DataFrame({'g': [a, b, a, b], 'v': [1, 2, 3, 4]})
    res = kx.toq(df).groupby('g')['v'].sum().pd()
    expected = df.groupby('g')[['v']].sum()
    assert [str(x) for x in res.index] == [str(x) for x in expected.index]
    assert list(res['v']) == list(expected['v'])
    keyed = kx.q('1!', kx.toq(pd.DataFrame({'k': [a, b], 'v': [5, 6]})))
    assert str(keyed.idxmax().py()['v']) == str(b)

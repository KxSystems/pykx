"""Tests for the Pandas API agg functionality"""

import pytest

import statistics


def test_single_func(q, kx):
    tab = q('([] til 10; 1)')
    gtab = q('([]sym:`a`a`a`b`b;x:1 2 2 3 3)')
    df = tab.pd()
    gdf = gtab.pd()

    def mode(x):
        return statistics.mode(x)

    min_tab = tab.agg('min')
    assert all(min_tab == tab.min().values())
    mode_tab = tab.agg(mode)
    mode_df = df.agg(mode)
    assert all(kx.toq(mode_df) == mode_tab.values())

    min_gtab = gtab.groupby('sym').agg('min')
    min_df = gdf.groupby('sym').agg('min')
    assert q('{x~y}', min_gtab, min_df)
    mode_gtab = gtab.groupby('sym').agg(mode)
    mode_gdf = gdf.groupby('sym').agg(mode)
    assert q('{x~y}', mode_gtab, mode_gdf)


def test_list_funcs(q, kx):
    tab = q('([] til 10; 1)')
    gtab = q('([]sym:`a`a`a`b`b;x:1 2 2 3 3)')
    df = tab.pd()
    gdf = gtab.pd()

    def mode(x):
        return statistics.mode(x)

    lst_tab = tab.agg(['min', mode])
    lst_df = df.agg(['min', mode])
    assert q('{key[x][`function]~key[y][`0]}', lst_tab, lst_df)
    assert q('{value[x]~value[y]}', lst_tab, lst_df)

    lst_gtab = gtab.groupby('sym').agg(['min', mode])
    lst_gdf = gdf.groupby('sym').agg(['min', mode])
    assert q('{x~y}', lst_gdf['x']['min'], lst_gtab['min'].values()['x'])
    assert q('{x~y}', lst_gdf['x']['mode'], lst_gtab['mode'].values()['x'])


def test_dict_funcs(q, kx):
    tab = q('([] til 10; 1)')
    dict_str = tab.agg({'x': 'min', 'x1': 'max'})
    max_ret = q('(enlist enlist[`function]!enlist[`max])!enlist enlist[`x1]!enlist 1')
    min_ret = q('(enlist enlist[`function]!enlist[`min])!enlist enlist[`x]!enlist 0')
    dict_str_max = kx.q.qsql.select(dict_str['x1'], where=['function=`max'])
    dict_str_min = kx.q.qsql.select(dict_str['x'], where=['function=`min'])
    assert isinstance(dict_str, kx.KeyedTable)
    assert q('{x~y}', dict_str_max, max_ret)
    assert q('{x~y}', dict_str_min, min_ret)

    def mode(x):
        return statistics.mode(x)

    dict_mode = tab.agg({'x': 'min', 'x1': mode})
    mode_ret = q('(enlist enlist[`function]!enlist[`mode])!enlist enlist[`x1]!enlist 1')
    min_ret = q('(enlist enlist[`function]!enlist[`min])!enlist enlist[`x]!enlist 0')
    dict_func_mode = kx.q.qsql.select(dict_mode['x1'], where=['function=`mode'])
    dict_func_min = kx.q.qsql.select(dict_mode['x'], where=['function=`min'])
    assert isinstance(dict_mode, kx.KeyedTable)
    assert q('{x~y}', dict_func_mode, mode_ret)
    assert q('{x~y}', dict_func_min, min_ret)


def test_errors(q, kx):
    tab = q('([]til 10;1)')
    ktab = q('([til 10] til 10; 1)')
    gtab = q('([]sym:`a`a`a`b`b;x:1 2 2 3 3)').groupby('sym')

    with pytest.raises(NotImplementedError) as err:
        ktab.agg('min')
    assert "KeyedTable" in str(err.value)

    with pytest.raises(NotImplementedError) as err:
        tab.agg('min', axis=1)
    assert 'axis parameter only presently supported' in str(err.value)

    with pytest.raises(NotImplementedError) as err:
        gtab.agg({'x': 'min'})
    assert 'Dictionary input func not presently supported for GroupbyTable' in str(err.value)

    with pytest.raises(NotImplementedError) as err:
        tab.agg({'x': ['min', 'max']})
    assert "Unsupported type '<class 'list'>' supplied as dictionary value" in str(err.value)

    with pytest.raises(kx.QError) as err:
        q('0#([]10?1f;10?1f)').agg('mean')
    assert "Application of 'agg' method not supported" in str(err.value)

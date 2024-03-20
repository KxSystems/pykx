# Do not import pykx here - use the `kx` fixture instead!

import pandas as pd
import pytest

import itertools

data = pd.DataFrame({'population': [59000000, 65000000, 59000000,
                                    434000, 337000, 337000, 337000,
                                    11300, 11300, 65000000, 59000000],
                     'GDP': [1937894, 2583560, 12011, 4520, 12128,
                             17036, 182, 38, 311, 666, 667],
                     'alpha-2': ["IT", "FR", "MT", "MV", "BN",
                                 "IS", "NR", "TV", "AI", "DV", "HJ"],
                     'name': ["Italy", "France", "Malta",
                              "Maldives", "Brunei", "Iceland",
                              "Nauru", "Tuvalu", "Anguilla", "Ireland", "Wales"]})
df = data.set_index('name')

df_two_column_index = data.set_index(['name', 'alpha-2'])

df_no_index = data


def test_sorting(q, kx):
    tab = q('([] column_a:100 20 3; column_b:42 15 56; column_c:8 80 45)')

    assert (tab.sort_values(by='column_b').pd().values == (
            tab.pd().sort_values(by='column_b').values)).all()

    assert (tab.sort_values(by='column_c', ascending=False).pd().values == (
            tab.pd().sort_values(by='column_c', ascending=False).values)).all()

    tab = kx.Table(data={'column_a': [2, 3, 2, 2, 1], 'column_b': [56, 15, 42, 102, 32],
                         'column_c': [45, 80, 8, 61, 87]})

    assert (tab.sort_values(by=['column_a', 'column_b']).pd().values == (
            tab.pd().sort_values(by=['column_a', 'column_b']).values)).all()

    with pytest.raises(ValueError) as err:
        kx.toq(df).sort_values(by='population', ascending='test')
        assert 'expected type bool' in str(err.value)


def test_nlargest(q, kx):
    tab = q('([] column_a:100 20 3 6 8; column_b:42 15 56 102 32; column_c:8 80 45 87 61)')

    assert (tab.nlargest(3, 'column_c').pd().values == (
            tab.pd().nlargest(3, 'column_c').values)).all()

    assert (tab.nlargest(0, 'column_a').pd().values == (
            tab.pd().nlargest(0, 'column_a').values)).all()

    assert((kx.toq(df).nlargest(4, 'population')==df.nlargest(4, 'population')).all())

    assert((kx.toq(df).nlargest(4, 'population', keep='first')==df.nlargest(
        4, 'population', keep='first')).all())

    assert((kx.toq(df).nlargest(4, 'population', keep='last')==df.nlargest(
        4, 'population', keep='last')).all())

    assert((kx.toq(df).nlargest(4, 'population', keep='all')==df.nlargest(
        4, 'population', keep='all')).all())

    assert((kx.toq(df.reset_index()).nlargest(4, 'population', keep='last')==df.reset_index(
    ).nlargest(4, 'population', keep='last')).all())

    assert len(df.nlargest(0, 'population', keep='all')) == 0

    with pytest.raises(ValueError) as err:
        kx.toq(df).nlargest(4, 'population', keep='test')
        assert 'keep must be' in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.toq(df).nlargest('x', 'population', keep='all')
        assert 'numeric values' in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.toq(df).nlargest(4, 0, keep='all')
        assert 'columns must be of type' in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.toq(df).nlargest(4, ['population', 0], keep='all')
        assert 'columns must be of type' in str(err.value)


def test_nsmallest(q, kx):
    tab = q('([] column_a:100 20 3 6 8; column_b:42 15 56 102 32; column_c:8 80 45 87 61)')

    assert (tab.nsmallest(2, 'column_a').pd().values == (
            tab.pd().nsmallest(2, 'column_a').values)).all()

    assert((kx.toq(df).nsmallest(4, 'population')==df.nsmallest(4, 'population')).all())

    assert((kx.toq(df).nsmallest(4, 'population', keep='first')==df.nsmallest(
        4, 'population', keep='first')).all())

    assert((kx.toq(df).nsmallest(4, 'population', keep='last')==df.nsmallest(
        4, 'population', keep='last')).all())

    assert((kx.toq(df).nlargest(4, 'population', keep='all')==df.nlargest(
        4, 'population', keep='all')).all())

    assert((kx.toq(df.reset_index()).nsmallest(4, 'population', keep='last')==df.reset_index(
    ).nsmallest(4, 'population', keep='last')).all())

    assert len(df.nsmallest(0, 'population', keep='all')) == 0

    with pytest.raises(ValueError) as err:
        kx.toq(df).nsmallest(4, 'population', keep='test')
        assert 'keep must be' in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.toq(df).nsmallest('x', 'population', keep='all')
        assert 'numeric values' in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.toq(df).nsmallest(4, 0, keep='all')
        assert 'columns must be of type' in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.toq(df).nsmallest(4, ['population', 0], keep='all')
        assert 'columns must be of type' in str(err.value)


def test_itertests(q, kx):
    df_list = [df, df_two_column_index]
    n_list = [1, 2, 3, 4, 5, 6]
    col_list = ['population', 'GDP']
    keepList = ['first', 'last', 'all']
    listOfLists = [df_list, n_list, col_list, keepList]
    listOfParams = [n_list, col_list, keepList]
    combo_list = list(itertools.product(*listOfLists))
    combo_list_no_index = list(itertools.product(*listOfParams))

    for i in combo_list:
        assert all((i[0].nsmallest(i[1], columns=i[2], keep=i[3]) == kx.toq(i[0]).nsmallest(
            i[1], columns=i[2], keep=i[3]).pd()))
        assert all((i[0].nlargest(i[1], columns=i[2], keep=i[3]) == kx.toq(i[0]).nlargest(
            i[1], columns=i[2], keep=i[3]).pd()))

    for i in combo_list_no_index:
        assert all(df_no_index.nlargest(*i).reset_index(drop=True) == kx.toq(
            df_no_index).nlargest(*i).pd())
        assert all(df_no_index.nsmallest(*i).reset_index(drop=True) == kx.toq(
            df_no_index).nsmallest(*i).pd())

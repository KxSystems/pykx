"""Tests for the Pandas API apply functionality"""

import os

import pytest


def _count(x):
    try:
        return len(x)
    except TypeError:
        return 1


def _multi_arg_count(x, y=0):
    try:
        count = len(x)
    except TypeError:
        count = 1
    return count + y


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_map_applymap(kx):
    tables = [kx.q('([]100?(1 2 3;"123";1 2;0n);100?1f;100?(0n;1f))'),
              kx.q('([til 100]x1:100?(1 2 3;"123";1 2;0n);x2:100?1f;x3:100?(0n;1f))')]
    for tab in tables:
        fn = kx.q('count')
        assert kx.q('~', tab.map(fn), tab.applymap(fn))

        fn = lambda x: len(str(x)) # noqa: E731
        assert kx.q('~', tab.map(fn), tab.applymap(fn))

        assert kx.q('~', tab.map(_count), tab.applymap(_count))
        assert kx.q('~', tab.map(_multi_arg_count), tab.applymap(_multi_arg_count))
        assert kx.q('~', tab.map(_multi_arg_count, y=2), tab.applymap(_multi_arg_count, y=2))
        assert not kx.q('~', tab.map(_multi_arg_count), tab.applymap(_multi_arg_count, y=2))

        assert not tab.map(fn).has_nulls
        assert tab.map(fn, na_action='ignore').has_nulls
        assert kx.q('~', tab.map(fn, na_action='ignore'), tab.applymap(fn, na_action='ignore'))

        fn = lambda x, y: y + len(str(x)) # noqa: E731
        assert kx.q('~', tab.map(fn, y=1), tab.applymap(fn, y=1))
        assert not kx.q('~', tab.map(fn, y=1), tab.applymap(fn, y=2))

        assert kx.q('~', tab.map(_count), tab.pd().applymap(_count))
        assert kx.q('~', tab.map(_multi_arg_count), tab.pd().applymap(_multi_arg_count))
        assert kx.q('~', tab.map(_multi_arg_count, y=1), tab.pd().applymap(_multi_arg_count, y=1))
        ignore_check = kx.q('=',
                            tab.map(_multi_arg_count, na_action='ignore', y=1),
                            tab.pd().applymap(_multi_arg_count, na_action='ignore', y=1))
        if isinstance(tab, kx.KeyedTable):
            ignore_check = ignore_check._values
        assert ignore_check.all().all()

        with pytest.raises(TypeError) as err:
            tab.map(_count, na_action=False)
        assert "na_action must be None or 'ignore'" in str(err.value)

        with pytest.raises(TypeError) as err:
            tab.map(1)
        assert "Provided value 'func' is not callable" in str(err.value)

        with pytest.raises(kx.QError) as err:
            tab.map(kx.q('{[x;y]x+y}'), y=2)
        assert "ERROR: Passing key" in str(err.value)

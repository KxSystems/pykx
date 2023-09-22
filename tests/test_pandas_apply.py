"""Tests for the Pandas API apply functionality"""

import numpy as np
import pytest


def test_q_add_axis_0_col_1(q, kx):
    tab = q('([] til 10)')
    add_data = tab.apply(q('{x+1}'))
    assert isinstance(add_data, kx.Table)
    assert all(add_data.keys() == ['x'])
    assert q('{all raze x}', add_data.values() == q('enlist 1+til 10'))


def test_q_sum_axis_0_col_1(q, kx):
    tab = q('([] til 10)')
    sum_data = tab.apply(q('sum'))
    assert isinstance(sum_data, kx.Dictionary)
    assert all(sum_data.keys() == ['x'])
    assert all(sum_data.values() == [45])


def test_q_add_axis_1_col_1(q, kx):
    tab = q('([] til 10)')
    add_data = tab.apply(q('{x+1}'), axis=1)

    def add_1(x):
        return(x+1)

    assert isinstance(add_data, kx.Table)
    assert all(add_data == kx.toq(tab.pd().apply(add_1, axis=1)))


def test_sum_axis_1_col_1(q, kx):
    tab = q('([] til 10)')
    sum_data = tab.apply(q('sum'), axis=1)
    assert isinstance(sum_data, kx.LongVector)
    assert all(sum_data == q('til 10'))


def test_q_add_axis_0_col_2(q, kx):
    tab = q('([] til 10; 1)')
    add_data = tab.apply(q('{x+1}'))
    assert isinstance(add_data, kx.Table)
    assert all(add_data.keys() == ['x', 'x1'])
    assert q('{all raze x}', add_data.values() == q('(1+til 10;10#2)'))


def test_q_sum_axis_0_col_2(q, kx):
    tab = q('([] til 10; 1)')
    sum_data = tab.apply(q('sum'))
    assert isinstance(sum_data, kx.Dictionary)
    assert all(sum_data.keys() == ['x', 'x1'])
    assert all(sum_data.values() == [45, 10])


def test_q_add_axis_1_col_2(q, kx):
    tab = q('([] til 10; 1)')
    add_data = tab.apply(q('{x+1}'), axis=1)

    def add_1(x):
        return(x+1)

    assert isinstance(add_data, kx.Table)
    assert all(add_data == kx.toq(tab.pd().apply(add_1, axis=1)))


def test_sum_axis_1_col_2(q, kx):
    tab = q('([] til 10; 1)')
    sum_data = tab.apply(q('sum'), axis=1)
    assert isinstance(sum_data, kx.LongVector)
    assert all(sum_data == q('1+til 10'))


def test_py_add_axis_0_cols_1(q, kx):
    tab = q('([] til 10)')

    def add_1(x):
        return(x+1)

    add_data = tab.apply(add_1)
    assert isinstance(add_data, kx.Table)
    assert all(add_data.keys() == ['x'])
    assert all(add_data == kx.toq(tab.pd().apply(add_1)))


def test_py_add_axis_0_cols_2(q, kx):
    tab = q('([] til 10; 1)')

    def add_1(x):
        return(x+1)

    add_data = tab.apply(add_1)
    assert isinstance(add_data, kx.Table)
    assert all(add_data.keys() == ['x', 'x1'])
    assert q('{all raze x}', add_data.values() == q('(1+til 10;10#2)'))


def test_py_add_axis_1_cols_1(q, kx):
    tab = q('([] til 10)')

    def add_1(x):
        return(x+1)

    add_data = tab.apply(add_1, axis=1)
    assert isinstance(add_data, kx.Table)
    assert all(add_data == kx.toq(tab.pd().apply(add_1, axis=1)))


def test_py_add_axis_1_cols_2(q, kx):
    tab = q('([] til 10; 1)')

    def add_1(x):
        return(x+1)

    add_data = tab.apply(add_1, axis=1)
    assert isinstance(add_data, kx.Table)
    assert all(add_data.keys() == ['x', 'x1'])
    assert q('{all raze x}', add_data.values() == q('(1+til 10;10#2)'))


def test_py_sum_axis_0_cols_1(q, kx):
    tab = q('([] til 10)')
    sum_data = tab.apply(np.sum)
    assert isinstance(sum_data, kx.Dictionary)
    assert all(sum_data.keys() == ['x'])
    assert all(sum_data.values() == [45])
    assert all(sum_data == kx.toq(tab.pd().apply(np.sum)))


def test_py_sum_axis_1_cols_1(q, kx):
    tab = q('([] til 10)')
    sum_data = tab.apply(np.sum, axis=1)
    assert isinstance(sum_data, kx.LongVector)
    assert all(sum_data == q('til 10'))
    assert all(sum_data == kx.toq(tab.pd().apply(np.sum, axis=1)))


def test_py_sum_axis_0_cols_2(q, kx):
    tab = q('([] til 10; 1)')
    sum_data = tab.apply(np.sum)
    assert isinstance(sum_data, kx.Dictionary)
    assert all(sum_data.keys() == ['x', 'x1'])
    assert all(sum_data.values() == [45, 10])
    assert all(sum_data == kx.toq(tab.pd().apply(np.sum)))


def test_py_sum_axis_1_cols_2(q, kx):
    tab = q('([] til 10; 1)')
    sum_data = tab.apply(np.sum, axis=1)
    assert isinstance(sum_data, kx.LongVector)
    assert all(sum_data == q('1+til 10'))
    assert all(sum_data == kx.toq(tab.pd().apply(np.sum, axis=1)))


def test_py_args(q, kx):
    tab = q('([] til 10; 1)')

    def add_value(x, param0=0):
        return(x+param0)

    sum_data = tab.apply(add_value, param0=1)
    assert isinstance(sum_data, kx.Table)
    assert q('{all raze x}', sum_data == q('([]1+til 10;2)'))


def test_q_args(q, kx):
    tab = q('([] til 10; 1)')
    sum_data = tab.apply(q('{x+y}'), 1)
    assert isinstance(sum_data, kx.Table)
    assert q('{all raze x}', sum_data == q('([]1+til 10;2)'))
    with pytest.raises(kx.QError):
        sum_data = tab.apply(q('{x+y}'), y=1)


def test_error_callable(q):
    tab = q('([] til 10; 1)')
    with pytest.raises(RuntimeError) as errinfo:
        tab.apply(1)
    assert "Provided value 'func' is not callable" in str(errinfo)


def test_error_result_type(q):
    tab = q('([] til 10; 1)')
    with pytest.raises(NotImplementedError) as errinfo:
        tab.apply(q('{x+1}'), result_type='broadcast')
    assert "'result_type' parameter not implemented, please set to None" in str(errinfo)


def test_error_raw(q):
    tab = q('([] til 10; 1)')
    with pytest.raises(NotImplementedError) as errinfo:
        tab.apply(q('{x+1}'), raw=True)
    assert "'raw' parameter not implemented, please set to None" in str(errinfo)

"""Tests for the Pandas API set_index."""
import pytest


def test_set_index_single(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    assert df.set_index('x').pd().equals(df.pd().set_index('x'))


def test_set_index_multi(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    assert df.set_index(['x', 'y']).pd().equals(df.pd().set_index(['x', 'y']))


# Duplicate columns names will break .pd()
# toq drops the key columns so only can test values
# def test_set_index_drop(q):
#     df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
#     kxi = df.set_index('x', drop=False)
#     pdi = df.pd().set_index('x', drop=False)
#     assert q('{value[x]~y}', kxi, pdi)


def test_set_index_drop(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    with pytest.raises(NotImplementedError):
        df.set_index('x', drop=False)


def test_set_index_append(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    kxi = df.set_index('x').set_index('y', append=True).pd()
    pdi = df.pd().set_index('x').set_index('y', append=True)
    assert kxi.equals(pdi)


def test_index_unkeyed(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    assert q('{x~til count y}', df.index, df)


def test_index_keyed(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    assert q('{x~z#y}', df.set_index('y').index, df, ['y'])


def test_index_verify(q):
    kxi = q('([] a:1 2 3; b:3 4 5)')
    kxi.set_index('a', verify_integrity=True)


def test_index_verify_fail(kx, q):
    kxi = q('([] a:1 1 3; b:3 4 5)')
    with pytest.raises(kx.exceptions.QError):
        kxi.set_index('a', verify_integrity=True)


def test_index_verify_no_fail(kx, q):
    kxi = q('([] a:1 1 3; b:3 4 5)')
    kxi.set_index('a', verify_integrity=False)


def test_index_verify_no_fail_default(q):
    kxi = q('([] a:1 1 3; b:3 4 5)')
    kxi.set_index('a')


def test_index_verufy_multi(q):
    kxi = q('([] a:1 1 3;b:1 2 3; c:3 4 5)')
    kxi.set_index(['a', 'b'], verify_integrity=True)


def test_index_verify_multi_fail(kx, q):
    kxi = q('([] a:1 1 3;b:1 1 3; c:3 4 5)')
    with pytest.raises(kx.exceptions.QError):
        kxi.set_index(['a', 'b'], verify_integrity=True)

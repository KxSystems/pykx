
"""Tests for the Pandas API set_index."""
import pytest


def test_reset_index_single(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    assert q('~', df, df.set_index('x').reset_index())


def test_reset_index_multi(q):
    df = q('([] x: til 10; y: 10 - til 10; z: 10?`a`b`c)')
    assert q('~', df, df.set_index(['x', 'y']).reset_index())


def test_reset_int(q):
    df = q('([x: til 10; y: 10 - til 10]z: 10?`a`b`c)')
    pddf = df.pd()
    assert q('~', df.reset_index(0), pddf.reset_index(0))
    assert q('~', df.reset_index(1), pddf.reset_index(1))
    assert q('~', df.reset_index([0, 1]), pddf.reset_index([0, 1]))


def test_reset_str(q):
    df = q('([x: til 10; y: 10 - til 10]z: 10?`a`b`c)')
    pddf = df.pd()
    assert q('~', df.reset_index('x'), pddf.reset_index('x'))
    assert q('~', df.reset_index('y'), pddf.reset_index('y'))
    assert q('~', df.reset_index(['x', 'y']), pddf.reset_index(['x', 'y']))


def test_reset_drop(q):
    df = q('([x: til 10; y: 10 - til 10]z: 10?`a`b`c)')
    pddf = df.pd()
    assert q('~', df.reset_index('x', drop=True), pddf.reset_index('x', drop=True))
    assert q('~', df.reset_index('y', drop=True), pddf.reset_index('y', drop=True))
    assert q('~', df.reset_index(['x', 'y'], drop=True), pddf.reset_index(['x', 'y'], drop=True))


def test_reset_duplicates(kx, q):
    df = q('([til 10;10?1f];10?1f;10?1f)')
    assert q('~', df.reset_index(allow_duplicates=True), q('0!', df))
    with pytest.raises(kx.QError) as err:
        df.reset_index()
    assert 'Cannot reset index' in str(err)


def test_reset_errors(kx, q):
    df = q('([til 10;10?1f];10?1f;10?1f)')
    with pytest.raises(kx.QError) as err:
        df.reset_index(col_level=1)
    assert "'col_level' not presently" in str(err)
    with pytest.raises(kx.QError) as err:
        df.reset_index(col_fill=1)
    assert "'col_fill' not presently" in str(err)
    with pytest.raises(kx.QError) as err:
        df.reset_index(names=['a', 'b'])
    assert "'names' not presently" in str(err)
    with pytest.raises(TypeError) as err:
        df.reset_index(levels=1.0, allow_duplicates=True)
    assert "Unsupported type provided for 'levels'" in str(err)
    with pytest.raises(kx.QError) as err:
        df.reset_index('missing_col', allow_duplicates=True)
    assert "Key(s) missing_col not found" in str(err)
    with pytest.raises(kx.QError) as err:
        df.reset_index(10, allow_duplicates=True)
    assert 'out of range' in str(err)

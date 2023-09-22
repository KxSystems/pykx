import pickle
from time import sleep
from uuid import uuid4

import pytest


@pytest.mark.unlicensed
def test_cached_property(kx):
    class A:
        @kx.util.cached_property
        def lazy(self):
            return uuid4()
    a = A()
    x = a.lazy
    assert x == a.lazy
    assert A.lazy.attrname == 'lazy'


@pytest.mark.unlicensed
def test_subclasses(kx):
    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    class D(B, C):
        pass

    class E(D):
        pass

    assert {A, B, C, D, E} == kx.util.subclasses(A)
    assert {E} == kx.util.subclasses(E)
    assert {C, D, E} == kx.util.subclasses(C)


@pytest.mark.unlicensed
def test_normalize_to_bytes(kx):
    assert b'' == kx.util.normalize_to_bytes('')
    assert b'Sphinx of black quartz, judge my vow.' ==\
        kx.util.normalize_to_bytes('Sphinx of black quartz, judge my vow.')
    assert b'' == kx.util.normalize_to_bytes(b'')
    assert b'Sphinx of black quartz, judge my vow.' ==\
        kx.util.normalize_to_bytes(b'Sphinx of black quartz, judge my vow.')
    with pytest.raises(TypeError) as ex:
        kx.util.normalize_to_bytes(None, name='Nothing')
    assert str(ex.value).startswith('Nothing')


@pytest.mark.unlicensed
def test_normalize_to_str(kx):
    assert '' == kx.util.normalize_to_str('')
    assert 'Sphinx of black quartz, judge my vow.' ==\
        kx.util.normalize_to_str('Sphinx of black quartz, judge my vow.')
    assert '' == kx.util.normalize_to_str(b'')
    assert 'Sphinx of black quartz, judge my vow.' ==\
        kx.util.normalize_to_str(b'Sphinx of black quartz, judge my vow.')
    with pytest.raises(TypeError) as ex:
        kx.util.normalize_to_str(None, name='Nothing')
    assert str(ex.value).startswith('Nothing')


@pytest.mark.unlicensed
def test_attr_as(kx):
    class AttrHaver:
        attr = True

    attr_haver = AttrHaver()
    assert attr_haver.attr
    with kx.util.attr_as(attr_haver, 'attr', False):
        assert not attr_haver.attr
    assert attr_haver.attr

    class AttrLacker:
        pass

    attr_lacker = AttrLacker()
    assert not hasattr(attr_lacker, 'attr')
    with kx.util.attr_as(attr_lacker, 'attr', None):
        assert hasattr(attr_lacker, 'attr')
    assert not hasattr(attr_lacker, 'attr')


@pytest.mark.unlicensed
def test_once(kx):
    from time import time_ns

    @kx.util.once
    def time_when_first_run():
        return time_ns()
    t = time_when_first_run()
    sleep(0.01)
    assert t == time_when_first_run()
    sleep(0.01)
    assert t == time_when_first_run()


@pytest.mark.ipc
def test_pickle_pykx_df_block_manager(q):
    """Test that a Pandas DataFrame created by PyKX can be deserialized without PyKX.

    DataFrames that originate from PyKX have a custom block manager. We have to take care to
    serialize it as a regular block manager so that it can be deserialized without PyKX installed.
    """
    df = q('([] date:9?.z.D; id:9?9; time:9?.z.N; bs:9?9; bp:9?9f; ap:9?9f; as:9?9)').pd()
    serialized = pickle.dumps(df)
    assert b'pykx' not in serialized
    # `df._data` is the block manager
    assert type(df._data).__name__.encode() not in serialized


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.util), list)
    assert sorted(dir(kx.util)) == dir(kx.util)


@pytest.mark.unlicensed
def test_debug_environment(kx):
    assert kx.util.debug_environment() is None


@pytest.mark.unlicensed
def test_debug_environment_ret(kx):
    assert isinstance(kx.util.debug_environment(return_info=True), str)

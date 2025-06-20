from pathlib import Path
import os
import pickle
import shutil
from time import sleep
from uuid import uuid4

import pytest
import toml


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


def test_debug_licensed(kx):
    ret = kx.util.debug_environment(return_info=True).split('\n')
    passed = False
    for i in ret:
        if i == 'pykx.licensed: True':
            passed = True
    assert passed


@pytest.mark.unlicensed
def test_install_q(kx):
    base_path = Path(os.path.expanduser('~'))
    folder = base_path / 'qfolder'
    config_path = base_path / '.pykx-config'
    assert not os.path.isdir(folder)
    kx.util.install_q(folder)
    assert os.path.isfile(config_path)
    with open(config_path, 'r') as file:
        data = toml.load(file)
    assert ['PYKX_Q_EXECUTABLE', 'QHOME'] == list(data['default'].keys())
    assert os.path.isdir(folder)
    assert os.path.isfile(folder / 'q.k')
    shutil.rmtree(str(folder))
    os.remove(str(base_path / '.pykx-config'))


def test_detect_bad_columns(kx):
    dup_col = kx.q('flip `a`a`a`b!4 4#16?1f')
    with pytest.warns(RuntimeWarning) as w:
        assert kx.util.detect_bad_columns(dup_col)
    assert "Duplicate columns: ['a']" in w[0].message.args[0]
    assert "Invalid columns" not in w[0].message.args[0]
    assert ['a'] == kx.util.detect_bad_columns(dup_col, return_cols=True)
    html_repr = dup_col._repr_html_()
    assert isinstance(html_repr, str)
    assert "pykx.Table" in html_repr

    invalid_col = kx.q('flip (`a;`b;`c;`$"a b")!4 4#16?1f')
    with pytest.warns(RuntimeWarning) as w:
        assert kx.util.detect_bad_columns(invalid_col)
    assert "Duplicate columns:" not in w[0].message.args[0]
    assert "Invalid columns: ['a b']" in w[0].message.args[0]
    assert ['a b'] == kx.util.detect_bad_columns(invalid_col, return_cols=True)
    html_repr = invalid_col._repr_html_()
    assert isinstance(html_repr, str)
    assert "pykx.Table" in html_repr

    dup_invalid_cols = kx.q('flip (`a;`a;`a;`b;`$"a b")!5 5#25?1f')
    with pytest.warns(RuntimeWarning) as w:
        assert kx.util.detect_bad_columns(dup_invalid_cols)
    assert "Duplicate columns: ['a']" in w[0].message.args[0]
    assert "Invalid columns: ['a b']" in w[0].message.args[0]
    assert ['a', 'a b'] == kx.util.detect_bad_columns(dup_invalid_cols, return_cols=True)
    html_repr = dup_invalid_cols._repr_html_()
    assert isinstance(html_repr, str)
    assert "pykx.Table" in html_repr

    for i in [dup_col, invalid_col, dup_invalid_cols]:
        t = i.set_index(1)
        with pytest.warns(RuntimeWarning) as w:
            assert kx.util.detect_bad_columns(t)
        assert "Duplicate columns or columns with" in w[0].message.args[0]
        html_repr = t._repr_html_()
        assert isinstance(html_repr, str)
        assert "pykx.KeyedTable" in html_repr

    tab = kx.q('{x set flip (`a;`$"a b")!2 10#20?1f;get x}`:multiColSplay/')
    with pytest.warns(RuntimeWarning) as w:
        assert kx.util.detect_bad_columns(tab)
    assert "Duplicate columns:" not in w[0].message.args[0]
    assert "Invalid columns: ['a b']" in w[0].message.args[0]
    assert ['a b'] == kx.util.detect_bad_columns(tab, return_cols=True)
    html_repr = tab._repr_html_()
    assert isinstance(html_repr, str)
    assert "pykx.Splay" in html_repr

    os.makedirs('HDB', exist_ok=True)
    os.chdir('HDB')
    kx.q('(`$":2001.01.01/partTab/") set flip(`a;`$"a b")!2 10#20?1f')
    kx.q('(`$":2001.01.02/partTab/") set flip(`a;`$"a b")!2 10#20?1f')
    kx.q('system"l ."')
    ptab = kx.q['partTab']
    with pytest.warns(RuntimeWarning) as w:
        assert kx.util.detect_bad_columns(ptab)
    assert "Duplicate columns:" not in w[0].message.args[0]
    assert "Invalid columns: ['a b']" in w[0].message.args[0]
    assert ['a b'] == kx.util.detect_bad_columns(ptab, return_cols=True)
    html_repr = ptab._repr_html_()
    assert isinstance(html_repr, str)
    assert "pykx.Part" in html_repr
    os.chdir('..')


def test_config_add_type(kx):
    fpath = Path(os.path.expanduser('~')) / '.pykx-config'
    with open(fpath, 'w') as f:
        f.write('[default]\n')

    kx.util.add_to_config({'PYKX_GC': 'True', 'PYKX_MAX_ERROR_LENGTH': 1,
                           'PYKX_BETA_FEATURES': True})

    with open(fpath, "r") as f:
        data = toml.load(f)
    assert data['default']['PYKX_GC'] == 'True'
    assert data['default']['PYKX_MAX_ERROR_LENGTH'] == 1
    assert data['default']['PYKX_BETA_FEATURES']
    os.remove(fpath)

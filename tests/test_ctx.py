from contextlib import contextmanager
import os
from pathlib import Path
import re
from tempfile import gettempdir

# Do not import pykx here - use the `kx` fixture instead!
import pytest


q_script_content = '''
lambda:{"this is a test lambda"};
data:10?10;
q:1b;
'''


k_script_content = '''
lambda:{"this is a test lambda"};
data:10?10;
q:0b;
'''


@pytest.fixture
def q_script_with_k_script_present(tmp_path):
    q_path = tmp_path/'script.q'
    k_path = tmp_path/'script.k'
    q_path.write_text(q_script_content)
    k_path.write_text(k_script_content)
    return q_path


@contextmanager
def cd(newdir):
    """Change the current working directory within the context."""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


@pytest.mark.ipc
def test_dir(q, kx):
    assert isinstance(dir(kx.ctx), list)
    assert sorted(dir(kx.ctx)) == dir(kx.ctx)
    default_contexts = ('h', 'j', 'o', 'Q', 'q', 'z')
    assert all(x in dir(q) for x in default_contexts)


def test_register_without_args(q):
    with pytest.raises(ValueError):
        q._register()


def test_register_by_path(q, q_script_with_k_script_present, kx):
    q_script = q_script_with_k_script_present
    q_script.rename(q_script.parent/'nameA.q')
    try:
        q.script
    except Exception as err:
        assert isinstance(err, AttributeError)
        assert isinstance(err.__cause__, FileNotFoundError)
    q._register(name='script', path=q_script.parent/'nameA.q')
    assert isinstance(q.script, kx.ctx.QContext)
    q._register(path=q_script.parent/'nameA.q')
    assert isinstance(q.nameA, kx.ctx.QContext)


def test_local_register_by_name_q(q, q_script_with_k_script_present):
    tmp_dir = q_script_with_k_script_present.resolve().parent
    with cd(str(tmp_dir)):
        # find local q file (even when the k file also exists)
        assert q.script.q


def test_find_namespace_in_q_file(q, tmp_path):
    with pytest.raises(AttributeError):
        q.testnamespace
    with cd(tmp_path):
        with open('testnamespace.q', 'w') as f:
            f.write('.testnamespace.true:1b\n')
        assert q.testnamespace.true


def test_reserved_in_ctx(q, tmp_path):
    with pytest.raises(AttributeError):
        q.testnamespace
    with cd(tmp_path):
        with open('name.q', 'w') as f:
            f.write('.name.test.update:{x+1}\n')
        assert q.name.test.update(1) == 2


def test_python_keyword_as_q_fn(q):
    assert q.except_ == q('except')
    assert q._except == q('except')
    assert q._except_ == q('except')


@pytest.mark.ipc
def test_paths(q, kx):
    assert q.paths == [x.resolve(strict=True) for x in kx.ctx.default_paths]
    q.paths = [kx.qhome, str(gettempdir())]
    assert q.paths == [kx.qhome.resolve(strict=True), Path(gettempdir()).resolve(strict=True)]
    with pytest.warns(RuntimeWarning):
        q.paths = ('.', 'a_path_that_does_not_exist')
    q.paths = kx.ctx.default_paths
    assert q.paths == kx.ctx.default_paths


def test_namespace_switch(q):
    q('.globalvar:`dotglobal')
    assert q['.globalvar'].py() == q('.globalvar').py() == q.globalvar.py() == 'dotglobal'
    switch_q_ctx = q('{system "d ", string x}')
    prev_ctx = q('system "d"')
    assert prev_ctx.py() == '.'
    new_ctx = '.pykxtest'
    try:
        switch_q_ctx(new_ctx)
        assert q('system "d"').py() == new_ctx
        q('asdf:`qwerty')
        assert q['asdf'].py() == q('asdf').py() == q.pykxtest.asdf.py() == 'qwerty'
        assert q['.globalvar'].py() == q('.globalvar').py() == q.globalvar.py() == 'dotglobal'
    finally:
        switch_q_ctx(prev_ctx)


@pytest.mark.ipc
def test_iter(q):
    assert set() == {'Q', 'h', 'j', 'o', 'q', 'z'} - set(q.ctx)


@pytest.mark.ipc
def test_repr(q, kx):
    m = kx.ctx.QContext.__module__

    ctx = q.Q
    pattern = f'<{m}.QContext(.*?){ctx._fqn}(.*?)'r'\[(?:\w+(?:, )?)+\]>'
    assert re.match(pattern, repr(ctx))

    ctx = q.ctx
    pattern = f'<{m}.QContext(.*?){ctx._fqn}(.*?)'r'\[(?:\w+(?:, )?)+\]>'
    assert re.match(pattern, repr(ctx))


@pytest.mark.ipc
def test_update_global_context(q):
    with pytest.raises(AttributeError):
        q.will_be_defined_later
    q.will_be_defined_later = 'defined_global'
    assert q.will_be_defined_later.py() == 'defined_global'


@pytest.mark.ipc
def test_update_context(q):
    with pytest.raises(AttributeError):
        q.Q.will_be_defined_later
    q.Q.will_be_defined_later = 'defined_pykx'
    assert q.Q.will_be_defined_later.py() == 'defined_pykx'


@pytest.mark.ipc
def test_fqn(q):
    assert q.ctx._fqn == ''
    q('.Q.i.vvv:`something')
    assert q.Q.i._fqn == '.Q.i'


@pytest.mark.ipc(licensed_only=True)
def test_del(q, kx):
    q('.Q.i.vvv:`something')
    q.Q.i.willdelete = 321
    assert 'willdelete' in q.Q.i
    assert q.Q.i.willdelete == 321
    del q.Q.i.willdelete
    assert 'willdelete' not in q.Q.i
    with pytest.raises(kx.PyKXException):
        del q.Q
    with pytest.raises(kx.PyKXException):
        del q.ctx.Q


@pytest.mark.ipc
def test_dot_q_errors(q, kx):
    with pytest.raises(AttributeError) as err:
        q.select
    assert 'select' in str(err.value)
    with pytest.raises(AttributeError) as err:
        q.exec
    assert 'exec' in str(err.value)
    with pytest.raises(AttributeError) as err:
        q.update
    assert 'update' in str(err.value)
    with pytest.raises(AttributeError) as err:
        q.delete
    assert 'delete' in str(err.value)


@pytest.mark.ipc
def test_dot_z(q):
    assert q.z.i.py() == q('.z.i').py()
    with pytest.raises(AttributeError):
        q.z.s # special AttributeError with detailed message
    with pytest.raises(AttributeError):
        q.z.fake # normal AttributeError
    assert set() == {
        'D', 'H', 'K', 'N', 'P', 'T', 'W', 'X', 'a', 'ac', 'b', 'bm', 'c', 'd', 'e', 'exit', 'f',
        'h', 'i', 'k', 'l', 'n', 'o', 'p', 'pc', 'pd', 'pg', 'ph', 'pi', 'pm', 'po', 'pp', 'pq',
        'ps', 'pw', 'q', 't', 'u', 'vs', 'w', 'wc', 'wo', 'ws', 'x', 'zd'
    } - set(dir(q.z))


@pytest.mark.ipc
def test_expunge(kx, q):
    q.z.ps = q('2*')
    assert q.z.ps.py() is not None
    if kx.licensed:
        assert q.z.ps(2).py() == 4
    del q.z.ps
    assert q.z.ps.py() is None


@pytest.mark.ipc(licensed_only=True)
def test_reserved_words(q):
    assert q.abs == q.q.abs == q('abs')
    assert q.like == q.q.like == q('like')
    assert q.xexp == q.q.xexp == q('xexp')


def test_with_block(q):
    assert q('system"d"') == '.'
    with q.pykx:
        assert q('system"d"') == '.pykx'
        with q.Q:
            assert q('system"d"') == '.Q'
            with q.pykx:
                assert q('system"d"') == '.pykx'
            with q.ctx:
                assert q('system"d"') == '.'
            with q.q:
                assert q('system"d"') == '.q'
            assert q('system"d"') == '.Q'
        assert q('system"d"') == '.pykx'
    assert q('system"d"') == '.'


@pytest.mark.ipc
def test_with_block_errors_over_ipc(q, kx):
    if isinstance(q, kx.EmbeddedQ):
        assert q('system"d"').py() == '.'
        with q.Q:
            assert q('system"d"').py() == '.Q'
    elif isinstance(q, kx.QConnection):
        assert q('system"d"').py() == '.'
        with pytest.raises(kx.PyKXException):
            with q.Q:
                pass


@pytest.mark.unlicensed
def test_ctx_no_overwrite_qerror(q_port, kx):
    with kx.QConnection(port=q_port) as q:
        q('cons:()!()')
        q('.z.pw: {[u; p] $[all p="pass"; cons::cons,(enlist .z.w)!(enlist 1b); '
          'cons::cons,(enlist .z.w)!(enlist 0b)]; 1b}')
        q('.z.pg: {[f] $[cons[.z.w]; value f; \'"Access denied"]}')

    with kx.QConnection(port=q_port, username='a', password='pass') as q:
        assert list(range(5)) == q('til 5').py()

    with pytest.raises(AttributeError) as err:
        with kx.QConnection(port=q_port, username='a', password='aaaa') as q:
            q('type')
        assert 'Access Denied' in str(err.value)

from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Do not import pykx here - use the `kx` fixture instead!
import pytest


@contextmanager
def cd(newdir):
    """Change the current working directory within the context."""
    prevdir = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)


@pytest.mark.ipc
def test_getitem(q, kx):
    key = 'test_key'
    q(f'{key}:1234')
    assert q[key].py() == 1234
    assert q[key].t == -7

    key = 'hyphenated-table'
    q(f'set[`$"{key}";([]til 3;"abc")]')
    assert isinstance(q[key], kx.Table)
    assert q[key].values().py() == [[0, 1, 2], b'abc']


@pytest.mark.ipc
def test_setitem(q, kx):
    key = 'test_key'
    q[key] = 1234
    assert q(key).py() == 1234
    assert q(key).t == -7

    key = 'hyphenated-table'
    q[key] = q('([]til 3;"abc")')
    assert isinstance(q[key], kx.Table)
    assert q[key].values().py() == [[0, 1, 2], b'abc']

    with pytest.raises(kx.PyKXException):
        q['abs'] = 'abs' # check reserved word
    with pytest.raises(kx.PyKXException):
        q.abs = 'abs'
    with pytest.raises(kx.PyKXException):
        q.ctx['abs'] = 'abs'
    with pytest.raises(kx.PyKXException):
        q.ctx.abs = 'abs'
    with pytest.raises(kx.PyKXException):
        q['views'] = 'views' # check element of the .q namespace


@pytest.mark.ipc
def test_delitem(q, kx):
    key = 'test_key'
    q[key] = 1234
    assert q(key).py() == 1234
    del q[key]
    assert key not in q('key `.').py()
    with pytest.raises(KeyError):
        del q[key]

    key = '.dotsomething'
    q[key] = q('([]til 3;"abc")')
    assert isinstance(q[key], kx.Table)
    assert q[key].values().py() == [[0, 1, 2], b'abc']
    with pytest.raises(kx.PyKXException):
        del q[key]


@pytest.mark.ipc
def test_setattr(q, kx):
    with pytest.raises(kx.PyKXException):
        q.abs = 'abs' # check reserved word
    with pytest.raises(kx.PyKXException):
        q.views = 'views' # check element of the .q namespace


def test_QARGS(q):
    assert '--testflag' in q('`$.z.X')


@pytest.mark.ipc
def test_call_with_params(kx, q):
    assert q('7').py() == 7
    assert q('til', 7).py() == [0, 1, 2, 3, 4, 5, 6]
    if kx.licensed:
        assert q('{z x*y}', 2, 3, q('{"j"$xexp[x;x]}')).py() == 46656
    assert q('{[a;b;c;d] a+b+c+d}', 1, 2, 3, 4).py() == 10
    assert q('{[a;b;c;d;e] a+b+c+d+e}', 1, 2, 3, 4, 5).py() == 15
    assert q('{[a;b;c;d;e;f] a+b+c+d+e+f}', 1, 2, 3, 4, 5, 6).py() == 21
    assert q('{[a;b;c;d;e;f;g] a+b+c+d+e+f+g}', 1, 2, 3, 4, 5, 6, 7).py() == 28
    assert q('{[a;b;c;d;e;f;g;h] a+b+c+d+e+f+g+h}', 1, 2, 3, 4, 5, 6, 7, 8).py() == 36
    with pytest.raises(TypeError):
        q('{[a;b;c;d;e;f;g;h] a+b+c+d+e+f+g+h}', 1, 2, 3, 4, 5, 6, 7, 8, 9)


@pytest.mark.isolate
def test_import_from_other_dir():
    with TemporaryDirectory() as tmp_dir:
        with cd(tmp_dir):
            import pykx as kx
            assert kx.q('k)45~+/!10')


@pytest.mark.ipc
def test_attributes(q):
    with pytest.raises(AttributeError):
        q.fake_attribute


def test_get_q_singleton_from_class(kx, q):
    _q = kx.EmbeddedQ()
    assert isinstance(_q, kx.EmbeddedQ)
    assert _q is q
    with pytest.raises(TypeError, match='(?i)takes 1 positional argument but 2 were given'):
        kx.EmbeddedQ(object())
    with pytest.raises(TypeError, match="(?i)got an unexpected keyword argument 'arg'"):
        kx.EmbeddedQ(arg=object())


@pytest.mark.unlicensed(unlicensed_only=True)
def test_unlicensed_call(kx):
    with pytest.raises(kx.LicenseException, match=f"(?i)run q code via '{kx.q!r}'"):
        kx.q('{[xn] xn-((xn*xn)-2)%2*xn}\\[1.5]')


@pytest.mark.ipc
def test_call_sync(q):
    query = '{[xn] xn-((xn*xn)-2)%2*xn}\\[1.5]'
    a = q(query, wait=True).py()
    b = [1.5, 1.4166666666666667, 1.4142156862745099, 1.4142135623746899, 1.4142135623730951]
    assert a == b
    assert q(f'steps:{query};steps', wait=False).py() is None
    assert q('steps').py() == b


@pytest.mark.unlicensed
def test_repr(kx):
    assert repr(kx.q) == 'pykx.q'


@pytest.mark.large
@pytest.mark.ipc # >10 GiB of memory is required to run the IPC variants of this test
def test_large_vector(q):
    v = q('til 671088640') # 5 GiB of data
    assert q('sum', v).py() == 225179981032980480


def test_path_arguments(q):
    # KXI-30172: Projections of PyKX functions don't support Path
    a = q("{[f;x] f x}")(lambda x: x)(Path('test'))
    assert q('`:test') == a


@pytest.mark.ipc
def test_dir(kx, q):
    assert set() == ({
        'Q', 'aj', 'aj0', 'ajf', 'ajf0', 'all', 'and', 'any', 'asc', 'asof', 'attr', 'avgs',
        'ceiling', 'cols', 'console', 'count', 'cross', 'csv', 'ctx', 'cut', 'deltas', 'desc',
        'differ', 'distinct', 'dsave', 'each', 'ej', 'ema', 'eval', 'except', 'fby', 'fills',
        'first', 'fkeys', 'flip', 'floor', 'get', 'group', 'gtime', 'h', 'hclose', 'hcount',
        'hdel', 'hsym', 'iasc', 'idesc', 'ij', 'ijf', 'inter', 'inv', 'j', 'key', 'keys', 'lj',
        'ljf', 'load', 'lower', 'lsq', 'ltime', 'ltrim', 'mavg', 'maxs', 'mcount', 'md5', 'mdev',
        'med', 'meta', 'mins', 'mmax', 'mmin', 'mmu', 'mod', 'msum', 'neg', 'next', 'not', 'null',
        'o', 'or', 'over', 'parse', 'peach', 'pj', 'prds', 'prev', 'prior', 'q', 'qsql', 'query',
        'rand', 'rank', 'ratios', 'raze', 'read0', 'read1', 'reciprocal', 'reserved_words',
        'reval', 'reverse', 'rload', 'rotate', 'rsave', 'rtrim', 'save', 'scan', 'scov', 'sdev',
        'set', 'show', 'signum', 'sql', 'ssr', 'string', 'sublist', 'sums', 'sv', 'svar', 'system',
        'tables', 'til', 'trim', 'type', 'uj', 'ujf', 'ungroup', 'union', 'upper', 'upsert',
        'value', 'view', 'views', 'vs', 'where', 'wj', 'wj1', 'ww', 'xasc', 'xbar', 'xcol',
        'xcols', 'xdesc', 'xgroup', 'xkey', 'xlog', 'xprev', 'xrank', 'z'
    } - set(dir(q)))
    assert isinstance(dir(kx.embedded_q), list)
    assert sorted(dir(kx.embedded_q)) == dir(kx.embedded_q)

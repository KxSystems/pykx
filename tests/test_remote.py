import os

# Do not import pykx here - use the `kx` fixture instead!
import pytest


@pytest.mark.isolate
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_session_create_clear(q_port):
    import pykx as kx
    session = kx.remote.session(port=q_port)
    assert isinstance(session._session, kx.SyncQConnection)
    assert session._session('1b')
    session.close()
    with pytest.raises(RuntimeError) as err:
        session._session('1b')
    assert 'Attempted to use a closed IPC connection' in str(err.value)


@pytest.mark.isolate
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_remote_exec(q_port):
    import pykx as kx
    session = kx.remote.session(port=q_port)

    @kx.remote.function(session)
    def func(x):
        return x+1
    assert kx.q('2') == func(1)


@pytest.mark.isolate
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_remote_library_exec(q_port):
    import pykx as kx
    session = kx.remote.session(port=q_port, libraries={'kx': 'pykx'})

    @kx.remote.function(session)
    def pykx_func(x, y):
        return kx.q.til(x) + y # noqa: F821
    assert all(kx.q('5+til 5') == pykx_func(5, 5))

    @kx.remote.function(session)
    def zero_arg():
        return 10
    assert kx.q('10') == zero_arg()


@pytest.mark.isolate
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_exec_failures(q_port):
    import pykx as kx

    @kx.remote.function(10)
    def test_func(x):
        return x+1
    with pytest.raises(Exception) as err:
        test_func(1)
    assert 'Supplied remote_session instance must be' in str(err.value)

    session = kx.remote.session(port=q_port)

    @kx.remote.function(session)
    def test_func(x):
        return numpy.array([x.py()]) # noqa: F821
    with pytest.raises(kx.QError) as err:
        test_func(10)
    assert "name 'numpy' is not defined" in str(err.value)

    with pytest.raises(kx.QError) as err:
        session.libraries({'un': 'undefined'})
    assert "Failed to load library 'undefined' with alias 'un'" in str(err.value)

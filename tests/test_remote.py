# Do not import pykx here - use the `kx` fixture instead!
import pytest


def test_session_create_clear(kx, q_port):
    session = kx.remote.session()
    assert session._session is None
    session.create(port=q_port)
    assert isinstance(session._session, kx.SyncQConnection)
    session.clear()
    assert session._session is None


def test_library_add_clear(kx, q_port):
    session = kx.remote.session()
    session.create(port=q_port)
    assert session._libraries == []
    session.add_library('numpy', 'pandas')
    assert session._libraries == ['numpy', 'pandas']
    session.clear()
    assert session._libraries == []


def test_session_errors(kx, q_port):
    session = kx.remote.session()
    with pytest.raises(Exception) as err:
        session.add_library('numpy')
    assert 'Unable to add packages in the absence' in str(err.value)
    session.create(port=q_port)
    with pytest.raises(Exception) as err:
        session.create(port=q_port)
    assert 'Active session in progress' in str(err.value)


@pytest.mark.xfail(reason="KXI-36200", strict=False)
@pytest.mark.unlicensed
def test_remote_exec(kx, q_port):
    session = kx.remote.session()
    session.create(port=q_port)

    @kx.remote.function(session)
    def func(x):
        return x+1
    assert kx.q('2') == func(1)


@pytest.mark.xfail(reason="KXI-36200", strict=False)
@pytest.mark.unlicensed
def test_remote_library_exec(kx, q_port):
    session = kx.remote.session()
    session.create(port=q_port)
    session.add_library('pykx')

    @kx.remote.function(session)
    def pykx_func(x, y):
        return pykx.q.til(x) + y # noqa: F821
    assert kx.q('5+til 5') == pykx_func(5, 5)


@pytest.mark.xfail(reason="KXI-36200", strict=False)
@pytest.mark.unlicensed
def test_exec_failures(kx, q_port):
    @kx.remote.function(10)
    def test_func(x):
        return x+1
    with pytest.raises(Exception) as err:
        test_func(1)
    assert 'Supplied remote_session instance must be' in str(err.value)

    session = kx.remote.session()

    @kx.remote.function(session)
    def test_func(x):
        return x+1
    with pytest.raises(Exception) as err:
        test_func(2)
    assert "User session must be generated using the 'create_session'" in str(err.value)

    session = kx.remote.session()
    session.create(port=q_port)

    @kx.remote.function(session)
    def test_func(x):
        return numpy.array([x.py()]) # noqa: F821
    with pytest.raises(kx.exceptions.QError) as err:
        test_func(10)
    assert "name 'numpy' is not defined" in str(err.value)

    session.add_library('undefined')

    @kx.remote.function(session)
    def test_func(x):
        return x+1
    with pytest.raises(kx.exception.QError) as err:
        test_func(1)
    assert "Failed to load package: undefined" in str(err.value)

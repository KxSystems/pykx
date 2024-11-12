from platform import system
import subprocess
import time

# Do not import pykx here - use the `kx` fixture instead!
import pytest


def test_natural_error(q, kx):
    try:
        q('2 + "abc"')
    except kx.exceptions.QError as e:
        k_type_error = e
    assert isinstance(k_type_error, kx.QError)
    assert isinstance(k_type_error, kx.PyKXException)
    assert k_type_error.args[0].startswith('type')
    assert str(k_type_error).startswith('type')


@pytest.mark.unlicensed
def test_artificially_raised_error(kx):
    with pytest.raises(kx.QError):
        raise kx.QError
    with pytest.raises(kx.QError):
        raise kx.QError('error message')
    r = 'PyArrow could not be loaded. Check `pykx._pyarrow.import_attempt_output` for the reason.'
    with pytest.raises(kx.PyArrowUnavailable, match=r):
        raise kx.PyArrowUnavailable()


def test_updated_messages(kx):
    with pytest.raises(kx.QError) as err:
        kx.q('cos:{x+1}')
    assert 'assign: Cannot redefine a reserved' in str(err.value)
    with pytest.raises(kx.QError) as err:
        tab = kx.q('([k:0 1]a:1 2)')
        tab.insert([0, 3])
    assert 'insert: Cannot insert a record with an existing key' in str(err.value)
    with pytest.raises(kx.QError) as err:
        kx.q('3 2').sorted()
    assert 's-fail: Cannot set "sorted" attribute on an unsorted' in str(err.value)
    with pytest.raises(kx.QError) as err:
        kx.q('2 3 2').unique()
    assert 'u-fail: Failed to do one of the following' in str(err.value)

    # Attempts to run the following tests on Windows results in failures due to
    # security issues on public runners
    if system() == 'Windows':
        return None

    # Test that existing IPC messages are maintained
    with pytest.raises(kx.QError) as err:
        kx.SyncQConnection(port=1234)
    assert 'hop. OS reports: Connection refused' in str(err.value)

    q_exe_path = subprocess.run(['which', 'q'], stdout=subprocess.PIPE).stdout.decode().strip()
    with kx.PyKXReimport():
        proc = subprocess.Popen([q_exe_path, 'tests/test_files/pw.q', '-p', '15001'])
    time.sleep(2)
    with pytest.raises(kx.QError) as err:
        kx.SyncQConnection(port=15001)
    assert 'access: Failed to connect' in str(err.value)
    proc.kill()


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.exceptions), list)
    assert sorted(dir(kx.exceptions)) == dir(kx.exceptions)

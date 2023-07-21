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


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.exceptions), list)
    assert sorted(dir(kx.exceptions)) == dir(kx.exceptions)

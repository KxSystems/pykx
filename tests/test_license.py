import base64
from io import StringIO
import os
import shutil
import re

# Do not import pykx here - use the `kx` fixture instead!
import pytest

from unittest.mock import patch


def test_initialization_using_unlicensed_mode(tmp_path, q):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['QARGS'] = '--unlicensed'
    import pykx as kx
    assert 2 == kx.toq(2).py()


def test_fallback_to_unlicensed_mode_error(tmp_path):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['QARGS'] = '--licensed'
    # Can't use PyKXException here because we have to import PyKX after entering the with-block
    with pytest.raises(Exception, match='(?i)Failed to initialize embedded q'):
        import pykx # noqa: F401


def test_unlicensed_signup(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['N'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    import pykx as kx
    assert 1 == kx.toq(1).py()
    assert not kx.licensed


def test_invalid_lic_continue(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['F'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    try:
        import pykx as kx # noqa: F401
    except Exception as e:
        assert str(e) == 'Invalid input provided please try again'


def test_licensed_signup_no_file(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'n', '1', '/test/test.blah'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    try:
        import pykx as kx # noqa: F401
    except Exception as e:
        assert str(e) == "Download location provided /test/test.blah does not exist."


def test_licensed_signup_invalid_b64(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'n', '2', 'data:image/png;test'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    try:
        import pykx as kx # noqa: F401
    except Exception as e:
        err_msg = 'Invalid license copy provided, '\
                  'please ensure you have copied the license information correctly'
        assert str(e) == err_msg


def test_licensed_success_file(monkeypatch):
    qhome_path = os.environ['QHOME']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    inputs = iter(['Y', 'n', '1', qhome_path + '/kc.lic'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()


def test_licensed_success_b64(monkeypatch):
    qhome_path = os.environ['QHOME']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    with open(qhome_path + '/kc.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read())
    inputs = iter(['Y', 'n', '2', str(license_content)])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()


@pytest.mark.parametrize(
    argnames='QARGS',
    argvalues=[
        '--licensed --unlicensed',
        '--unlicensed --licensed',
        '--unlicensed -S 987654321 --licensed',
    ],
    ids=['A', 'B', 'C'],
)
def test_use_both_licensed_and_unlicensed_flags(QARGS):
    os.environ['QARGS'] = QARGS
    # Can't use PyKXException here because we have to import PyKX after entering the with-block
    with pytest.raises(Exception, match='(?i)mutually exclusive'):
        import pykx # noqa: F401


def test_env_combination():
    os.environ['QARGS'] = '--licensed'
    os.environ['PYKX_UNLICENSED'] = 'true'
    with pytest.raises(Exception, match="(?i)'licensed' and 'unlicensed' behaviour"):
        import pykx # noqa: F401


def test_check_license_invalid_file(kx):
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('/test/test.blah')
    assert 'Unable to locate license /test/test.blah for comparison\n' == test_out.getvalue()


def test_check_license_no_qlic(kx):
    err_msg = f'Unable to find an installed license: k4.lic at location: {str(kx.qhome)}.\n'\
              'Please consider installing your license again using pykx.util.install_license\n'
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('/test/test.blah', license_type='k4.lic')
    assert err_msg == test_out.getvalue()


def test_check_license_format(kx):
    try:
        kx.license.check('/test/location', format='UNSUPPORTED')
    except Exception as e:
        assert str(e) == 'Unsupported option provided for format parameter'


def test_check_license_success_file(kx):
    assert kx.license.check(os.environ['QHOME'] + '/kc.lic')


def test_check_license_success_b64(kx):
    with open(os.environ['QHOME'] + '/kc.lic', 'rb') as f:
        license = base64.encodebytes(f.read())
    assert kx.license.check(license, format='STRING')


@pytest.mark.xfail(reason="Manual testing works correctly, seems to be a persistance issue")
@pytest.mark.skipif('KDB_LICENSE_EXPIRED' not in os.environ,
                    reason='Test required KDB_LICENSE_EXPIRED environment variable to be set')
def test_exp_license(kx):
    exp_lic = os.environ['KDB_LICENSE_EXPIRED']
    lic_folder = '/tmp/license'
    os.makedirs(lic_folder, exist_ok=True)
    with open(lic_folder + '/k4.lic', 'wb') as binary_file:
        binary_file.write(base64.b64decode(exp_lic))
    qhome_loc = os.environ['QHOME']
    os.environ['QLIC'] = os.environ['QHOME'] = lic_folder
    pattern = re.compile('Your PyKX license has now.*')
    with patch('sys.stdout', new=StringIO()) as test_out:
        try:
            import pykx  # noqa: F401
        except Exception as e:
            assert str(e) == "EOF when reading a line"
    shutil.rmtree(lic_folder)
    os.environ['QLIC'] = os.environ['QHOME'] = qhome_loc
    assert pattern.match(test_out.getvalue())


def test_check_license_invalid(kx):
    pattern = re.compile("Supplied license information does not match.*")
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('test', format='STRING')
    assert pattern.match(test_out.getvalue())


def test_install_license_exists(kx):
    pattern = re.compile("Installed license: kc.lic at location:*")
    try:
        kx.license.install('test', format='STRING')
    except Exception as e:
        assert pattern.match(str(e))


def test_install_license_invalid_format(kx):
    try:
        kx.license.install('test', format='UNSUPPORTED')
    except Exception as e:
        assert str(e) == 'Unsupported option provided for format parameter'


def test_install_license_invalid_file(kx):
    pattern = re.compile("Download location provided*")
    try:
        kx.license.install('/test/location.lic', force=True)
    except Exception as e:
        assert pattern.match(str(e))

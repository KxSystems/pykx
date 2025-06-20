import base64
from io import StringIO
import os
import re

# Do not import pykx here - use the `kx` fixture instead!
import pytest

from unittest.mock import patch


def test_initialization_using_unlicensed_fallback(tmp_path, q):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    import pykx as kx
    assert 2 == kx.toq(2).py()


def test_initialization_using_unlicensed_mode(tmp_path, q):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['QARGS'] = '--unlicensed'
    import pykx as kx
    assert 2 == kx.toq(2).py()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_fallback_to_unlicensed_mode_error(tmp_path):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['QARGS'] = '--licensed'
    # Can't use PyKXException here because we have to import PyKX after entering the with-block
    with pytest.raises(Exception, match='(?i)Failed to initialize embedded q'):
        import pykx # noqa: F401


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_unlicensed_signup(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['N', 'N'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    import pykx as kx
    assert 1 == kx.toq(1).py()
    assert not kx.licensed


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_invalid_lic_continue(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['F'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == 'Invalid input provided please try again'


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_invalid_existing_lic_input(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'F'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == 'Invalid input provided please try again'


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_invalid_commercial_input(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'N', 'F'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == 'User provided option was not one of [1/2]'


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_signup_no_file(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'n', '1', 'n', '1', '/test/test.blah'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == "Download location provided /test/test.blah does not exist."


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_signup_invalid_b64(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'n', '1', 'n', '2', 'data:image/png;test'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == 'Invalid license copy provided, '\
                         'please ensure you have copied the license information correctly'


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_success_file(monkeypatch):
    qlic_path = os.environ['QLIC']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    inputs = iter(['Y', 'n', '1', 'n', '1', qlic_path + '/kc.lic'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_success_b64(monkeypatch):
    qlic_path = os.environ['QLIC']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    with open(qlic_path + '/kc.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read())
    inputs = iter(['Y', 'n', '1', 'n', '2', str(license_content)])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_invalid_licensed_available_type_input(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'Y', 'F'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == 'User provided option was not one of [1/2]'


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_invalid_licensed_available_method_input(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'Y', '2', 'F'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == 'User provided option was not one of [1/2]'


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_available_no_file(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    inputs = iter(['Y', 'Y', '1', '/test/test.blah'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert str(e) == "Download location provided /test/test.blah does not exist."


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_available(monkeypatch):
    qlic_path = os.environ['QLIC']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    inputs = iter(['Y', 'Y', '1', qlic_path + '/kc.lic'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_available_b64(monkeypatch):
    qlic_path = os.environ['QLIC']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    with open(qlic_path + '/kc.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read())
    inputs = iter(['Y', 'Y', '2', '1', str(license_content)])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_envvar_init():
    qlic_path = os.environ['QLIC']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    with open(qlic_path + '/kc.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read())
    os.environ['KDB_LICENSE_B64'] = license_content.decode('utf-8')

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
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


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_env_combination():
    os.environ['QARGS'] = '--licensed'
    os.environ['PYKX_UNLICENSED'] = 'true'
    with pytest.raises(Exception, match="(?i)'licensed' and 'unlicensed' behaviour"):
        import pykx # noqa: F401


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_check_license_invalid_file(kx):
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('/test/test.blah')
    assert 'Unable to locate license /test/test.blah for comparison\n' == test_out.getvalue()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_check_license_no_qlic(kx):
    err_msg = f'Unable to find an installed license: k4.lic at location: {str(kx.qlic)}.\n'\
              'Please consider installing your license again using pykx.license.install\n'
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('/test/test.blah', license_type='k4.lic')
    assert err_msg == test_out.getvalue()
    assert hasattr(kx.license, 'install')


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_check_license_format(kx):
    with pytest.raises(Exception) as e:
        kx.license.check('/test/location', format='UNSUPPORTED')
        assert str(e) == 'Unsupported option provided for format parameter'


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_check_license_success_file(kx):
    assert kx.license.check(os.environ['QLIC'] + '/kc.lic')


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_check_license_success_b64(kx):
    with open(os.environ['QLIC'] + '/kc.lic', 'rb') as f:
        license = base64.encodebytes(f.read())
    license = license.decode()
    license = license.replace('\n', '')
    license = bytes(license, 'utf-8')
    assert kx.license.check(license, format='STRING')


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_check_license_invalid(kx):
    pattern = re.compile("Supplied license information does not match.*")
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('test', format='STRING')
    assert pattern.match(test_out.getvalue())


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_install_license_exists(kx):
    pattern = re.compile("Installed license: kc.lic at location:*")
    with pytest.raises(Exception) as e:
        kx.license.install('test', format='STRING')
        assert pattern.match(str(e))


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_install_license_invalid_format(kx):
    with pytest.raises(Exception) as e:
        kx.license.install('test', format='UNSUPPORTED')
        assert str(e) == 'Unsupported option provided for format parameter'


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_install_license_invalid_file(kx):
    pattern = re.compile("Download location provided*")
    with pytest.raises(Exception) as e:
        kx.license.install('/test/location.lic', force=True)
        assert pattern.match(str(e))


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_string_conversions(kx):
    licFile = kx.qlic/kx.config.lic_type
    with open(licFile, 'rb') as f:
        lic_contents = base64.encodebytes(f.read()).decode('utf-8')
        lic_contents = lic_contents.replace('\n', '')

    assert kx.license.check(kx.qlic/kx.config.lic_type)
    assert kx.license.check(lic_contents, format='string')
    with pytest.raises(Exception) as err:
        kx.license.check(lic_contents, format='string', license_type='blah.lic')
        assert "License type" in str(err)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_install_from_directory(monkeypatch):
    qlic_path = os.environ['QLIC']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    inputs = iter(['Y', 'Y', '1', qlic_path])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert kx.licensed


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_install_from_directory_no_file(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    qlic_path = os.environ['QLIC']
    inputs = iter(['Y', 'Y', '1', qlic_path])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception) as err:
        import pykx as kx
        assert not kx.licensed
        assert str(err) == "No license detected in given directory."


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_license_invalid_lic_name(monkeypatch):
    qlic_path = os.environ['QLIC']
    os.unsetenv('QLIC')
    os.unsetenv('QHOME')
    inputs = iter(['Y', 'n', '1', 'n', '1', qlic_path + '/junk.lic'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    with pytest.raises(Exception) as e:
        import pykx as kx # noqa: F401
        assert "Supplied license file" in str(e)


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='KXI-63218 PYKX_THREADING licence path differs'
)
@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    'KDB_LICENSE_EXPIRED' not in os.environ,
    reason='Test required KDB_LICENSE_EXPIRED environment variable to be set'
)
# To run test manually first "export KDB_LICENSE_EXPIRED=`cat {expired_lic_location} | base64 -w 0`"
def test_expired_license(monkeypatch):
    exp_lic = os.environ['KDB_LICENSE_EXPIRED']
    lic_folder = '/tmp/license'
    os.makedirs(lic_folder, exist_ok=True)
    with open(lic_folder + '/k4.lic', 'wb') as binary_file:
        binary_file.write(base64.b64decode(exp_lic))
    os.environ['QLIC'] = os.environ['QHOME'] = lic_folder
    inputs = iter(['n', 'n'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert not kx.licensed


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_mode_licensed_true():
    os.environ['PYKX_LICENSED'] = 'True'
    lic_folder = '/tmp/license'
    os.makedirs(lic_folder, exist_ok=True)
    with open(lic_folder + '/k4.lic', 'w') as f:
        f.write("badfile")
    os.environ['QLIC'] = os.environ['QHOME'] = lic_folder

    with pytest.raises(Exception):
        import pykx as kx
        kx.licensed


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_mode_monkeypatch_licensed_true(monkeypatch):
    os.environ['PYKX_LICENSED'] = 'True'
    lic_folder = '/tmp/license'
    os.makedirs(lic_folder, exist_ok=True)
    with open(lic_folder + '/k4.lic', 'w') as f:
        f.write("badfile")
    os.environ['QLIC'] = os.environ['QHOME'] = lic_folder

    inputs = iter([])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    with pytest.raises(Exception):
        import pykx as kx
        kx.licensed


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='KXI-63218 PYKX_THREADING licence path differs'
)
@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_mode_monkeypatch_licensed_false(monkeypatch):
    os.environ['PYKX_LICENSED'] = 'False'
    lic_folder = '/tmp/license'
    os.makedirs(lic_folder, exist_ok=True)
    with open(lic_folder + '/k4.lic', 'w') as f:
        f.write("badfile")
    os.environ['QLIC'] = os.environ['QHOME'] = lic_folder

    inputs = iter(['n', 'n'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    import pykx as kx
    assert not kx.licensed


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='KXI-63218 PYKX_THREADING licence path differs'
)
@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_bad_license():
    lic_folder = '/tmp/license'
    os.makedirs(lic_folder, exist_ok=True)
    with open(lic_folder + '/k4.lic', 'w') as f:
        f.write("badfile")
    os.environ['QLIC'] = os.environ['QHOME'] = lic_folder

    import pykx as kx
    assert not kx.licensed

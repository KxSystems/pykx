import base64
from io import StringIO
import os
from pathlib import Path
import re
import sys

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
    # Can't use PyKXException here, we have to import KDB-X Python after entering the with-block
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
    inputs = iter(['Y', 'n', '1', 'n', '1', qlic_path])
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
    with open(qlic_path + '/k4.lic', 'rb') as f:
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
    inputs = iter(['Y', 'Y', '1', qlic_path + '/k4.lic'])
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
    with open(qlic_path + '/k4.lic', 'rb') as f:
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
    with open(qlic_path + '/k4.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read())
    os.environ['KDB_LICENSE_B64'] = license_content.decode('utf-8')

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
def test_licensed_available_k4_envvar(tmp_path, monkeypatch):
    with open(os.environ['QLIC'] + '/k4.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read()).decode('utf-8')
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    monkeypatch.delenv('KDB_LICENSE_B64', raising=False)
    os.environ['KDB_K4LICENSE_B64'] = license_content
    inputs = iter(['Y', 'Y', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()
    assert (tmp_path/'k4.lic').exists()
    assert not (tmp_path/'kc.lic').exists()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.getenv('QLIC', ''), 'kc.lic')),
    reason='Test requires a kc.lic license within QLIC to install as kc.lic'
)
def test_licensed_available_kc_envvar(tmp_path, monkeypatch):
    # A kc.lic source is needed here: q rejects k4 license content saved as kc.lic
    with open(os.environ['QLIC'] + '/kc.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read()).decode('utf-8')
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    monkeypatch.delenv('KDB_K4LICENSE_B64', raising=False)
    os.environ['KDB_LICENSE_B64'] = license_content
    inputs = iter(['Y', 'Y', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()
    assert (tmp_path/'kc.lic').exists()
    assert not (tmp_path/'k4.lic').exists()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.getenv('QLIC', ''), 'kc.lic')),
    reason='Test requires a kc.lic license within QLIC to install as kc.lic'
)
def test_licensed_available_envvar_precedence(tmp_path, monkeypatch):
    # KDB_LICENSE_B64 is checked before KDB_K4LICENSE_B64
    with open(os.environ['QLIC'] + '/kc.lic', 'rb') as f:
        kc_content = base64.encodebytes(f.read()).decode('utf-8')
    with open(os.environ['QLIC'] + '/k4.lic', 'rb') as f:
        k4_content = base64.encodebytes(f.read()).decode('utf-8')
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['KDB_LICENSE_B64'] = kc_content
    os.environ['KDB_K4LICENSE_B64'] = k4_content
    inputs = iter(['Y', 'Y', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    import pykx as kx
    assert kx.licensed
    assert [0, 1, 2, 3, 4] == kx.q.til(5).py()
    assert (tmp_path/'kc.lic').exists()
    assert not (tmp_path/'k4.lic').exists()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_available_envvar_absent(tmp_path, monkeypatch):
    # Without either environment variable set option [3] is not offered
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    monkeypatch.delenv('KDB_LICENSE_B64', raising=False)
    monkeypatch.delenv('KDB_K4LICENSE_B64', raising=False)
    inputs = iter(['Y', 'Y', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    with pytest.raises(Exception, match=r'User provided option was not one of \[1/2\]'):
        import pykx # noqa: F401


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
@pytest.mark.parametrize('value', ['', '  \n'])
def test_licensed_available_envvar_blank(tmp_path, monkeypatch, value):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['KDB_LICENSE_B64'] = value
    monkeypatch.delenv('KDB_K4LICENSE_B64', raising=False)
    inputs = iter(['Y', 'Y', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    with pytest.raises(Exception, match=r'User provided option was not one of \[1/2\]'):
        import pykx # noqa: F401
    assert not (tmp_path/'kc.lic').exists()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_licensed_available_blank_kc_envvar_falls_back_to_k4(tmp_path, monkeypatch):
    with open(os.environ['QLIC'] + '/k4.lic', 'rb') as f:
        license_content = base64.encodebytes(f.read()).decode('utf-8')
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['KDB_LICENSE_B64'] = ''
    os.environ['KDB_K4LICENSE_B64'] = license_content
    inputs = iter(['Y', 'Y', '3'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    import pykx as kx
    assert kx.licensed
    assert (tmp_path/'k4.lic').exists()
    assert not (tmp_path/'kc.lic').exists()


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
    # Can't use PyKXException here, we have to import KDB-X Python after entering the with-block
    with pytest.raises(Exception, match='(?i)mutually exclusive'):
        import pykx # noqa: F401


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_env_combination():
    os.environ['QARGS'] = '--licensed'
    os.environ['PYKX_UNLICENSED'] = 'true'
    with pytest.raises(Exception, match="(?i)'licensed' and 'unlicensed' behavior"):
        import pykx # noqa: F401


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_check_license_invalid_file(kx):
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('/test/test.blah')
    expected = f'Unable to locate license {Path("/test/test.blah")} for comparison\n'
    assert expected == test_out.getvalue()


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_check_license_no_qlic(kx):
    err_msg = f'Unable to find an installed license: kc.lic at location: {str(kx.qlic)}.\n'\
              'Please consider installing your license again using pykx.license.install\n'
    with patch('sys.stdout', new=StringIO()) as test_out:
        kx.license.check('/test/test.blah', license_type='kc.lic')
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
    assert kx.license.check(os.environ['QLIC'] + '/k4.lic')


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_check_license_success_b64(kx):
    with open(os.environ['QLIC'] + '/k4.lic', 'rb') as f:
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
    pattern = re.compile("Installed license: k4.lic at location:*")
    with pytest.raises(Exception) as e:
        kx.license.install('test', format='STRING', license_type='k4.lic')
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
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    import pykx as kx
    assert not kx.licensed
    with pytest.raises(Exception) as e:
        kx.q('1+1')
    assert "license is expired" in str(e)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.xfail(reason='Flaky test', strict=False)
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


@pytest.mark.xfail(reason='Flaky license initialization ordering', strict=False)
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


@pytest.mark.xfail(reason='Flaky license initialization ordering', strict=False)
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
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    import pykx as kx
    assert not kx.licensed
    with pytest.raises(Exception) as err:
        kx.q('1+1')
    assert "corrupt" in str(err.value)
    print(str(err.value))


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.xfail(reason='Flaky test', strict=False)
def test_bad_license():
    lic_folder = '/tmp/license'
    os.makedirs(lic_folder, exist_ok=True)
    with open(lic_folder + '/k4.lic', 'w') as f:
        f.write("badfile")
    os.environ['QLIC'] = os.environ['QHOME'] = lic_folder

    import pykx as kx
    assert not kx.licensed


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_unlicensed_reason_env_var():
    os.environ['PYKX_UNLICENSED'] = 'True'
    import pykx as kx
    assert not kx.licensed
    with pytest.raises(Exception) as err:
        kx.q('1+1')
    assert "PYKX_UNLICENSED" in str(err.value)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_unlicensed_reason_qargs():
    os.environ['QARGS'] = '--unlicensed'
    import pykx as kx
    assert not kx.licensed
    with pytest.raises(Exception) as err:
        kx.q('1+1')
    assert "--unlicensed" in str(err.value)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_unlicensed_reason_light_load():
    os.environ['PYKX_LIGHT_LOAD'] = 'true'
    import pykx as kx
    assert not kx.licensed
    with pytest.raises(Exception) as err:
        kx.q('1+1')
    assert "light_load" in str(err.value)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
def test_unlicensed_reason_no_license_found(tmp_path, monkeypatch):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())

    inputs = iter(['N', 'N'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)

    import pykx as kx
    assert not kx.licensed
    with pytest.raises(Exception) as e:
        kx.q('1+1')
    assert "No license was found" in str(e)


def _threading_disabled(kx):
    """PYKX_THREADING was asked for, unlicensed mode was reached, so it must have been turned off
    everywhere it is observable."""
    assert not kx.licensed
    assert not kx.config.pykx_threading
    assert not kx.core.pykx_threading_enabled
    assert 'PYKX_THREADING: False' in kx.util.debug_environment(return_info=True)
    # Unlicensed mode is still fully usable
    assert [1, 2] == kx.toq([1, 2]).py()


@pytest.mark.isolate
def test_threading_unlicensed_qargs(recwarn):
    os.environ['PYKX_THREADING'] = 'True'
    os.environ['QARGS'] = '--unlicensed'
    import pykx as kx
    _threading_disabled(kx)
    assert not [w for w in recwarn if 'PYKX_THREADING' in str(w.message)]


@pytest.mark.isolate
def test_threading_unlicensed_env_var():
    os.environ['PYKX_THREADING'] = 'True'
    os.environ['PYKX_UNLICENSED'] = 'True'
    import pykx as kx
    _threading_disabled(kx)


@pytest.mark.isolate
def test_threading_unlicensed_config_file(tmp_path):
    config = tmp_path/'config-pykx'
    config.write_text('[default]\nPYKX_UNLICENSED="true"\n')
    os.environ['PYKX_THREADING'] = 'True'
    os.environ['PYKX_CONFIGURATION_LOCATION'] = str(config)
    import pykx as kx
    _threading_disabled(kx)


@pytest.mark.isolate
def test_threading_light_load():
    os.environ['PYKX_THREADING'] = 'True'
    os.environ['PYKX_LIGHT_LOAD'] = 'True'
    import pykx as kx
    assert not kx.config.pykx_threading
    assert not kx.core.pykx_threading_enabled


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.isolate
def test_threading_no_license(tmp_path):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['PYKX_THREADING'] = 'True'
    import pykx as kx
    _threading_disabled(kx)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.isolate
def test_threading_corrupt_license(tmp_path):
    (tmp_path/'kc.lic').write_bytes(b'not a valid license')
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['PYKX_THREADING'] = 'True'
    import pykx as kx
    _threading_disabled(kx)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.isolate
def test_threading_no_license_interactive(tmp_path, monkeypatch):
    (tmp_path/'kc.lic').write_bytes(b'not a valid license')
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['PYKX_THREADING'] = 'True'
    inputs = iter(['N', 'N'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    monkeypatch.setattr(sys, 'ps1', '>>> ', raising=False)
    import pykx as kx
    _threading_disabled(kx)


@pytest.mark.skipif(
    os.getenv('SKIP_LIC_TESTS') is not None,
    reason='License tests are being skipped'
)
@pytest.mark.isolate
def test_threading_no_license_forced_licensed(tmp_path):
    os.environ['QLIC'] = os.environ['QHOME'] = str(tmp_path.absolute())
    os.environ['PYKX_THREADING'] = 'True'
    os.environ['QARGS'] = '--licensed'
    with pytest.raises(Exception, match='(?i)Failed to initialize embedded q'):
        import pykx # noqa: F401


@pytest.mark.isolate
def test_threading_licensed_still_enabled():
    os.environ['PYKX_THREADING'] = 'True'
    import pykx as kx
    assert kx.licensed
    assert kx.config.pykx_threading
    assert kx.core.pykx_threading_enabled
    assert 'PYKX_THREADING: True' in kx.util.debug_environment(return_info=True)
    kx.shutdown_thread()

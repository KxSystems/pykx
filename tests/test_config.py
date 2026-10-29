import os
from pathlib import Path
from platform import system
import shutil
import subprocess
from tempfile import TemporaryDirectory
import warnings

import pytest


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.config), list)
    assert sorted(dir(kx.config)) == dir(kx.config)


@pytest.mark.isolate
def test_missing_profile(capsys):
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        open('my-config-pykx', 'a').close()
        os.environ['PYKX_CONFIGURATION_LOCATION'] = str(Path(tmp_dir)/'my-config-pykx')
        import pykx as kx # noqa
    out, _ = capsys.readouterr()
    assert "Unable to locate specified 'PYKX_PROFILE': 'default' in file" in out
    del os.environ['PYKX_CONFIGURATION_LOCATION']


@pytest.mark.isolate
def test_boolean_config():
    config = '''
    [default]
    PYKX_SUPPRESS_WARNINGS = true
    PYKX_QDEBUG = 'true'
    '''
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        with open('my-config-pykx', 'w+') as f:
            f.writelines(config)
        os.environ['PYKX_CONFIGURATION_LOCATION'] = str(Path(tmp_dir)/'my-config-pykx')
        import pykx as kx
        assert kx.config.suppress_warnings
        assert kx.config.pykx_qdebug
        del os.environ['PYKX_CONFIGURATION_LOCATION']


@pytest.mark.isolate
def test_invalid_qlic():
    os.environ['QLIC'] = 'invalid'
    with pytest.warns() as warnings:
        import pykx as kx # noqa: F401
    warnings = [x for x in warnings if "pykx/config.py" in x.filename]
    assert len(warnings) == 1
    assert 'Configuration value QLIC set to non directory' in str(warnings[0].message)


@pytest.mark.isolate
def test_qargs_single():
    os.environ['QARGS'] = '-p 5050'
    with pytest.warns() as warnings:
        import pykx as kx
    warnings = [x for x in warnings if "pykx/config.py" in x.filename]
    assert len(warnings) == 1
    assert 'setting a port in this way' in str(warnings[0].message)
    assert 2 == kx.q('2').py()


@pytest.mark.isolate
def test_qargs_multi():
    os.environ['QARGS'] = '-p 5050 -t 1000'
    with pytest.warns() as warnings:
        import pykx as kx
    warnings = [x for x in warnings if "pykx/config.py" in x.filename]
    assert len(warnings) == 2
    assert 'setting a port in this way' in str(warnings[0].message)
    assert 'setting timers in this way' in str(warnings[1].message)
    assert 2 == kx.q('2').py()


@pytest.mark.isolate
def test_suppress_warnings(recwarn):
    os.environ['PYKX_SUPPRESS_WARNINGS'] = 'True'
    os.environ['QARGS'] = '-p 5050'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import numpy as np
        import pykx as kx
        np.max(kx.q.til(10))
    for i in w:
        message = str(i.message)
        assert 'setting a port in this way' not in message
        assert 'Attempting to call numpy' not in message


@pytest.mark.isolate
def test_PYKX_GC_UNLICENSED():
    os.environ['PYKX_GC']='True'
    os.environ['PYKX_UNLICENSED']='True'
    import pykx as kx # noqa: F401
    assert True


@pytest.mark.skipif(
    system() == 'Windows',
    reason='PYKX_THREADING is unsupported on Windows'
)
def test_under_q_threading_disabled(tmp_path):
    q_exe = shutil.which('q')
    if q_exe is None:
        pytest.skip('Unable to locate a q executable')
    script = tmp_path/'threading.q'
    script.write_text(
        '\\l pykx.q\n'
        '.pykx.pyexec"import pykx;print(\'THREADING=\'+str(pykx.config.pykx_threading))";\n'
        'exit 0;\n'
    )
    proc = subprocess.run(
        [q_exe, str(script)],
        env={**os.environ, 'PYKX_THREADING': '1'},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert 0 == proc.returncode, proc.stdout
    assert 'THREADING=False' in proc.stdout
    assert 'SIGSEGV' not in proc.stdout
    assert 'Warning' not in proc.stdout

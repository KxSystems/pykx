from pathlib import Path
import os
from tempfile import TemporaryDirectory
import warnings

import pytest


@pytest.mark.unlicensed
def test_QHOME(kx):
    assert isinstance(kx.config.qhome, Path)
    assert (kx.config.qhome/'q.k').exists()


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.config), list)
    assert sorted(dir(kx.config)) == dir(kx.config)


@pytest.mark.isolate
def test_missing_profile(capsys):
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        open('.pykx-config', 'a').close()
        import pykx as kx # noqa
    out, _ = capsys.readouterr()
    assert "Unable to locate specified 'PYKX_PROFILE': 'default' in file" in out


@pytest.mark.isolate
def test_boolean_config():
    config = '''
    [default]
    PYKX_SUPPRESS_WARNINGS = true
    PYKX_QDEBUG = 'true'
    '''
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        with open('.pykx-config', 'w+') as f:
            f.writelines(config)
        import pykx as kx
        assert kx.config.suppress_warnings
        assert kx.config.pykx_qdebug


@pytest.mark.isolate
def test_valid_qlic():
    os.environ['QLIC'] = 'invalid'
    with pytest.warns() as warnings:
        import pykx as kx
    assert len(warnings) == 1
    assert 'Configuration value QLIC set to non directory' in str(warnings[0].message)
    assert 2 == kx.q('2').py()


@pytest.mark.isolate
def test_qargs_single():
    os.environ['QARGS'] = '-p 5050'
    with pytest.warns() as warnings:
        import pykx as kx
    assert len(warnings) == 1
    assert 'setting a port in this way' in str(warnings[0].message)
    assert 2 == kx.q('2').py()


@pytest.mark.isolate
def test_qargs_multi():
    os.environ['QARGS'] = '-p 5050 -t 1000'
    with pytest.warns() as warnings:
        import pykx as kx
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

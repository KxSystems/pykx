from pathlib import Path

import pytest


@pytest.mark.unlicensed
def test_QHOME(kx):
    assert isinstance(kx.config.qhome, Path)
    assert (kx.config.qhome/'q.k').exists()


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.config), list)
    assert sorted(dir(kx.config)) == dir(kx.config)

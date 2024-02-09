# Do not import pykx here - use the `kx` fixture instead!
import subprocess


def test_reimport(kx):
    with kx.PyKXReimport():
        ret = subprocess.run(["python", "-c", "import pykx"])
    assert 0 == ret.returncode


def test_reimport_kdefault(kx):
    with kx.PyKXReimport():
        ret = subprocess.run(
            ["python", "-c", "import os;os.environ['PYKX_DEFAULT_CONVERSION']='k';import pykx"])
    assert 0 == ret.returncode

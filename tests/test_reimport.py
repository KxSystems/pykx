# Do not import pykx here - use the `kx` fixture instead!
import os
import subprocess
import sys


# --- Reimport no longer requires the PyKXReimport helper (Python child) -----

def test_reimport(kx):
    ret = subprocess.run([sys.executable, "-c", "import pykx"])
    assert 0 == ret.returncode


def test_reimport_kdefault(kx):
    ret = subprocess.run(
        [sys.executable, "-c", "import os;os.environ['PYKX_DEFAULT_CONVERSION']='k';import pykx"])
    assert 0 == ret.returncode


# --- PyKXReimport remains available and functional --------------------------
# (Still required e.g. for spawning kdb+ 4.0/4.1 q subprocesses from Python -
#  see tests/manual_reimport_matrix.py for the full cross-process matrix.)

def test_reimport_old(kx):
    with kx.PyKXReimport():
        ret = subprocess.run([sys.executable, "-c", "import pykx"])
    assert 0 == ret.returncode


def test_reimport_kdefault_old(kx):
    with kx.PyKXReimport():
        ret = subprocess.run(
            [sys.executable, "-c",
             "import os;os.environ['PYKX_DEFAULT_CONVERSION']='k';import pykx"])
    assert 0 == ret.returncode


def test_pykxreimport_qhome_reset_restore(kx):
    # Inside the context QHOME is reset to the user's original QHOME; on exit the
    # pykx-internal QHOME is restored.
    qhome_before = os.environ.get('QHOME')
    with kx.PyKXReimport():
        assert os.environ['QHOME'] == os.environ['PYKX_OLD_QHOME']
    assert os.environ.get('QHOME') == qhome_before


# --- Environment-variable regressions ---------------------------------------

def test_pykx_old_qpath_populated(kx):
    # Regression: PYKX_OLD_QHOME/QPATH must hold real values so a child process
    # can restore them. PYKX_OLD_QPATH used to always be empty.
    assert os.environ.get('PYKX_OLD_QHOME', '') != ''
    assert os.environ.get('PYKX_OLD_QPATH', '') != ''


def test_qpath_no_duplicate_on_nested_import(kx, tmp_path):
    # Regression: nested KDB-X Python imports must not keep appending
    # `$QHOME/mod` to QPATH (it used to grow one entry per nesting level).
    script = tmp_path / "nested_import.py"
    script.write_text(
        "import os, sys, subprocess\n"
        "import pykx  # noqa: F401\n"
        "qp = os.environ.get('QPATH', '')\n"
        "parts = [p for p in qp.split(os.pathsep) if p]\n"
        "assert len(parts) == len(set(parts)), 'duplicate QPATH entries: ' + qp\n"
        "depth = int(sys.argv[1]) if len(sys.argv) > 1 else 0\n"
        "if depth < 3:\n"
        "    sys.exit(subprocess.run([sys.executable, sys.argv[0], str(depth + 1)]).returncode)\n"
    )
    ret = subprocess.run([sys.executable, str(script), "0"])
    assert 0 == ret.returncode


def test_setdefault_does_not_export_env(kx):
    # Regression: `.pykx.setdefault` must only change the current process, it must
    # NOT export PYKX_DEFAULT_CONVERSION (which would leak to child processes).
    before = os.environ.get('PYKX_DEFAULT_CONVERSION')
    try:
        kx.q('.pykx.setdefault["pd"]')
        assert os.environ.get('PYKX_DEFAULT_CONVERSION') == before
    finally:
        kx.q('.pykx.setdefault["default"]')

"""Initialization tests for KDB-X Python under q (KXI-69194, KXI-73830).

These faults crash q rather than raise, so each test runs a q process.
"""
import os
import subprocess
import sys
from pathlib import Path
from platform import system

import pytest


pytestmark = pytest.mark.skipif(
    system() == 'Windows',
    reason='pykx-under-q initialization is exercised on POSIX only',
)

_QSCRIPT = 'tests/qscripts/init_under_q.q'
_UNINIT_QSCRIPT = 'tests/qscripts/uninit_pyrun.q'


def _python_env_paths():
    """Python paths for the q script, from a subprocess so pykx is not imported here."""
    out = subprocess.run(
        (str(Path(sys.executable).as_posix()), '-c',
         'import sysconfig,sys,pykx;'
         'print(sysconfig.get_config_var("LIBDIR"));'
         'print(sysconfig.get_config_var("LDVERSION"));'
         'print(sys.base_prefix);'
         'print(sys.executable);'
         'print(pykx.config.pykx_dir)'),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or len(lines) < 5:
        pytest.skip(f'could not resolve the Python environment paths: {out.stdout!r}')
    libdir, ldversion, base_prefix, executable, pykx_dir = lines[-5:]
    return {
        'libpython': Path(libdir)/f'libpython{ldversion}.so',
        'pyhome': base_prefix,
        'pyexec': executable,
        'pykxdir': pykx_dir,
    }


def _install_qhome():
    """Install pykx.q into QHOME for the q subprocess."""
    subprocess.run(
        (str(Path(sys.executable).as_posix()), '-c',
         'import pykx;pykx.install_into_QHOME()'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _run_under_q(q_args=(), extra_env=None, timeout=90, script=_QSCRIPT):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ('q', script, '-q', *q_args),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=timeout, env=env,
    )


def _report(out, prefix):
    """Collect the `prefix,key,value` lines a q script prints."""
    report = {}
    for line in out.stdout.splitlines():
        parts = line.split(',', 2)
        if len(parts) == 3 and parts[0] == prefix:
            report[parts[1]] = parts[2]
    return report


def _assert_clean(out):
    """Check for a crash before checking the report."""
    for fatal in ('SIGSEGV', 'SIGABRT', 'SIGBUS', 'munmap_chunk', 'free():',
                  'double free', 'Bad memory block'):
        assert fatal not in out.stdout, f'{fatal} during pykx.q init:\n{out.stdout}'
    assert 'init_under_q,PASS' in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout


@pytest.mark.timeout(150)
def test_init_under_q_does_not_corrupt_memory():
    """Loading pykx.q, reloading it and dropping references must not corrupt memory."""
    _install_qhome()
    try:
        out = _run_under_q(timeout=90)
    except subprocess.TimeoutExpired:
        pytest.fail('pykx.q initialization checks timed out')
    _assert_clean(out)


@pytest.mark.timeout(150)
def test_init_under_q_argument_refcounts_preserved():
    """Arguments passed to k_init_python keep their reference counts."""
    _install_qhome()
    out = _run_under_q(timeout=90)
    report = dict(
        line.split(',', 2)[1:] for line in out.stdout.splitlines()
        if line.startswith('init_under_q,') and line.count(',') >= 2
    )
    # A missing entry means q crashed before reporting
    assert 'refcount_supported' in report, (
        f'q produced no reference count report, it likely crashed:\n{out.stdout}'
    )
    if report['refcount_supported'] == '0':
        pytest.skip('this q build does not support -16! reference counts')
    assert report.get('refcount_ok') == '1', (
        'reference count of an argument passed to k_init_python does not match '
        f'the control: {report}\n{out.stdout}'
    )


@pytest.mark.timeout(150)
@pytest.mark.parametrize('slaves', ['0', '4'])
def test_init_under_q_with_slave_threads(slaves):
    """Initialization is clean with and without secondary threads."""
    _install_qhome()
    out = _run_under_q(q_args=('-s', slaves), timeout=90)
    _assert_clean(out)


@pytest.mark.timeout(150)
def test_python_error_before_init_is_reported_not_fatal():
    """A Python error raised before initialization reaches q as an error."""
    paths = _python_env_paths()
    if not paths['libpython'].exists():
        pytest.skip(
            f'no shared libpython to initialize an interpreter from: {paths["libpython"]}'
        )

    q_args = (
        '-pykxdir', str(Path(paths['pykxdir']).as_posix()),
        '-libpython', str(paths['libpython'].as_posix()),
        '-pyhome', paths['pyhome'],
        '-pyexec', str(Path(paths['pyexec']).as_posix()),
    )
    try:
        out = _run_under_q(q_args=q_args, timeout=90, script=_UNINIT_QSCRIPT)
    except subprocess.TimeoutExpired:
        pytest.fail('uninitialized k_pyrun checks timed out')

    for fatal in ('SIGSEGV', 'SIGABRT', 'SIGBUS'):
        assert fatal not in out.stdout, (
            f'{fatal} converting a Python error raised before initialization:\n{out.stdout}'
        )
    report = _report(out, 'uninit_pyrun')
    # A missing entry means q crashed before reporting
    assert 'import_error' in report, (
        f'q produced no report, it likely crashed:\n{out.stdout}'
    )
    assert 'ImportError' in report['import_error'], report
    assert 'simulated GLIBC/libffi failure' in report['import_error'], report
    assert report.get('success_after') == 'ok', report
    assert 'ValueError' in report.get('second_error', ''), report
    assert 'not initialized' in report.get('foreign_to_q', ''), report
    assert 'uninit_pyrun,PASS' in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout

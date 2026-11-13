import os
import subprocess
import sys
from pathlib import Path
from platform import system

import pytest


pytestmark = pytest.mark.skipif(
    system() == 'Windows',
    reason='pykx-under-q threading (peach/GIL) fixes are POSIX-only',
)

_QSCRIPT = 'tests/qscripts/peach_under_q.q'


def _install_qhome():
    """Ensure pykx.q and the q libs are present in QHOME for the q subprocess."""
    subprocess.run(
        (str(Path(sys.executable).as_posix()), '-c',
         'import pykx;pykx.install_into_QHOME()'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _run_under_q(q_args=(), extra_env=None, timeout=90):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ('q', _QSCRIPT, '-q', *q_args),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=timeout, env=env,
    )


@pytest.mark.timeout(150)
def test_peach_under_q_no_deadlock_or_corruption():
    _install_qhome()
    try:
        out = _run_under_q(q_args=('-s', '4'), timeout=90)
    except subprocess.TimeoutExpired:
        pytest.fail('peach under q timed out - deadlock regression (KXI-67372)')
    assert 'SIGSEGV' not in out.stdout, out.stdout
    assert 'peach_under_q,PASS' in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout


@pytest.mark.timeout(120)
def test_serial_under_q_no_segfault():
    _install_qhome()
    try:
        out = _run_under_q(q_args=('-s', '0'), timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail('serial under q timed out unexpectedly')
    assert 'SIGSEGV' not in out.stdout, out.stdout
    assert 'peach_under_q,PASS' in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout


@pytest.mark.timeout(150)
def test_pykx_threading_flag_under_q_does_not_crash():
    _install_qhome()
    try:
        out = _run_under_q(q_args=('-s', '4'), extra_env={'PYKX_THREADING': '1'}, timeout=90)
    except subprocess.TimeoutExpired:
        pytest.fail('under q with PYKX_THREADING=1 timed out')
    assert 'SIGSEGV' not in out.stdout, out.stdout
    assert 'peach_under_q,PASS' in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout

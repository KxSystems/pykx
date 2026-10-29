"""Reference count and GIL regression tests for KDB-X Python under q (KXI-73830).

Several entry points in the under-q C library took a Python reference, or acquired
the GIL, and returned without giving it back: `k_py_error`, `import`, `get_global`,
`set_attr`, `repr`, `foreign_to_q` and `call_func`.

The GIL leak is specific to running under q, where the process holds no GIL between
calls, so an unbalanced `PyGILState_Ensure` starves every other Python thread for the
life of the process. None of it is observable from inside the process that leaked, so
these run a real q process and read the report it prints.
"""
import os
import subprocess
from platform import system

import pytest


pytestmark = pytest.mark.skipif(
    system() != 'Linux',
    reason='the resident memory probe reads /proc',
)

_QSCRIPT = 'tests/qscripts/refcounts_under_q.q'

# The repr leak was 168 pages per call, and page accounting drifts a little on its own.
_MAX_REPR_PAGES_PER_OP = 20

# The probe thread ticks every 5ms, so a second of q being idle should show near 200.
# With the GIL held it managed 1.
_MIN_THREAD_TICKS = 50


@pytest.fixture(scope='module')
def report():
    """Run the q script once and return the values it printed."""
    out = subprocess.run(
        ('q', _QSCRIPT, '-q'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=300, env=dict(os.environ),
    )
    for fatal in ('SIGSEGV', 'SIGABRT', 'SIGBUS'):
        assert fatal not in out.stdout, f'{fatal} during the checks:\n{out.stdout}'
    assert 'refcounts_under_q,PASS' in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout
    values = {}
    for line in out.stdout.splitlines():
        parts = line.split(',', 2)
        if len(parts) == 3 and parts[0] == 'refcounts_under_q':
            values[parts[1]] = parts[2]
    return values


@pytest.mark.timeout(400)
@pytest.mark.parametrize('entry_point', [
    'exception_type',   # k_py_error
    'module',           # import
    'global_object',    # get_global
    'attr_owner',       # set_attr
])
def test_no_reference_leak_under_q(report, entry_point):
    assert entry_point in report, report
    assert int(report[entry_point]) == 0, (
        f'{entry_point} leaks {report[entry_point]} reference(s) per call'
    )


@pytest.mark.timeout(400)
def test_repr_does_not_leak_memory_under_q(report):
    assert 'repr_pages_per_op' in report, report
    pages = int(report['repr_pages_per_op'])
    assert pages < _MAX_REPR_PAGES_PER_OP, (
        f'repr grows resident memory by {pages} pages per call'
    )


@pytest.mark.timeout(400)
def test_gil_released_when_foreign_is_not_callable(report):
    # The error itself is checked too, the fix must not have turned a clean error
    # into something else.
    assert report.get('non_callable_errors') == '1', (
        f'calling a non-callable foreign did not raise the expected q error: {report}'
    )
    ticks = int(report['thread_ticks_while_idle'])
    assert ticks > _MIN_THREAD_TICKS, (
        f'a Python thread advanced only {ticks} ticks while q was idle, '
        'which means the GIL was not released'
    )

from contextlib import closing, contextmanager
import marshal
import multiprocessing as mp
import os
from platform import system
from random import randint
import signal
import socket
import subprocess
import sys
from time import sleep

import _pytest
from _pytest.runner import runtestprotocol
import py
import pytest


py_minor_version = sys.version_info[1]


@pytest.fixture
def pa(kx):
    """Fixture that safely imports PyArrow, and skips the test if PyArrow is unavailable."""
    pyarrow = kx._pyarrow.pyarrow
    if pyarrow is None:
        pytest.skip('PyArrow is unavailable')
    return pyarrow


@pytest.fixture
def pd(kx):
    """Fixture that safely imports Pandas after PyKX has potentially blocked PyArrow imports."""
    import pandas
    return pandas


@pytest.fixture
def kx(request):
    markers = {x.name: x.kwargs for x in request.node.own_markers}
    if 'isolate' in markers and 'pykx' in sys.modules:
        raise Exception('PyKX has been imported - test isolation has failed')
    # Use the seed set by the `pytest-randomly` plugin as q's random seed
    seed = int(request.config.getoption('randomly_seed')) % (2**31 - 1)
    os.environ['QARGS'] = f'-S {seed} --testflag {request.param}'
    os.environ['PYKX_RELEASE_GIL'] = '1'
    os.environ['PYKX_Q_LOCK'] = '-1'
    if os.getenv('CI') and system() == 'Linux' and randint(0, 1) % 2 == 0:
        os.environ['PYKX_Q_LIB_LOCATION'] = '/specified_q_lib_path'
    import pykx as kx
    return kx


@pytest.fixture
def q(request, kx, q_init):
    if request.param == 'embedded':
        kx.q._inter_test_q = kx.q.__call__
        kx.q.__call__ = (
            lambda *args, **kwargs: kx.q._inter_test_q(
                *args,
                **kwargs,
                debug=True if randint(0, 1) == 0 else False
            )
        )
        yield kx.q
    elif request.param == 'ipc':
        with q_proc(q_init) as port:
            with kx.QConnection(port=port) as conn:
                _inter_test_q = conn.__call__
                conn.__call__ = (
                    lambda *args, **kwargs: _inter_test_q(
                        *args,
                        **kwargs,
                        debug=True if randint(0, 1) == 0 else False
                    )
                )
                yield conn


def random_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


# Need to know what $QHOME to use for the q subprocesses before PyKX changes it.
original_QHOME = os.environ['QHOME']


@contextmanager
def q_proc(q_init):
    proc = None
    try:
        port = random_free_port()
        proc = subprocess.Popen(
            (lambda x: x.split() if system() != 'Windows' else x)(f'q -p {port}'),
            stdin=subprocess.PIPE,
            stdout=subprocess.sys.stdout,
            stderr=subprocess.sys.stderr,
            start_new_session=True,
            env={**os.environ, 'QHOME': original_QHOME},
        )
        if system() != 'Windows':
            q_init.append((f'system"kill -USR1 {os.getpid()}"').encode())
        proc.stdin.write(b'\n'.join((*q_init, b'')))
        proc.stdin.flush()
        if system() == 'Windows':
            sleep(2) # Windows does not support the signal-based approach used here
        else:
            signal.sigwait([signal.SIGUSR1]) # Wait until all initial queries have been run
        yield port
    finally:
        if proc is not None:
            proc.stdin.close()
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.terminate()
            proc.wait()


@pytest.fixture
def q_init():
    return [b''] # Default value for initialization of a q process


@pytest.fixture
def q_port(q_init):
    with q_proc(q_init) as port:
        yield port


def serialize_report(rep):
    d = rep.__dict__.copy()
    d['longrepr'] = str(rep.longrepr) if hasattr(rep.longrepr, 'toterminal') else rep.longrepr
    for name in d:
        if isinstance(d[name], py.path.local):
            d[name] = str(d[name])
        elif name == 'result':
            d[name] = None
    return d


EXITSTATUS_TESTEXIT = 4


def isolated_test_runner(test_item, pipe):
    try:
        reports = runtestprotocol(test_item, log=False)
    except KeyboardInterrupt:
        os._exit(EXITSTATUS_TESTEXIT)
    pipe.send(marshal.dumps([serialize_report(x) for x in reports]))
    pipe.close()


def isolate_test(item):
    mp_ctx = mp.get_context('fork')
    out_pipe, in_pipe = mp_ctx.Pipe(duplex=False)
    proc = mp_ctx.Process(target=isolated_test_runner, args=(item, in_pipe))
    proc.start()
    timeout = 50.0
    for x in item.own_markers:
        if x.name == 'timeout':
            timeout = x.args[0]
    try:
        if out_pipe.poll(timeout):
            return out_pipe.recv()
        raise TimeoutError(f'No response from isolated test {item.name!r} after {timeout} seconds')
    finally:
        proc.join()


def pytest_configure(config):
    config.option.verbose = 3
    config.option.reportchars = 'a'
    config.option.cov_source = ['pykx']


def pytest_addoption(parser):
    parser.addoption(
        '--large',
        action='store_true',
        default=False,
        help='Run large tests (processor/memory/time intensive)'
    )


def pytest_collection_modifyitems(session, config, items):
    def f(item) -> bool:
        # Skip all tests that would attempt to run using embedded q in unlicensed mode, because that
        # will never work.
        if item.name.endswith('[embedded-unlicensed]'):
            return False

        # Skip tests decorated by `pytest.mark.large` by default. Run these tests by including the
        # `--large` flag. Exclusively run these tests with `-m large --large`.
        if not config.getoption('--large') and item.get_closest_marker('large'):
            return False

        # Include every other test by default.
        return True

    items[:] = [item for item in items if f(item)] # Slice-assignment to update in-place.


def pytest_itemcollected(item):
    # Rename embedded-licensed -> embedded; "licensed" is a given when running with embedded q.
    item._nodeid = item._nodeid.replace('[embedded-licensed]', '[embedded]')


test_count = 0


def pytest_generate_tests(metafunc): # noqa
    global test_count
    markers = {x.name: x.kwargs for x in metafunc.definition.own_markers}

    unlicensed = 'unlicensed' in markers
    ipc = 'ipc' in markers

    if 'unlicensed' in markers and 'kx' not in metafunc.fixturenames:
        raise ValueError("Cannot run test in unlicensed mode if the 'kx' fixture is not provided.")
    if ipc and 'q' not in metafunc.fixturenames:
        raise ValueError("Cannot run test in ipc mode if the 'q' fixture is not provided.")

    if 'kx' in metafunc.fixturenames:
        # The request params are given to QARGS, so both the empty string and '--licensed' will
        # result in the 'kx' fixture being in licensed mode. We use each 50% of the time because
        # while they should do exactly the same thing, we don't want to only test one of the two.
        licensed_arg = '--licensed' if test_count % 2 else ''
        kx_fixture_kwargs = {
            'argnames': 'kx',
            'argvalues': [pytest.param(licensed_arg, marks=[pytest.mark.licensed])],
            'indirect': True,
            'ids': lambda argvalue: argvalue.lstrip('-') if argvalue else 'licensed',
        }
        if 'nep49' in markers and py_minor_version >= 8:
            kx_fixture_kwargs['argvalues'].append(
                pytest.param(' --pykxalloc --pykxgc',
                             marks=[pytest.mark.licensed, pytest.mark.nep49]
                )
            )
    embedded_argvalue = pytest.param('embedded', marks=[pytest.mark.embedded, pytest.mark.licensed])
    if 'q' in metafunc.fixturenames:
        metafunc.parametrize(
            argnames='q',
            argvalues=[embedded_argvalue, 'ipc'] if ipc else [embedded_argvalue],
            indirect=True,
        )

    if unlicensed or (ipc and not markers['ipc'].get('licensed_only', False)):
        if 'unlicensed' in markers and markers['unlicensed'].get('unlicensed_only', False):
            kx_fixture_kwargs['argvalues'].pop()
            if 'nep49' in markers and py_minor_version >= 8:
                kx_fixture_kwargs['argvalues'].pop()
        kx_fixture_kwargs['argvalues'].append(
            pytest.param('--unlicensed', marks=[pytest.mark.isolate, pytest.mark.unlicensed])
        )
        if 'nep49' in markers and py_minor_version >= 8:
            kx_fixture_kwargs['argvalues'].append(
                pytest.param('--unlicensed --pykxalloc',
                             marks=[pytest.mark.isolate,
                                    pytest.mark.unlicensed,
                                    pytest.mark.nep49]
                )
            )
        metafunc.parametrize(**kx_fixture_kwargs)
    elif 'kx' in metafunc.fixturenames:
        metafunc.parametrize(**kx_fixture_kwargs)

    test_count += 1


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item):
    # FIXME: Because fork is being used instead of spawn, as soon as the first licensed test runs,
    # all tests afterwards must be licensed. To work around this issue, we run every test within
    # a fork that has not imported PyKX, unless we are on Windows.
    if (system() != 'Windows') and (item.get_closest_marker('isolate') or True):
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        reports = [_pytest.runner.TestReport(**x) for x in marshal.loads(isolate_test(item))]
        for rep in reports:
            item.ihook.pytest_runtest_logreport(report=rep)
        item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        return True


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_setupnodes(config, specs):
    # Shorten worker names when possible. This is to help avoid the pytest-xdist glitch where the
    # worker init logging hits the right edge of your terminal, then prints out many dozens of junk
    # lines.
    if len(specs) <= 26:
        for i, spec in enumerate(specs):
            spec.id = chr(ord('A') + i)

from io import StringIO
from contextlib import contextmanager
import sys

import pytest


@contextmanager
def replace_stdin(new_stdin):
    orig_stdin = sys.stdin
    sys.stdin = new_stdin
    yield
    sys.stdin = orig_stdin


@contextmanager
def attr_set_and_restore(x, attr, value):
    """Sets `x.attr = value` when entered and exited."""
    setattr(x, attr, value)
    try:
        yield x
    finally:
        setattr(x, attr, value)


@pytest.mark.ipc
def test_basic_usage(q):
    with StringIO('a:til 10\n\\\\') as stdin:
        with replace_stdin(stdin):
            q.console()
    assert q['a'].py() == list(range(10))


@pytest.mark.ipc
def test_k_mode(q):
    with attr_set_and_restore(q.console, 'k_mode', False):
        with StringIO('\\\nb:!10\n\\\n\\\\') as stdin:
            with replace_stdin(stdin):
                q.console()
    assert q['b'].py() == list(range(10))


@pytest.mark.ipc
def test_switching_mode(q):
    with attr_set_and_restore(q.console, 'k_mode', False):
        with StringIO('c:til 8\n\\\n\\\\') as stdin:
            with replace_stdin(stdin):
                q.console()
        with StringIO('d:!2\n\\\ne:til 12\n\\\n\\\\') as stdin:
            with replace_stdin(stdin):
                q.console()
    assert q['c'].py() == list(range(8))
    assert q['d'].py() == list(range(2))
    assert q['e'].py() == list(range(12))


@pytest.mark.ipc
def test_entering_blank_lines(q):
    with StringIO('\n\n\n\n\\\\') as stdin:
        with replace_stdin(stdin):
            q.console()


# FIXME: The `test_input_eof_*` tests should be parametrized, but for whatever reason the
# `parametrize` deocrator is causing embedded q to be used during the unlicensed IPC test.

# @pytest.mark.parametrize(
#     'input_text',
#     argvalues=['', '\n\n', '\n([]1 2 3 4;"atcg")\n', 't:([]1 2 3 4;"atcg")'],
#     ids=['empty', 'newlines', 'data', 'assignment'],
# )


@pytest.mark.ipc
def test_input_eof_empty(q):
    with StringIO('') as stdin:
        with replace_stdin(stdin):
            q.console()


@pytest.mark.ipc
def test_input_eof_newlines(q):
    with StringIO('\n\n') as stdin:
        with replace_stdin(stdin):
            q.console()


@pytest.mark.ipc
def test_input_eof_assignment(q):
    with StringIO('t:([]1 2 3 4;"atcg")') as stdin:
        with replace_stdin(stdin):
            q.console()
    assert q['t'].py() == {'x': [1, 2, 3, 4], 'x1': b'atcg'}


@pytest.mark.ipc
def test_q_error(q, capsys):
    with StringIO('(`$())#([]x:())') as stdin:
        with replace_stdin(stdin):
            q.console()
    captured = capsys.readouterr()
    assert captured.out == 'q)q)\n'
    assert captured.err == "'rank\n"

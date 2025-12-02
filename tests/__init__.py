from contextlib import contextmanager
import os
from pathlib import Path
from platform import system
import signal
import sys
import threading

import pytest
import toml


os.environ['PYTHONWARNINGS'] = 'ignore:No data was collected,ignore:Module pykx was never imported'

if system() != 'Windows':
    if threading.current_thread() == threading.main_thread():
        signal.signal(signal.SIGUSR1, lambda *_: None)


# Decorator for tests that may damage the environment they are run in, and thus should only be run
# in disposable environments such as within Docker containers in CI. GitLab Runners provides the
# env var we check for: https://docs.gitlab.com/ee/ci/variables/predefined_variables.html
disposable_env_only = pytest.mark.skipif(
    os.environ.get('CI_DISPOSABLE_ENVIRONMENT', '').lower() not in ('true', '1'),
    reason='Test must be run in a disposable environment',
)


@contextmanager
def cd(newdir):
    """Change the current working directory within the context."""
    prevdir = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)


@contextmanager
def attr_set_and_restore(x, attr, value):
    """Sets `x.attr = value` when entered and exited."""
    setattr(x, attr, value)
    try:
        yield x
    finally:
        setattr(x, attr, value)


@contextmanager
def replace_stdin(new_stdin):
    orig_stdin = sys.stdin
    sys.stdin = new_stdin
    yield
    sys.stdin = orig_stdin

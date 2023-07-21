"""A single point in PyKX through which PyArrow can be imported safely.

PyArrow can appear to install without error, but then fail to import with a Python exception, or a
segmentation fault.

Because it's very hard to know when a segfault may occur when importing PyArrow, we test it out
first by importing it in a subprocess. If that works, then we import it here, and provide it for
the rest of PyKX to use under the name `pyarrow`. Otherwise `pyarrow` is set to `None`.
"""
import os

from .config import load_pyarrow_unsafe

if load_pyarrow_unsafe:
    import pyarrow
else:
    # We do the import in this weird way because otherwise it fails on macOS in some cases with:
    # `AttributeError: module 'importlib' has no attribute 'util'`
    # See: https://stackoverflow.com/a/39661116/5946921
    from importlib import util as importlib_util

    import builtins
    from pathlib import Path
    import signal
    import subprocess
    import sys
    from warnings import warn

    from .exceptions import PyKXWarning

    def _msg_from_return_code(return_code): # nocov
        # On POSIX, subprocess provides a negative return code if the process exits from a signal
        if os.name != 'posix' or return_code not in range(-1, -signal.NSIG, -1):
            return f'Process exited with return code {return_code}'
        generic_signal_msg = f'Process exited from signal {-return_code}'
        if not hasattr(signal, 'strsignal'):
            return generic_signal_msg
        sig_str = signal.strsignal(-return_code)
        return sig_str if sig_str else generic_signal_msg

    pyarrow = None
    import_attempt_output = None
    original_importer = builtins.__import__

    def pyarrow_importer(name, globals=None, locals=None, fromlist=(), level=0): # nocov
        if name == 'pyarrow':
            raise ImportError('PyArrow could not be imported. Check '
                              '`pykx._pyarrow.import_attempt_output` for the reason.')
        return original_importer(name, globals, locals, fromlist, level)

    # Only try to import PyArrow (and potentially issue a warning) if it is installed
    if importlib_util.find_spec('pyarrow'):
        cmd = (str(Path(sys.executable).as_posix()), '-c', 'import pyarrow')
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if p.returncode: # nocov
            import_attempt_output = p.stdout if p.stdout else _msg_from_return_code(p.returncode)
            # Don't print out `import_attempt_output` by default.
            warn('PyArrow failed to load - PyArrow related functionality has been disabled. Check '
                 '`pykx._pyarrow.import_attempt_output` for the reason.', PyKXWarning)
            # Replace the `__import__` function to prevent other from trying to import PyArrow
            builtins.__import__ = pyarrow_importer
        else:
            import pyarrow # noqa: F401
    else: # nocov
        pass # PyArrow is not installed; nothing need be done!

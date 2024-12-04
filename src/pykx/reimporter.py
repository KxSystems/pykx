"""PyKX reimport helper module.

PyKX uses various environment variables to monitor the state of various modules initialization. This
is required to manage all of the different modes of operation, however it can cause issues when
attempting to reimport PyKX within a spawned subprocess.

This module provides a mechanism to allow users to safely reimport PyKX within spawned subprocesses
without having to manually manage any of these internal environment variables.
"""
import os

from .config import pykx_executable, qhome


class PyKXReimport:
    """Helper class to help manage the environment variables around reimporting PyKX in a
    subprocess.

    It is strongly recommended to use this class by using the python `with` syntax. This will ensure
    all the environment variables are reset and restored correctly, without the need to manage this
    yourself.

    Examples:

    ```
    with kx.PyKXReimport():
        # This process can safely import PyKX
        subprocess.Popen(f"python other_file.py")
    ```
    """

    def __init__(self):
        self.envlist = ('PYKX_DEFAULT_CONVERSION',
                        'PYKX_UNDER_Q',
                        'PYKX_UNDER_PYTHON',
                        'PYKX_SKIP_UNDERQ',
                        'PYKX_Q_LOADED_MARKER',
                        'PYKX_LOADED_UNDER_Q',
                        'QHOME',
                        'PYKX_EXECUTABLE',
                        'PYKX_DIR')
        self.envvals = [os.getenv(x) for x in self.envlist]

    def __enter__(self):
        self.reset()
        return self

    def reset(self):
        """Reset all the required environment variables.

        Note: It is not recommended to use this function directly instead use the `with` syntax.
            This will automatically manage setting and restoring the environment variables for you.
        """
        for x, y in zip(self.envlist, self.envvals):
            os.unsetenv(x)
            if y is not None:
                del os.environ[x]
        os.environ['QHOME'] = str(qhome)
        os.environ['PYKX_EXECUTABLE'] = pykx_executable

    def restore(self):
        """Restore all the required environment variables.

        Note: It is not recommended to use this function directly instead use the `with` syntax.
            This will automatically manage setting and restoring the environment variables for you.
        """
        for x, y in zip(self.envlist, self.envvals):
            if y is not None:
                os.environ[x] = y

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.restore()

    def __del__(self):
        self.restore()

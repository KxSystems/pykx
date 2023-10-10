"""PyKX reimport helper module.

PyKX uses various environment variables to monitor the state of various modules initialization. This
is required to manage all of the different modes of operation, however it can cause issues when
attempting to reimport PyKX within a spawned subprocess.

This module provides a mechanism to allow users to safely reimport PyKX within spawned subprocesses
without having to manually manage any of these internal environment variables.
"""
import os


original_qhome = str(os.getenv('QHOME'))


class PyKXReimport:
    """Helper class to help manage the enviroment variables around reimporting PyKX in a
    subprocess.

    It is strongly reccomended to use this class by using the python `with` syntax. This will ensure
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
        self.defaultconv = str(os.getenv('PYKX_DEFAULT_CONVERSION'))
        self.pykxunderq = str(os.getenv('PYKX_UNDER_Q'))
        self.skipunderq = str(os.getenv('SKIP_UNDERQ'))
        self.underpython = str(os.getenv('UNDER_PYTHON'))
        self.pykxqloadedmarker = str(os.getenv('PYKX_Q_LOADED_MARKER'))
        self.pykxloadedunderq = str(os.getenv('PYKX_LOADED_UNDER_Q'))
        self.qhome = str(os.getenv('QHOME'))

    def __enter__(self):
        self.reset()
        return self

    def reset(self):
        """Reset all the required environment variables.

        Note: It is not reccomended to use this function directly instead use the `with` syntax.
            This will automatically manage setting and restoring the environment variables for you.
        """
        os.unsetenv("PYKX_DEFAULT_CONVERSION")
        os.unsetenv("PYKX_UNDER_Q")
        os.unsetenv("SKIP_UNDERQ")
        os.unsetenv("UNDER_PYTHON")
        os.unsetenv("PYKX_Q_LOADED_MARKER")
        os.unsetenv("PYKX_LOADED_UNDER_Q")
        os.environ['QHOME'] = original_qhome

    def restore(self):
        """Restore all the required environment variables.

        Note: It is not reccomended to use this function directly instead use the `with` syntax.
            This will automatically manage setting and restoring the environment variables for you.
        """
        os.environ['PYKX_DEFAULT_CONVERSION'] = self.defaultconv
        os.environ['PYKX_UNDER_Q'] = self.pykxunderq
        os.environ['SKIP_UNDERQ'] = self.skipunderq
        os.environ['UNDER_PYTHON'] = self.underpython
        os.environ['PYKX_Q_LOADED_MARKER'] = self.pykxqloadedmarker
        os.environ['PYKX_LOADED_UNDER_Q'] = self.pykxloadedunderq
        os.environ['QHOME'] = self.qhome

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.restore()

    def __del__(self):
        self.restore()

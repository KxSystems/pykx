"""KDB-X Python reimport helper module.

KDB-X Python uses various environment variables to monitor the state of various modules
initialization. This is required to manage all of the different modes of operation,
however it can cause issues when attempting to reimport KDB-X Python within a spawned subprocess.

This module provides a mechanism to allow users to safely reimport KDB-X Python within spawned
subprocesses without having to manually manage any of these internal environment variables.
"""
import os


class PyKXReimport:
    """Helper class to help manage the environment variables around reimporting KDB-X Python in a
    subprocess.

    It is strongly recommended to use this class by using the python `with` syntax. This will ensure
    all the environment variables are reset and restored correctly, without the need to manage this
    yourself.

    Examples:

    ```
    with kx.PyKXReimport():
        # This process can safely import KDB-X Python
        subprocess.Popen(f"python other_file.py")
    ```

    Note: Not required in most cases.
        KDB-X Python now automatically manages these environment variables, so this helper is not
        needed for the majority of use cases (for example spawning a Python subprocess, or a
        KDB-X 5.0 q process).

    Note: Still required for spawning kdb+ 4.0/4.1 q subprocesses from Python.
        When KDB-X Python is imported in a Python process it redirects `QHOME` to its internal
        directory. A kdb+ 4.0/4.1 `q` child inheriting that `QHOME` fails to start because the
        pykx-internal `QHOME` does not contain `q.k`. Wrapping the spawn in `PyKXReimport` restores
        the user's `QHOME` (which contains `q.k`) for the child. This also applies where the QHOME
        symlinks pykx creates are unavailable or unreliable (e.g. Windows). Alternatively set
        `PYKX_IGNORE_QHOME=true` so `QHOME` is not redirected in the first place.
    """

    def __init__(self):
        self.pykx_qhome = os.getenv('QHOME', '')

    def __enter__(self):
        self.reset()
        return self

    def reset(self):
        """Reset all the required environment variables.

        Note: It is not recommended to use this function directly instead use the `with` syntax.
            This will automatically manage setting and restoring the environment variables for you.
        """
        os.environ['QHOME'] = os.getenv('PYKX_OLD_QHOME', self.pykx_qhome)

    def restore(self):
        """Restore all the required environment variables.

        Note: It is not recommended to use this function directly instead use the `with` syntax.
            This will automatically manage setting and restoring the environment variables for you.
        """
        os.environ['QHOME'] = self.pykx_qhome

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.restore()

    def __del__(self):
        # Best-effort restore; `__init__` may not have completed (e.g. QHOME unset).
        qhome = getattr(self, 'pykx_qhome', None)
        if qhome is not None:
            os.environ['QHOME'] = qhome

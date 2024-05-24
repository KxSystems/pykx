"""System command wrappers for PyKX."""
import os
from pathlib import Path
from warnings import warn

from . import Q, wrappers as k
from .exceptions import PyKXWarning, QError


__all__ = ['SystemCommands']


def __dir__():
    return __all__


class SystemCommands:
    """Wrappers for `q` system commands.

    More documentation on all the system commands available to `q` can be
    [found here](https://code.kx.com/q/basics/syscmds/).
    """

    def __init__(self, q: Q):
        self._q = q

    def __call__(self, x):
        return self._q('{system x}', k.CharVector(x))

    def tables(self, namespace=None):
        """Lists the tables associated with a namespace/dictionary

        Examples:

        Retrieve the tables within a provided namespace:

        ```python
        kx.system.tables('.foo')
        ```

        Retrieve the tables within a provided dictionary:

        ```python
        kx.system.tables('foo')
        ```
        """
        if namespace is not None:
            namespace = str(namespace)
            return self._q._call(f'\\a {namespace}', wait=True)
        return self._q._call('\\a', wait=True)

    @property
    def console_size(self):
        """The maximum console size for the q process in the format rows, columns.

        The size of the output for the q process before truncating the rest with `...`.

        Expects a 2 item list denoting the console width and console height.

        Note: Does not work over IPC, only within EmbeddedQ.

        Examples:

        Get the maximum console_size size of output for EmbeddedQ to 10 columns and 10 rows.

        ```python
        kx.q.system.console_size
        ```

        Set the maximum console size of output for EmbeddedQ to 10 columns and 10 rows.

        ```python
        kx.q.system.console_size = [10, 10]
        ```
        """
        return self._q._call('\\c', wait=True)

    @console_size.setter
    def console_size(self, size):
        if 'QConnection' in str(self._q):
            raise QError('Setting console size does not work over IPC.')
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError('Expected 2 item list for setting q console size.')
        return self._q._call(f'\\c {int(size[0])} {int(size[1])}', wait=True)

    @property
    def display_size(self):
        """The maximum console display size for the q process in the format rows, columns.

        The size of the output for the q process before truncating the rest with `...`.

        Expects a 2 item list denoting the display width and console height.

        Note: Does not work over IPC, only within EmbeddedQ.

        Examples:

        Get the maximum display size of output for EmbeddedQ to 10 columns and 10 rows.

        ```python
        kx.q.system.display_size
        ```

        Set the maximum display size of output for EmbeddedQ to 10 columns and 10 rows.

        ```python
        kx.q.system.display_size = [10, 10]
        ```
        """
        return self._q._call('\\c', wait=True)

    @display_size.setter
    def display_size(self, size):
        if 'QConnection' in str(self._q):
            raise QError('Setting display size does not work over IPC.')
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError('Expected 2 item list for setting q console size.')
        return self._q._call(f'\\c {int(size[0])} {int(size[1])}', wait=True)

    def cd(self, directory=None):
        """Get the current directory or change the current directory.

        Examples:

        Get the current working directory.

        ```python
        kx.q.system.cd()
        ```

        Change the current working directory to the root directory on a `UNIX` like machine.

        ```python
        kx.q.system.cd('/')
        ```
        """
        if directory is not None:
            return self._q._call(f'\\cd {directory}', wait=True)
        return self._q._call('\\cd', wait=True)

    def namespace(self, ns=None):
        """Get the current namespace or change to a new namespace.

        Examples:

        Get the current namespace.

        ```python
        kx.q.system.namespace()
        ```

        Change the current namespace to `.foo`, note the leading `.` may be ommited.

        ```python
        kx.q.system.namespace('foo')
        ```

        Return to the default namespace.

        ```python
        kx.q.system.namespace('')
        ```
        """
        if 'QConnection' in str(self._q):
            raise QError('Namespaces do not work over IPC.')
        if ns is not None:
            ns = str(ns)
            if len(ns) > 0:
                if ns[0] != '.':
                    ns = '.' + ns
            else:
                ns = '.'
            return self._q._call(f'\\d {ns}', wait=True)
        return self._q._call('\\d', wait=True)

    def functions(self, ns=None):
        """Get the functions available in the current namespace or functions in a
        provided namespace or dictionary.

        Examples:

        Get the functions within the current namespace.

        ```python
        kx.q.system.functions()
        ```

        Get the functions within the `.foo` namespace.

        ```python
        kx.q.system.functions('.foo')
        ```

        Get the functions within a dictionary.

        ```python
        kx.q.system.function('foo')
        ```
        """
        if ns is not None:
            ns = str(ns)
            return self._q._call(f'\\f {ns}', wait=True)
        return self._q._call('\\f', wait=True)

    @property
    def garbage_collection(self):
        """The current garbage collection mode that `q` will use.

        0 - default deferred collection.
        1 - immediate collection.

        Examples:

        Get the current garbage collection mode.

        ```python
        kx.q.system.garbage_collection
        ```

        Set the current garbage collection mode to immediate collection.

        ```python
        kx.q.system.garbage_collection = 1
        ```
        """
        return self._q._call('\\g', wait=True)

    @garbage_collection.setter
    def garbage_collection(self, value):
        value = int(value)
        if value == 0 or value == 1:
            return self._q._call(f'\\g {value}', wait=True)
        raise ValueError('Garbage collection mode can only be set to either 0 or 1.')

    def load(self, path):
        """Loads a q script or a directory of a splayed table.

        Examples:

        Load a q script named `foo.q`.

        ```python
        kx.q.system.load('foo.q')
        ```
        """
        if isinstance(path, k.CharAtom) or isinstance(path, k.CharVector):
            path = path.py().decode()
        elif isinstance(path, k.SymbolAtom):
            path = path.py()
            if path[0] == ':':
                path = path[1:]
        elif isinstance(path, Path):
            path = str(path)
        if ' ' not in path:
            if path[-1] == '/':
                path = path[:-1]
            print(path)
            return self._q._call(f'\\l {path}', wait=True)
        warn('Detected a space in supplied path\n'
             f'  Path: \'{path}\'\n'
             'q system loading does not support spaces, attempting load '
             'using alternative load operation', PyKXWarning)
        full_path = os.path.abspath(path)
        load_path = Path(full_path)
        folder = load_path.parent.as_posix()
        file = load_path.name

        if not (load_path.is_dir() or load_path.is_file()):
            raise ValueError(f'Provided user path \'{str(load_path)} \'is not a file/directory')
        return self._q._call('.pykx.util.loadfile', k.CharVector(folder),
                             k.CharVector(file), wait=True)

    @property
    def utc_offset(self):
        """Get the local time offset as an integer from UTC.

        If abs(offset) > 23 then return the offset in minutes.

        Examples:

        Get the current local time offset.

        ```python
        kx.q.system.utc_offset
        ```

        Set the current local time offset to be -4:00 from UTC.

        ```python
        kx.q.system.utc_offset = -4
        ```
        """
        return self._q._call('\\o', wait=True)

    @utc_offset.setter
    def utc_offset(self, value):
        value = int(value)
        return self._q._call(f'\\o {value}', wait=True)

    @property
    def precision(self):
        """Get the precision for floating point numbers (number of digits shown).

        Note: Setting this property does not work over IPC, only works within `EmbeddedQ`.

        Examples:

        Get the current level of float precision.

        ```python
        kx.q.system.precision
        ```

        Set the Level of float precision to 2.

        ```python
        kx.q.system.precision = 2
        ```
        """
        return self._q._call('\\P', wait=True)

    @precision.setter
    def precision(self, prec):
        if 'QConnection' in str(self._q):
            raise QError('Setting float precision does not work over IPC.')
        prec = int(prec)
        return self._q._call(f'\\P {prec}', wait=True)

    def rename(self, src, dest):
        """Rename file `src` to `dest`.

        Equivalent to the unix `mv` command or Windows `move` command.

        Note: Cannot rename to a different disk.

        Examples:

        Rename a file `foo.q` to `bar.q`.

        ```python
        kx.q.system.rename('foo.q', 'bar.q')
        ```
        """
        return self._q._call(f'\\r {src} {dest}', wait=True)

    @property
    def max_num_threads(self):
        """The maximum number of secondary threads available to q.

        This value can be set using the `-s` command-line flag, provided to q via the $QARGS
        environment variable. For example: `QARGS='-s 8' python`, or `QARGS='-s 0' python`.
        """
        # system "s 0N" returns the threads set with -s not the current system "s" setting
        return int(self._q._call('system"s 0N"', wait=True))

    @max_num_threads.setter
    def max_num_threads(self, value):
        raise AttributeError("Can't set attribute - the maximum number of threads available to "
                             "q is fixed on startup by the '-s' flag.")

    @property
    def num_threads(self):
        """The current number of secondary threads being used by q.

        Computations in q will automatically be parallelized across this number of threads as q
        deems appropriate.

        This property is meant to be modified on `EmbeddedQ` and `QConnection` instances.

        Examples:

        Set the number of threads for embedded q to use to 8.

        ```python
        kx.q.num_threads = 8
        ```

        Set the number of threads for a q process being connected to over IPC to 8.

        ```python
        q = kx.SyncQConnection('localhost', 5001)
        q.num_threads = 8
        ```
        """
        return int(self._q._call('system"s"', wait=True))

    @num_threads.setter
    def num_threads(self, value):
        max_num_threads = self.max_num_threads
        value = int(value)
        if value > max_num_threads:
            raise ValueError(f'Cannot use {value} secondary threads - '
                             f'cannot exceed {max_num_threads}')
        self._q._call(f'system"s {value}"', wait=True)

    @property
    def random_seed(self):
        """The last value the the random seed was initialized with.

        Examples:

        Get the current seed value.

        ```python
        kx.q.system.random_seed
        ```

        Set the random seed value to 23.

        ```python
        kx.q.system.random_seed = 23
        ```
        """
        return self._q._call('\\S', wait=True)

    @random_seed.setter
    def random_seed(self, value):
        return self._q._call(f'\\S {int(value)}', wait=True)

    def variables(self, ns=None):
        """Get the variables in the current namespace or variables in a given namespace.

        Examples:

        Get the variables defined in the current namespace.

        ```python
        kx.q.system.variables()
        ```

        Get the variables associated with a q namespace/dictionary

        ```python
        kx.q.system.variables('.foo')
        kx.q.system.variables('foo')
        ```
        """
        if ns is not None:
            ns = str(ns)
            return self._q._call(f'\\v {ns}', wait=True)
        return self._q._call('\\v', wait=True)

    @property
    def workspace(self):
        """Outputs various information about the current memory usage of the `q` process.

        Examples:

        Get the memory usage of `EmbeddedQ`.

        ```python
        kx.q.system.workspace
        ```
        """
        return self._q._call('.Q.w[]', wait=True)

    @property
    def week_offset(self):
        """The start of week offset where 0 is Saturday.

        Examples:

        Get the current week offset.

        ```python
        kx.q.system.week_offset
        ```

        Set the current week offset so Monday is the first day of the week.

        ```python
        kx.q.system.week_offset = 2
        ```
        """
        return self._q._call('\\W', wait=True)

    @week_offset.setter
    def week_offset(self, value):
        return self._q._call(f'\\W {int(value)}', wait=True)

    @property
    def date_parsing(self):
        """Get the current format for date parsing.

        0 - is for `mm/dd/yyyy`
        1 - is for `dd/mm/yyyy`

        Examples:

        Get the current value for date parsing.

        ```python
        kx.q.system.date_parsing
        ```

        Get the current value for date parsing so the format is `dd/mm/yyyy`.

        ```python
        kx.q.system.date_parsing = 1
        ```
        """
        return self._q._call('\\z', wait=True)

    @date_parsing.setter
    def date_parsing(self, value):
        value = int(value)
        if value == 0 or value == 1:
            return self._q._call(f'\\z {value}', wait=True)
        raise ValueError('Date parsing mode can only be set to either 0 or 1.')

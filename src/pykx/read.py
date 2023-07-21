from pathlib import Path
from typing import List, Optional, Union

from . import licensed, wrappers as k
from .exceptions import LicenseException
from .ipc import QConnection


__all__ = [
    'QReader',
]


def __dir__():
    return __all__


JSONKTypes = Union[
    k.Table, k.Dictionary, k.BooleanAtom, k.BooleanVector, k.FloatAtom, k.FloatVector,
    k.CharVector, k.List
]


class QReader:
    """Read data using q."""

    def __init__(self, q):
        self._q = q

    def csv(self,
            path: Union[str, Path, k.SymbolAtom],
            types: Optional[Union[bytes, k.CharAtom, k.CharVector]] = None,
            delimiter: Union[str, bytes, k.CharAtom] = ',',
            as_table: Union[bool, k.BooleanAtom] = True,
    ) -> Union[k.Table, k.Dictionary]:
        """Reads a CSV file as a table or dictionary.

        Column types are guessed if not provided.

        Parameters:
            path: The path to the CSV file.
            types: A `str`-like object of uppercase characters representing the types. Space is
                used to drop a column. If `None`, the types will be guessed using `.csvutil.info`.
            delimiter: A single character representing the delimiter between values.
            as_table: `True` if the first line of the CSV file should be treated as column names,
                in which case a `pykx.Table` is returned. If `False` a `pykx.List` of
                `pykx.Vector` is returned - one for each column in the CSV file.

        Returns:
            The CSV data as a `pykx.Table` or `pykx.List`, depending on the value of `as_table`.

        See Also:
            [`q.write.csv`][pykx.write.QWriter.csv]

        Examples:

        Read a comma seperated CSV file into a `pykx.Table` guessing the datatypes of each
        column.

        ```python
        table = q.read.csv('example.csv')
        ```

        Read a tab seperated CSV file into a `pykx.Table` while specifying the columns
        datatypes to be a `pykx.SymbolVector` followed by two `pykx.LongVector` columns.

        ```python
        table = q.read.csv('example.csv', 'SJJ', '	')
        ```

        Read a comma seperated CSV file into a `pykx.Dictionary`, guessing the datatypes of
        each column.

        ```python
        table = q.read.csv('example.csv', None, None, False)
        ```
        """
        as_table = 'enlist' if as_table else ''
        if types is None:
            if not licensed:
                raise LicenseException('guess CSV column types')
            if isinstance(self._q, QConnection):
                raise ValueError('Cannot guess types of CSV columns over IPC.')
            types = self._q.csvutil.info(k.SymbolAtom(path))['t']
        elif isinstance(types, k.CharAtom):
            # Because of the conversion to `CharVector` later, converting the char atom to bytes
            # here essentially causes `types` to be enlisted without executing any q code.
            types = bytes(types)
        return self._q(
            f'{{(x; {as_table} y) 0: hsym z}}',
            k.CharVector(types),
            k.CharAtom(delimiter),
            k.SymbolAtom(path),
            wait=True,
        )

    def splayed(self,
                root: Union[str, Path, k.SymbolAtom],
                name: Union[str, k.SymbolAtom],
    ) -> k.SplayedTable:
        """Loads a splayed table.

        Parameters:
            root: The path to the root directory of the splayed table.
            name: The name of the table to read.

        Returns:
            The splayed table as a `pykx.SplayedTable`.

        See Also:
            [`q.write.splayed`][pykx.write.QWriter.splayed]

        Examples:

        Reads a splayed table named `t` found within the current directory

        ```python
        table = q.read.splayed('.', 't')
        ```

        Reads a splayed table named `splayed` found within the `/tmp` directory

        ```python
        table = q.read.splayed('/tmp', 'splayed')
        ```
        """
        return self._q(
            '{get hsym `$"/" sv string x,y}',
            k.SymbolAtom(Path(str(root)).as_posix()),
            k.SymbolAtom(name) + '/',
            wait=True,
        )

    def fixed(self,
              path: Union[str, Path, k.SymbolAtom],
              types: Union[bytes, k.CharVector],
              widths: Union[List[int], k.LongVector],
    ) -> k.List:
        """Loads a file of typed data with fixed-width fields.

        It is expected that there will either be a newline after every record, or none at all.

        Parameters:
            path: The path to the file containing the fixed-width field data.
            types: A string of uppercase characters representing the types. Space is used to
                drop a column.
            widths: The width in bytes of each field.

        Returns:
            The data as a `pykx.List` with a `pykx.Vector` for each column.

        Examples:

        Read a file of fixed width data into a `pykx.List` of two `pykx.LongVectors` the first
        with a size of 1 character and the second with a size of 2 characters.

        ```python
        data = q.read.fixed('example_file', [b'J', b'J'], [1, 2])
        ```
        """

        if isinstance(types, bytes):
            types = types.decode('utf-8')
        return self._q(
            '{(x;y) 0: z}',
            k.CharVector(types),
            k.LongVector(widths),
            k.SymbolAtom(path),
            wait=True,
        )

    def json(self, path: Union[str, Path, k.SymbolAtom]) -> JSONKTypes:
        """Reads a JSON file into a k.Table.

        Parameters:
            path: The path to the JSON file.

        Returns:
            The JSON data as a `pykx.K` object.

        See Also:
            [`q.write.json`][pykx.write.QWriter.json]

        Examples:

        Read a JSON file.

        ```python
        data = q.read.json('example.json')
        ```
        """
        return self._q('{.j.k raze read0 x}', k.SymbolAtom(path), wait=True)

    def serialized(self, path: Union[str, Path, k.SymbolAtom]) -> k.K:
        """Reads a binary file containing serialized q data.

        Parameters:
            path: The path to the q data file.

        Returns:
            The q data file converted to a `pykx` object.

        See Also:
            [`q.write.serialized`][pykx.write.QWriter.serialized]

        Examples:

        Read a q data file containing a serialized table into a `pykx.Table` object.

        ```python
        table = q.read.serialized('q_table_file')
        ```
        """
        return self._q('{get hsym x}', k.SymbolAtom(path), wait=True)

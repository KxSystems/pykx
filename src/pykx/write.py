from typing import Any, Optional, Union
from pathlib import Path

import pandas as pd

from . import wrappers as k

__all__ = [
    'QWriter',
]


def __dir__():
    return __all__


class QWriter:
    """Write data using q."""

    def __init__(self, q):
        self._q = q

    def splayed(self,
                root: Union[str, Path, k.SymbolAtom],
                name: Union[str, k.SymbolAtom],
                table: Union[k.Table, pd.DataFrame],
    ) -> Path:
        """Splays and writes a q table to disk.

        Parameters:
            root: The path to the root directory of the splayed table.
            name: The name of the table, which will be written to disk.
            table: A table-like object to be written as a splayed table.

        Returns:
            The path to the splayed table on disk.

        See Also:
            [`q.read.splayed`][pykx.read.QReader.splayed]

        Examples:

        Write a pandas `DataFrame` to disk as a splayed table in the current directory.

        ```python
        df = pd.DataFrame([[x, 2 * x] for x in range(5)])
        q.write.splayed('.', 'splayed_table', df)
        ```

        Write a `pykx.Table` to disk as a splayed table at `/tmp/splayed_table`.

        ```python
        table = q('([] a: 10 20 30 40; b: 114 113 98 121)')
        q.write.splayed('/tmp', 'splayed_table', table)
        ```
        """
        return Path(str(self._q._call(
            '{(hsym `$"/" sv string x,y) set .Q.en[hsym x;] z}',
            k.SymbolAtom(root),
            k.SymbolAtom(name) + '/',
            table,
            wait=True,
        ))[1:]) # Remove the leading ':'

    def serialized(self,
                   path: Union[str, Path, k.SymbolAtom],
                   data: Any,
    ) -> Path:
        """Writes a q object to a binary data file using q serialization.

        This method is a wrapper around the q function `set`, and as with any q function, arguments
        which are not `pykx.K` objects are automatically converted into them.

        Parameters:
            path: The path to write the q object to.
            data: An object that will be converted to q, then serialized to disk.

        Returns:
            A `pykx.SymbolAtom` that can be used as a file descriptor for the file.

        See Also:
            [`q.read.serialized`][pykx.read.QReader.serialized]

        Examples:

        Serialize and write a `pandas.DataFrame` to disk in the current directory.

        ```python
        df = q('([] a: til 5; b: 2 * til 5)').pd()
        q.write.serialized('serialized_table', df)
        ```

        Serialize and write a Python `int` to disk in the current directory.

        ```python
        q.write.serialized('serialized_int', 145)
        ```
        """
        return Path(str(self._q._call(
            '{hsym[x] set y}',
            k.SymbolAtom(path),
            data,
            wait=True,
        ))[1:]) # Remove the leading ':'

    def csv(self,
            path: Union[str, Path, k.SymbolAtom],
            table: Union[k.Table, pd.DataFrame],
            delimiter: Optional[Union[str, bytes, k.CharAtom]] = ',',
    ) -> Path:
        """Writes a given table to a CSV file.

        Parameters:
            path: The path to the CSV file.
            delimiter: A single character representing the delimeter between values.
            table: A table like object to be written as a csv file.

        Returns:
            A `pykx.SymbolAtom` that can be used as a file descriptor for the file.

        See Also:
            [`q.read.csv`][pykx.read.QReader.csv]

        Examples:

        Write a pandas `DataFrame` to disk as a csv file in the current directory using a
        comma as a seperator between values.

        ```python
        df = q('([] a: til 5; b: 2 * til 5)').pd()
        q.write.csv('example.csv', df)
        ```

        Write a `pykx.Table` to disk as a csv file in the current directory using a tab as a
        seperator between values.

        ```python
        table = q('([] a: 10 20 30 40; b: 114 113 98 121)')
        q.write.csv('example.csv', table, '	')
        ```
        """
        delimiter = ',' if delimiter is None else delimiter
        return Path(str(self._q._call(
            '{hsym[x] 0: y 0: z}',
            k.SymbolAtom(path),
            k.CharAtom(delimiter),
            table,
            wait=True,
        ))[1:]) # Remove the leading ':'

    def json(self,
             path: Union[str, Path, k.SymbolAtom],
             data: Any,
    ) -> Path:
        """Writes a JSON representation of the given q object to a file.

        Parameters:
            path: The path to the JSON file.
            data: Any type to be serialized and written as a JSON file.

        Returns:
            A `pykx.SymbolAtom` that can be used as a file descriptor for the file.

        See Also:
            [`q.read.json`][pykx.read.QReader.json]

        Examples:

        Convert a pandas `Dataframe` to JSON and then write it to disk in the current
        directory.

        ```python
        df = q('([] a: til 5; b: 2 * til 5)').pd()
        q.write.json('example.json', df)
        ```

        Convert a Python `int` to JSON and then write it to disk in the current directory.

        ```python
        q.write.json('example.json', 143)
        ```

        Convert a Python `dictionary` to JSON and then write it to disk in the current
        directory.

        ```python
        dictionary = {'a': 'hello', 'b':'pykx', 'c':2.71}
        q.write.json('example.json', dictionary)
        ```
        """
        return Path(str(self._q._call(
            '{hsym[x] 0: enlist .j.j y}',
            k.SymbolAtom(path),
            data,
            wait=True,
        ))[1:]) # Remove leading ':'

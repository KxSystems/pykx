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


_type_mapping = {
    'List': "*",
    'GUID': "G",
    'Boolean': "B",
    'Byte': "X",
    'Short': "H",
    'Int': "I",
    'Long': "J",
    'Real': "E",
    'Float': "F",
    'Char': "C",
    'Symbol': "S",
    'Timestamp': "P",
    'Month': "M",
    'Date': "D",
    'Datetime': "Z",
    'Timespan': "N",
    'Minute': "U",
    'Second': "V",
    'Time': "T",
}

_filter_types = [None, "basic", "only", "like"]

JSONKTypes = Union[
    k.Table, k.Dictionary, k.BooleanAtom, k.BooleanVector, k.FloatAtom, k.FloatVector,
    k.CharVector, k.List
]


class QReader:
    """Read data using q."""

    def __init__(self, q):
        self._q = q

    def update_types(self, types):
        upd_types = []
        for i in types:
            t = str(i)
            if len(t) == 1:
                upd_types.append(i)
            else:
                found = False
                for key in _type_mapping.keys():
                    if key in t:
                        found = True
                        upd_types.append(_type_mapping[key])
                        break
                if not found:
                    raise TypeError(f'Unsupported type: {type(i)} supplied')
        return upd_types

    def csv(self,
            path: Union[str, Path, k.SymbolAtom],
            types: Optional[Union[bytes, k.CharAtom, k.CharVector]] = None,
            delimiter: Union[str, bytes, k.CharAtom] = ',',
            as_table: Union[bool, k.BooleanAtom] = True,
            filter_type: Union[str, k.CharVector] = None,
            filter_columns: Union[str, list, k.CharVector, k.SymbolAtom, k.SymbolVector] = None,
            custom: dict = None,
    ) -> Union[k.Table, k.Dictionary]:
        """Reads a CSV file as a table or dictionary.

        Column types are guessed if not provided.

        Parameters:
            path: The path to the CSV file.
            types: Can be a dictionary of columns and their types or a `str`-like object of
                uppercase characters representing the types. Space is used to drop a column.
                If `None`, the types will be guessed using [csvutil.q](https://github.com/KxSystems/kdb/blob/master/utils/csvutil.q).
                A breakdown of this process is illustrated in the table below.
            delimiter: A single character representing the delimiter between values.
            as_table: `True` if the first line of the CSV file should be treated as column names,
                in which case a `pykx.Table` is returned. If `False` a `pykx.List` of
                `pykx.Vector` is returned - one for each column in the CSV file.
            filter_type: Can be `basic`, `only`, or `like`. `basic` will not search for
                any types with the `extended` flag in [csvutil.q]. `only` will only process
                columns that are passed in `filter_columns`. `like` will only process columns that match
                a string pattern passed in `filter_columns`.
            filter_columns: Used in tandem with `filter_type` when `only` or `like` is passed.
                `only` accepts str or list of str. `like` accepts only a str pattern.
            custom: A dictionary used to change default values in [csvutil.q](https://github.com/KxSystems/pykx/blob/main/src/pykx/lib/csvutil.q#L34).

        Returns:
            The CSV data as a `pykx.Table` or `pykx.List`, depending on the value of `as_table`.

        See Also:
            [`q.write.csv`][pykx.write.QWriter.csv]


        CSV Type Guessing Table:
            | Type Character | Type  | Condition(s)   |
            |---|---|---|
            | *  | List  |- Any type of width greater than 30.<br>- Remaining unknown types. |
            | B  | BooleanAtom  |- Matching Byte or Char, maxwidth 1, no decimal points, at least 1 of `[0fFnN]` and 1 of `[1tTyY]` in columns.<br>- Matching Byte or Char, maxwidth 1, no decimal points, all elements in `[01tTfFyYnN]`.   |
            | G  | GUIDAtom  |- Matches GUID-like structure.<br>- Matches structure wrapped in `{ }`.  |
            | X  | ByteAtom  |- Maxwidth of 2, comprised of `[0-9]` AND `[abcdefABCDEF]`.  |
            | H  | ShortAtom   |- Matches Integer with maxwidth less than 7. |
            | I  | IntAtom   |- Numerical of size between 7 and 15 with exactly 3 decimal points (IP Address).<br>- Matches Long with maxwidth less than 12. |
            | J  | LongAtom  |- Numerical, no decimal points, all elements `+-` or `0-9`. |
            | E  | RealAtom |- Matches float with maxwidth less than 9.  |
            | F  | FloatAtom |-  Numerical, maxwidth greater than 2, fewer than 2 decimal points, `/` present.<br>- Numerical, fewer than 2 decimal points, maxwidth greater than 1.  |
            | C  | CharAtom |- Empty columns. Remaining unknown types of size 1.  |
            | S  | SymbolAtom  |- Remaining unknown types of maxwidth 2-11 and granularity of less than 10. |
            | P  | TimestampAtom  |- Numerical, maxwidth 11-29, fewer than 4 decimals matching `YYYY[./-]MM[./-]DD` |
            | M  | MonthAtom   |- Matching either numerical, Int, Byte, Real or Float, fewer than 2 decimal points, maxwidth 4-7 |
            | D  | DateAtom |- Matching Integer, maxwidth 6 or 8.<br>- Numerical, 0 decimal points, maxwidth 8-10.<br>- Numerical, 2 decimal points, maxwidth 8-10.<br>- No decimal points maxwidth 5-9, matching date with 3 letter month code eg.(9nov1989). |
            | N  | TimespanAtom  |- Numerical, maxwidth 15, no decimal points, all values `0-9`.<br>- Numerical, maxwidth 3-29, 1 decimal point, matching `*[0-9]D[0-9]*`.<br>- Numerical, maxwidth 3-28, 1 decimal point.   |
            | U  | MinuteAtom |- Matching Byte, maxwidth 4, matching `[012][0-9][0-5][0-9]`.<br>- Numerical, maxwidth 4 or 5, no decimal points, matching `*[0-9]:[0-5][0-9]`.  |
            | V  | SecondAtom |- Matching Integer, maxwidth 6, matching `[012][0-9][0-5][0-9][0-5][0-9]`.<br>- Matching Time, maxwidth 7 or 8, no decimal points. |
            | T  | TimeAtom  |- Numerical, maxwidth 9, no decimal points, all values numeric.<br>- Numerical, maxwidth 7 - 12, fewer than 2 decimal points, matching  `[0-9]:[0-5][0-9]:[0-5][0-9]`.<br>- Matching Real or Float, maxwidth 7-12, 1 decimal point, matching `[0-9][0-5][0-9][0-5][0-9]`.   |

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

        Read a comma separated CSV file into a `pykx.Dictionary`, guessing the datatypes of
        each column.

        ```python
        table = q.read.csv('example.csv', None, None, False)
        ```

        Read a comma separated CSV file specifying the type of the three columns
        named `x1`, `x2` and `x3` to be of type `Integer`, `GUID` and `Timestamp`.

        ```python
        table = q.read.csv('example.csv', {'x1':kx.IntAtom,'x2':kx.GUIDAtom,'x3':kx.TimestampAtom})
        ```

        Read a comma separated CSV file specifying only columns that include the word "name" in them.

        ```python
        table = q.read.csv('example.csv', filter_type = "like", filter_columns = '*name*')
        ```

        Read a comma separated CSV file changing the guessing variables to change the number of lines
        read and used to guess the type of the column.

        ```python
        table = q.read.csv('example.csv', custom = {"READLINES":1000})
        ```
        """ # noqa: E501
        as_table = 'enlist' if as_table else ''
        dict_conversion = None
        if types is None or isinstance(types, dict):
            if not licensed:
                raise LicenseException('guess CSV column types')
            if isinstance(self._q, QConnection):
                raise ValueError('Cannot guess types of CSV columns over IPC.')
            if filter_type not in _filter_types:
                raise ValueError(f'Filter type {filter_type} not in supported filter types.')
            self._q.csvutil
            cache = self._q('.csvutil')
            try:
                if custom is not None:
                    self._q('''{
                        if[0<count i:where not (r:type each value x)=e:type each .csvutil[key x];
                        \'"csvutil setting type incorrect for:",.Q.s1[key[x]i]," received: ",.Q.s1[r i]," expected: ",.Q.s1[e i]]
                        }''', custom) # noqa: E501
                    for attr, val in custom.items():
                        setattr(self._q.csvutil, attr, val)
                if isinstance(types, dict):
                    dict_conversion = types
                    types = None
                if filter_type == "basic":
                    types = self._q.csvutil.basicinfo(k.SymbolAtom(path))['t']
                elif filter_type == "only":
                    types = self._q.csvutil.infoonly(k.SymbolAtom(path), filter_columns)['t']
                elif filter_type == "like":
                    types = self._q.csvutil.infolike(k.SymbolAtom(path),
                                                     k.CharVector(filter_columns))['t']
                else:
                    types = self._q.csvutil.info(k.SymbolAtom(path))['t']
            finally:
                self._q['.csvutil'] = cache
        elif isinstance(types, k.CharAtom):
            # Because of the conversion to `CharVector` later, converting the char atom to bytes
            # here essentially causes `types` to be enlisted without executing any q code.
            types = bytes(types)
        elif isinstance(types, list):
            types = self.update_types(types)
        res = self._q(
            f'{{(x; {as_table} y) 0: hsym z}}',
            k.CharVector(types),
            k.CharAtom(delimiter),
            k.SymbolAtom(path),
            wait=True,
        )
        if dict_conversion is not None:
            return res.astype(dict_conversion)
        return res

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

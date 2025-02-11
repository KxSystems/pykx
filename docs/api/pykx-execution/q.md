---
title: PyKX q functions and operators
description: PyKX implementation of a subset of the q language's functions and operators
author: KX Systems
date: September 2024
tags: operators
---
# q functions and operators

_This page documents the PyKX implementations of a selection of keywords and operators available in q._

The functions listed here are accessible in PyKX as attributes of `#!python pykx.q`, or as attributes of `#!python pykx.QConnection` instances. Refer to [the q reference card in the q docs](https://code.kx.com/q/ref/#by-category) for more details about these functions as they are used in a q process. This page documents using them in Python via PyKX.

These functions take and return q objects, which are wrapped in PyKX as `#!python pykx.K` objects. Any arguments of other types are converted appropriately. Refer to [the PyKX wrappers documentation](../pykx-q-data/wrappers.md) for more information about `#!python pykx.K` objects.

## By Category

Category                    | Elements
--------------------------- | -----------------------------------------------------------------------------------------
[Environment](#environment) | [`getenv`](#getenv), [`gtime`](#gtime), [`ltime`](#ltime), [`setenv`](#setenv)
[Interpret](#interpret)     | [`eval`](#eval), [`parse`](#parse), [`reval`](#reval), [`show`](#show), [`system`](#system), [`value`](#value)
[IO](#io)                   | [`dsave`](#dsave), [`get`](#get), [`hclose`](#hclose), [`hcount`](#hcount), [`hdel`](#hdel), [`hopen`](#hopen), [`hsym`](#hsym), [`load`](#load), [`read0`](#read0), [`read1`](#read1), [`rload`](#rload), [`rsave`](#rsave), [`save`](#save), [`set`](#set)
[Iterate](#iterate)         | [`each`](#each), [`over`](#over), [`peach`](#peach), [`prior`](#prior), [`scan`](#scan)
[Join](#join)               | [`aj`](#aj), [`aj0`](#aj0), [`ajf`](#ajf), [`ajf0`](#ajf0), [`asof`](#asof), [`ej`](#ej), [`ij`](#ij), [`ijf`](#ijf), [`lj`](#lj), [`ljf`](#ljf), [`pj`](#pj), [`uj`](#uj), [`ujf`](#ujf), [`wj`](#wj), [`wj1`](#wj1)
[List](#list)               | [`count`](#count), [`cross`](#cross), [`cut`](#cut), [`enlist`](#enlist), [`fills`](#fills), [`first`](#first), [`flip`](#flip), [`group`](#group), [`inter`](#inter), [`last`](#last), [`mcount`](#mcount), [`next`](#next), [`prev`](#prev), [`raze`](#raze), [`reverse`](#reverse), [`rotate`](#rotate), [`sublist`](#sublist), [`sv`](#sv), [`til`](#til), [`union`](#union), [`vs`](#vs), [`where`](#where), [`xprev`](#xprev)
[Logic](#logic)             | [`all`](#all), [`any`](#any)
[Math](#math)               | [`abs`](#abs), [`acos`](#acos), [`asin`](#asin), [`atan`](#atan), [`avg`](#avg), [`avgs`](#avgs), [`ceiling`](#ceiling), [`cor`](#cor), [`cos`](#cos), [`cov`](#cov), [`deltas`](#deltas), [`dev`](#dev), [`div`](#div), [`ema`](#ema), [`exp`](#exp), [`floor`](#floor), [`inv`](#inv), [`log`](#log), [`lsq`](#lsq), [`mavg`](#mavg), [`max`](#max), [`maxs`](#maxs), [`mdev`](#mdev), [`med`](#med), [`min`](#min), [`mins`](#mins), [`mmax`](#mmax), [`mmin`](#mmin), [`mmu`](#mmu), [`mod`](#mod), [`msum`](#msum), [`neg`](#neg), [`prd`](#prd), [`prds`](#prds), [`rand`](#rand), [`ratios`](#ratios), [`reciprocal`](#reciprocal), [`scov`](#scov), [`sdev`](#sdev), [`signum`](#signum), [`sin`](#sin), [`sqrt`](#sqrt), [`sum`](#sum), [`sums`](#sums), [`svar`](#svar), [`tan`](#tan), [`var`](#var), [`wavg`](#wavg), [`within`](#within), [`wsum`](#wsum), [`xexp`](#xexp), [`xlog`](#xlog)
[Meta](#meta)               | [`attr`](#attr), [`null`](#null), [`tables`](#tables), [`type`](#type), [`view`](#view), [`views`](#views)
[Query](#queries)           | [`fby`](#fby)
[Sort](#sort)               | [`asc`](#asc), [`bin`](#bin), [`binr`](#binr), [`desc`](#desc), [`differ`](#differ), [`distinct`](#distinct), [`iasc`](#iasc), [`idesc`](#idesc), [`rank`](#rank), [`xbar`](#xbar), [`xrank`](#xrank)
[Table](#table)             | [`cols`](#cols), [`csv`](#csv), [`fkeys`](#fkeys), [`insert`](#insert), [`key`](#key), [`keys`](#keys), [`meta`](#meta), [`ungroup`](#ungroup), [`upsert`](#upsert), [`xasc`](#xasc), [`xcol`](#xcol), [`xcols`](#xcols), [`xdesc`](#xdesc), [`xgroup`](#xgroup), [`xkey`](#xkey)
[Text](#text)               | [`like`](#like), [`lower`](#lower), [`ltrim`](#ltrim), [`md5`](#md5), [`rtrim`](#rtrim), [`ss`](#ss), [`ssr`](#ssr), [`string`](#string), [`trim`](#trim), [`upper`](#upper)
[Operators](#operators)     | [`drop`](#drop), [`coalesce`](#coalesce), [`fill`](#fill), [`take`](#take), [`set_attribute`](#set_attribute), [`join`](#join), [`find`](#find), [`enum_extend`](#enum_extend), [`roll`](#roll), [`deal`](#deal), [`dict`](#dict), [`enkey`](#enkey), [`unkey`](#unkey), [`enumeration`](#enumeration), [`enumerate`](#enumerate), [`pad`](#pad), [`cast`](#cast), [`tok`](#tok), [`compose`](#compose)

Some keywords listed on [the q reference card](https://code.kx.com/q/ref/#by-category) are unavailable in this API:

 - `#!q select`, `#!q exec`, `#!q update` and `#!q delete` are not q functions, but a part of the q language itself

 - functions that have names which would result in syntax errors in Python, such as `#!q not` and `#!q or`

The unavailable functions can still be used in PyKX by executing q code with `#!python pykx.q`, i.e. `#!python pykx.q('not')` instead of `#!python pykx.q.not`. For the qSQL functions (`#!q select`, `#!q exec`, `#!q update`, and `#!q delete`) use [PyKX qSQL](../query.md).

## Environment

### [getenv](https://code.kx.com/q/ref/getenv/)

Get the value of an environment variable.

```python
>>> pykx.q.getenv('EDITOR')
pykx.CharVector(q('"nvim"'))
```

### [gtime](https://code.kx.com/q/ref/gtime/)

UTC equivalent of local timestamp.

```python
>>> import datetime
>>> pykx.q.gtime(datetime.datetime.fromisoformat('2022-05-22T12:23:45.123'))
pykx.TimestampAtom(q('2022.05.22D16:23:45.123000000'))
```

### [ltime](https://code.kx.com/q/ref/gtime/#ltime)

Local equivalent of UTC timestamp.

```python
>>> import datetime
>>> pykx.q.ltime(datetime.datetime.fromisoformat('2022-05-22T12:23:45.123'))
pykx.TimestampAtom(q('2022.05.22D08:23:45.123000000'))

```

### [setenv](https://code.kx.com/q/ref/getenv/#setenv)

Set the value of an environment variable.

```python
>>> pykx.q.setenv('RTMP', b'/home/user/temp')
>>> pykx.q.getenv('RTMP')
pykx.CharVector(q('"/home/user/temp"'))
```

## Interpret

### [eval](https://code.kx.com/q/ref/eval/)

Evaluate parse trees.

```python
>>> pykx.q.eval([pykx.q('+'), 2, 3])
pykx.LongAtom(q('5'))
```

### [parse](https://code.kx.com/q/ref/parse/)

Parse a char vector into a parse tree, which can be evaluated with [`pykx.q.eval`](#eval).

```python
>>> pykx.q.parse(b'{x * x}')
pykx.Lambda(q('{x * x}'))
>>> pykx.q.parse(b'2 + 3')
pykx.List(pykx.q('
+
2
3
'))
```

### [reval](https://code.kx.com/q/ref/eval/#reval)

Restricted evaluation of a parse tree.

Behaves similar to [`eval`](#eval) except the evaluation is blocked from modifying state  for any handle context other than 0.

```python
>>> pykx.q.reval(pykx.q.parse(b'til 10'))
pykx.LongVector(q('0 1 2 3 4 5 6 7 8 9'))
```

### [show](https://code.kx.com/q/ref/show/)

Print the string representation of the given q object.

Note: `show` bypasses typical Python output redirection.
    The q function `show` prints directly to file descriptor 1, so typical Python output redirection methods, e.g. [`contextlib.redirect_stdout`](https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout), will not affect it.

```python
>>> pykx.q.show(range(5))
0
1
2
3
4
pykx.Identity(q('::'))
```

### [system](https://code.kx.com/q/ref/system/)

Execute a system command.

Where x is a string representing a [system command](https://code.kx.com/q/basics/syscmds/) and any parameters to it, executes the command and returns any result.

```python
>>> pykx.q.system(b'pwd')
pykx.List(q('"/home/user"'))
```

### [value](https://code.kx.com/q/ref/value/)

Returns the value of x.

| Input Type       | Output Type                                |
|------------------|--------------------------------------------|
| dictionary       | value of the dictionary                    |
| symbol atom      | value of the variable it names             |
| enumeration      | corresponding symbol vector                |
| string           | result of evaluating it in current context |
| list             | result of evaluating list as a parse tree  |
| projection       | list: function followed by argument/s      |
| composition      | list of composed values                    |
| derived function | argument of the iterator                   |
| operator         | internal code                              |
| view             | list of metadata                           |
| lambda           | structure                                  |
| file symbol      | content of datafile                        |

```python
>>> pykx.q.value(pykx.q('`q`w`e!(1 2; 3 4; 5 6)'))
pykx.List(q('
1 2
3 4
5 6
'))
```

## IO

### [dsave](https://code.kx.com/q/ref/dsave/)

Write global tables to disk as splayed, enumerated, indexed q tables.

```python
>>> from pathlib import Path
>>> pykx.q['t'] = kx.Table(data={'x': [1, 2, 3], 'y': [10, 20, 30]})
>>> pykx.q.dsave(Path('v'), 't')
pykx.SymbolAtom(q('`t'))
```

### [get](https://code.kx.com/q/ref/get/)

Read or memory-map a variable or q data file.

```python
>>> pykx.q['a'] = 10
>>> pykx.q.get('a')
pykx.LongAtom(q('10'))
```

### [hclose](https://code.kx.com/q/ref/hopen/#hclose)

Where x is a connection handle, closes the connection, and destroys the handle.
```python
>>> pykx.q.hclose(pykx.q('3i'))
```
### [hcount](https://code.kx.com/q/ref/hcount/)

Size of a file in bytes.
```python
>>> pykx.q.hcount('example.txt')
pykx.LongAtom(q('11'))
```
### [hdel](https://code.kx.com/q/ref/hdel/)

Where `x` is a [file symbol atom](#hsym), deletes the file or folder (if empty), and returns `x`.

```python
>>> pykx.q.hdel('example.txt')
```

### [hopen](https://code.kx.com/q/ref/hopen/)

Open a connection to a file or process.

```python
>>> pykx.q.hopen('example.txt')
pykx.IntAtom(q('3i'))
```

### [hsym](https://code.kx.com/q/ref/hsym/)

Convert symbols to handle symbols, which can be used for I/O as file descriptors or handles.

```python
>>> pykx.q.hsym('10.43.23.197')
pykx.SymbolAtom(q('`:10.43.23.197'))
```

### [load](https://code.kx.com/q/ref/load/)

Load binary data from a file.

```python
>>> pykx.q['t'] = pykx.Table([[1, 10], [2, 20], [3, 30]], columns=['x', 'y'])
>>> pykx.q('t')
pykx.Table(pykx.q('
x y
----
1 10
2 20
3 30
'))
>>> pykx.q.save('t') # Save t to disk
pykx.SymbolAtom(pykx.q('`:t'))
>>> pykx.q('delete t from `.') # Delete t from memory
pykx.SymbolAtom(pykx.q('`.'))
>>> pykx.q('t') # t is not longer defined
Traceback (most recent call last):
pykx.exceptions.QError: t
>>> pykx.q.load('t') # Load t from disk
pykx.SymbolAtom(pykx.q('`t'))
>>> pykx.q('t')
pykx.Table(pykx.q('
x y
----
1 10
2 20
3 30
'))
```

### [read0](https://code.kx.com/q/ref/read0/)

Read text from a file or process handle.

```python
>>> pykx.q.read0('example.txt')
pykx.List(q('
"Hello"
"World"
'))
```

### [read1](https://code.kx.com/q/ref/read1/)

Read bytes from a file or named pipe.

```python
>>> pykx.q.read1('example.txt')
pykx.ByteVector(q('0x48656c6c6f0a576f726c64'))
```

### [rload](https://code.kx.com/q/ref/load/#rload)

Load a splayed table from a directory.

```python
>>> pykx.q.rload('t')
>>> pykx.q('t')
pykx.Table(q('
x y
----
1 10
2 20
3 30
'))
```

### [rsave](https://code.kx.com/q/ref/save/#rsave)

Write a table splayed to a directory.

```python
>>> pykx.q['t'] = pykx.Table([[1, 10], [2, 20], [3, 30]])
>>> pykx.q.rsave('t')
pykx.SymbolAtom(q('`:t/'))
```

### [save](https://code.kx.com/q/ref/save/)

Write global data to file or splayed to a directory.

```python
>>> pykx.q['t'] = pykx.Table([[1, 10], [2, 20], [3, 30]])
>>> pykx.q.save('t')
pykx.SymbolAtom(q('`:t'))
```

### [set](https://code.kx.com/q/ref/get/#set)

Assign a value to a global variable.

Persist an object as a file or directory.

| Types                               | Result                             |
|-------------------------------------|------------------------------------|
| pykx.q.set(nam, y)                  | set global `nam` to `y`                |
| pykx.q.set(fil, y)                  | write `y` to a file                  |
| pykx.q.set(dir, y)                  | splay `y` to a directory             |
| pykx.q.set([fil, lbs, alg, lvl], y) | write `y` to a file, compressed      |
| pykx.q.set([dir, lbs, alg, lvl], y) | splay `y` to a directory, compressed |
| pykx.q.set([dir, dic], y)           | splay `y` to a directory, compressed |

Where

| Abbreviation | K type       | Explanation                 |
|--------------|--------------|-----------------------------|
| alg          | integer atom | compression algorithm       |
| dic          | dictionary   | compression specifications  |
| dir          | filesymbol   | directory in the filesystem |
| fil          | filesymbol   | file in the filesystem      |
| lbs          | integer atom | logical block size          |
| lvl          | integer atom | compression level           |
| nam          | symbol atom  | valid q name                |
| t            | table        |                             |
| y            | (any)        | any q object                |

[Compression parameters alg, lbs, and lvl](https://code.kx.com/q/kb/file-compression/#parameters)

[Compression specification dictionary](https://code.kx.com/q/ref/get/#compression)

```python
>>> pykx.q.set('a', 42)
pykx.SymbolAtom(q('`a'))
>>> pykx.q('a')
pykx.LongAtom(q('42'))
```

## Iterate

### [each](https://code.kx.com/q/ref/each/)

Iterate over list and apply a function to each element.

```python
>>> pykx.q.each(pykx.q.count, [b'Tis', b'but', b'a', b'scratch'])
pykx.LongVector(q('3 3 1 7'))
>>> pykx.q.each(pykx.q.sums, [[2, 3, 4], [[5, 6], [7, 8]], [9, 10, 11, 12]])
pykx.List(q('
2 5 9
((5;6);12 14)
9 19 30 42
'))
```

### [over](https://code.kx.com/q/ref/over/)

The keywords over and [`scan`](#scan) are covers for the accumulating iterators, Over and Scan. It is good style to use over and scan with unary and binary values.

Just as with Over and Scan, over and scan share the same syntax and perform the same computation; but while scan returns the result of each evaluation, over returns only the last.

```python
>>> pykx.q.over(pykx.q('*'), [1, 2, 3, 4, 5])
pykx.LongAtom(q('120'))
```

### [peach](https://code.kx.com/q/ref/each/)

[`each`](#each) and peach perform the same computation and return the same result, but peach will parallelize the work across available threads.

```python
>>> pykx.q.peach(pykx.q.count, [b'Tis', b'but', b'a', b'scratch'])
pykx.LongVector(q('3 3 1 7'))
>>> pykx.q.peach(pykx.q.sums, [[2, 3, 4], [[5, 6], [7, 8]], [9, 10, 11, 12]])
pykx.List(q('
2 5 9
((5;6);12 14)
9 19 30 42
'))
```

### [prior](https://code.kx.com/q/ref/prior/)

Applies a function to each item of `x` and the item preceding it, and returns a result of the same length.

```python
>>> pykx.q.prior(pykx.q('+'), [1, 2, 3, 4, 5])
pykx.LongVector(pykx.q('1 3 5 7 9'))
>>> pykx.q.prior(lambda x, y: x + y, pykx.LongVector([1, 2, 3, 4, 5]))
pykx.LongVector(pykx.q('0N 3 5 7 9'))
```

### [scan](https://code.kx.com/q/ref/over/)

The keywords [over](#over) and scan are covers for the accumulating iterators, Over and Scan. It is good style to use over and scan with unary and binary values.

Just as with Over and Scan, over and scan share the same syntax and perform the same computation; but while scan returns the result of each evaluation, over returns only the last.

```python
>>> pykx.q.scan(pykx.q('+'), [1, 2, 3, 4, 5])
pykx.LongVector(q('1 3 6 10 15'))
```

## Join

### [aj](https://code.kx.com/q/ref/aj/)

Performs an as-of join across temporal columns in tables. Returns a table with records from the left-join of the first table and the second table. For each record in the first table, it is matched with the second table over the columns specified in the first input parameter and if there is a match  the most recent match will be joined to the record.

The resulting time column is the value of the boundary used in the first table.

```python
>>> import pandas as pd
>>> import numpy as np
>>> df1 = pd.DataFrame({
...    'time': np.array([36061, 36063, 36064], dtype='timedelta64[s]'),
...    'sym': ['msft', 'ibm', 'ge'], 'qty': [100, 200, 150]
... })
>>> df2 = pd.DataFrame({
...     'time': np.array([36060, 36060, 36060, 36062], dtype='timedelta64[s]'),
...     'sym': ['ibm', 'msft', 'msft', 'ibm'], 'qty': [100, 99, 101, 98]
... })
>>> pykx.q.aj(pykx.SymbolVector(['sym', 'time']), df1, df2)
pykx.Table(q('
time                 sym  qty
-----------------------------
0D10:01:01.000000000 msft 101
0D10:01:03.000000000 ibm  98
0D10:01:04.000000000 ge   150
'))
```

### [aj0](https://code.kx.com/q/ref/aj/)

Performs an as-of join across temporal columns in tables. Returns a table with records from the left-join of the first table and the second table. For each record in the first table, it is matched with the second table over the columns specified in the first input parameter and if there is a match  the most recent match will be joined to the record.

The resulting time column is the actual time of the last value in the second table.

```python
>>> import pandas as pd
>>> import numpy as np
>>> df1 = pd.DataFrame({
...     'time': np.array([36061, 36063, 36064], dtype='timedelta64[s]'),
...     'sym': ['msft', 'ibm', 'ge'], 'qty': [100, 200, 150]
... })
>>> df2 = pd.DataFrame({
...     'time': np.array([36060, 36060, 36060, 36062], dtype='timedelta64[s]'),
...     'sym': ['ibm', 'msft', 'msft', 'ibm'], 'qty': [100, 99, 101, 98]
... })
>>> pykx.q.aj0(pykx.SymbolVector(['sym', 'time']), df1, df2)
pykx.Table(q('
time                 sym  qty
-----------------------------
0D10:01:00.000000000 msft 101
0D10:01:02.000000000 ibm  98
0D10:01:04.000000000 ge   150
'))
```

### [ajf](https://code.kx.com/q/ref/aj/)

Performs an as-of join across temporal columns in tables with null values being filled. Returns a table with records from the left-join of the first table and the second table. For each record in the first table, it is matched with the second table over the columns specified in the first input parameter and if there is a match  the most recent match will be joined to the record.

The resulting time column is the value of the boundary used in the first table.

```python
>>> import pandas as pd
>>> import numpy as np
>>> df1 = pd.DataFrame({
...     'time': np.array([1, 1], dtype='timedelta64[s]'),
...     'sym': ['a', 'b'],
...     'p': pykx.LongVector([0, 1]),
...     'n': ['r', 's']
... })
>>> df2 = pd.DataFrame({
...     'time': np.array([1, 1], dtype='timedelta64[s]'),
...     'sym':['a', 'b'],
...     'p': pykx.q('1 0N')
... })
>>> pykx.q.ajf(pykx.SymbolVector(['sym', 'time']), df1, df2)
pykx.Table(q('
time                 sym p n
----------------------------
0D00:00:01.000000000 a   1 r
0D00:00:01.000000000 b   1 s
'))
```

### [ajf0](https://code.kx.com/q/ref/aj/)

Performs an as-of join across temporal columns in tables with null values being filled. Returns a table with records from the left-join of the first table and the second table. For each record in the first table, it is matched with the second table over the columns specified in the first input parameter and if there is a match  the most recent match will be joined to the record.

The resulting time column is the actual time of the last value in the second table.

```python
>>> import pandas as pd
>>> import numpy as np
>>> df1 = pd.DataFrame({
...     'time': np.array([1, 1], dtype='timedelta64[s]'),
...     'sym':['a', 'b'],
...     'p': pykx.LongVector([0, 1]),
...     'n': ['r', 's']
... })
>>> df2 = pd.DataFrame({
...     'time': np.array([1, 1], dtype='timedelta64[s]'),
...     'sym': ['a', 'b'],
...     'p': pykx.q('1 0N')
... })
>>> pykx.q.ajf0(pykx.SymbolVector(['sym', 'time']), df1, df2)
pykx.Table(q('
time                 sym p n
----------------------------
0D00:00:01.000000000 a   1 r
0D00:00:01.000000000 b   1 s
'))
```

### [asof](https://code.kx.com/q/ref/asof/)

Performs an as-of join across temporal columns in tables. The last column the second table must be temporal and correspond to a column in the first table argument. The return is the data from the first table is the last time that is less than or equal to the time in the second table per key. The time column will be removed from the output.

```python
>>> import pandas as pd
>>> import numpy as np
>>> df1 = pd.DataFrame({
...     'time': np.array([1, 2, 3, 4], dtype='timedelta64[s]'),
...     'sym': ['a', 'a', 'b', 'b'], 'p': pykx.LongVector([2, 4, 6, 8])})
>>> df2 = pd.DataFrame({'sym':['b'], 'time': np.array([3], dtype='timedelta64[s]')})
>>> pykx.q.asof(df1, df2)
pykx.Table(q('
p
-
6
'))
```

### [ej](https://code.kx.com/q/ref/ej/)

Equi join. The result has one combined record for each row in the second table that matches the first table on the columns specified in the first function parameter.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'sym':['a', 'a', 'b', 'a', 'c', 'b', 'c', 'a'], 'p': pykx.LongVector([2, 4, 6, 8, 1, 3, 5, 7])})
>>> df2 = pd.DataFrame({'sym':['a', 'b'], 'w': ['alpha', 'beta']})
>>> pykx.q.ej('sym', df1, df2)
pykx.Table(q('
sym p w
-----------
a   2 alpha
a   4 alpha
b   6 beta
a   8 alpha
b   3 beta
a   7 alpha
'))
```

### [ij](https://code.kx.com/q/ref/ij/)

Inner join. The result has one combined record for each row in the first table that matches the second table on the columns specified in the first function parameter.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'sym':['IBM', 'FDP', 'FDP', 'FDP', 'IBM', 'MSFT'], 'p': pykx.LongVector([7, 8, 6, 5, 2, 5])})
>>> df2 = pd.DataFrame({'sym':['IBM', 'MSFT'], 'ex': ['N', 'CME'], 'MC': pykx.LongVector([1000, 250])})
>>> df2 = pykx.q.xkey('sym', df2)
>>> pykx.Table(df1)
pykx.Table(q('
sym  p
------
IBM  7
FDP  8
FDP  6
FDP  5
IBM  2
MSFT 5
'))
>>> df2
pykx.KeyedTable(q('
sym | ex  MC
----| --------
IBM | N   1000
MSFT| CME 250
'))
>>> pykx.q.ij(df1, df2)
pykx.Table(q('
sym  p ex  MC
---------------
IBM  7 N   1000
IBM  2 N   1000
MSFT 5 CME 250
'))
```

### [ijf](https://code.kx.com/q/ref/ij/)

Inner join nulls filled. The result has one combined record for each row in the first table that matches the second table on the columns specified in the first function parameter.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'sym':['IBM', 'FDP', 'FDP', 'FDP', 'IBM', 'MSFT'], 'p': pykx.LongVector([7, 8, 6, 5, 2, 5])})
>>> df2 = pd.DataFrame({'sym':['IBM', 'MSFT'], 'ex': ['N', 'CME'], 'MC': pykx.LongVector([1000, 250])})
>>> b = pykx.q.xkey('sym', df2)
>>> pykx.Table(df1)
pykx.Table(q('
sym  p
------
IBM  7
FDP  8
FDP  6
FDP  5
IBM  2
MSFT 5
'))
>>> df2
pykx.KeyedTable(q('
sym | ex  MC
----| --------
IBM | N   1000
MSFT| CME 250
'))
>>> pykx.q.ijf(df1, df2)
pykx.Table(q('
sym  p ex  MC
---------------
IBM  7 N   1000
IBM  2 N   1000
MSFT 5 CME 250
'))
```

### [lj](https://code.kx.com/q/ref/lj/)

Left join. For each record in the first table, the result has one record with the columns of second table joined to columns of the first using the primary keys of the second table, if no value is present in the second table the record will contain null values in the place of the columns of the second table.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'sym':['IBM', 'FDP', 'FDP', 'FDP', 'IBM', 'MSFT'], 'p': pykx.LongVector([7, 8, 6, 5, 2, 5])})
>>> df2 = pd.DataFrame({'sym':['IBM', 'MSFT'], 'ex': ['N', 'CME'], 'MC': pykx.LongVector([1000, 250])})
>>> b = pykx.q.xkey('sym', df2)
>>> pykx.Table(df2)
pykx.Table(q('
sym  p
------
IBM  7
FDP  8
FDP  6
FDP  5
IBM  2
MSFT 5
'))
>>> df1
pykx.KeyedTable(q('
sym | ex  MC
----| --------
IBM | N   1000
MSFT| CME 250
'))
>>> pykx.q.lj(df1, df2)
pykx.Table(q('
sym  p ex  MC
---------------
IBM  7 N   1000
FDP  8
FDP  6
FDP  5
IBM  2 N   1000
MSFT 5 CME 250
'))
```

### [ljf](https://code.kx.com/q/ref/lj/)

Left join nulls filled. For each record in the first table, the result has one record with the columns of second table joined to columns of the first using the primary keys of the second table, if no value is present in the second table the record will contain null values in the place of the columns of the second table.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'sym':['IBM', 'FDP', 'FDP', 'FDP', 'IBM', 'MSFT'], 'p': pykx.LongVector([7, 8, 6, 5, 2, 5])})
>>> df2 = pd.DataFrame({'sym':['IBM', 'MSFT'], 'ex': ['N', 'CME'], 'MC': pykx.LongVector([1000, 250])})
>>> b = pykx.q.xkey('sym', df2)
>>> pykx.Table(df1)
pykx.Table(q('
sym  p
------
IBM  7
FDP  8
FDP  6
FDP  5
IBM  2
MSFT 5
'))
>>> df1
pykx.KeyedTable(q('
sym | ex  MC
----| --------
IBM | N   1000
MSFT| CME 250
'))
>>> pykx.q.ljf(df1, df2)
pykx.Table(q('
sym  p ex  MC
---------------
IBM  7 N   1000
FDP  8
FDP  6
FDP  5
IBM  2 N   1000
MSFT 5 CME 250
'))
```

### [pj](https://code.kx.com/q/ref/pj/)

Plus join. For each record in the first table, the result has one record with the columns of second table joined to columns of the first using the primary keys of the second table, if a value is present it is added to the columns of the first table, if no value is present the columns are left unchanged and new columns are set to 0.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'a': pykx.LongVector([1, 2, 3]), 'b':['x', 'y', 'z'], 'c': pykx.LongVector([10, 20, 30])})
>>> pykx.Table(df1)
pykx.Table(q('
a b c
------
1 x 10
2 y 20
3 z 30
'))
>>> df2 = pd.DataFrame({
...     'a': pykx.LongVector([1, 3]),
...     'b':['x', 'z'],
...     'c': pykx.LongVector([1, 2]),
...     'd': pykx.LongVector([10, 20])
... })
>>> df2 = pykx.q.xkey(pykx.SymbolVector(['a', 'b']), df2)
pykx.KeyedTable(q('
a b| c d
---| ----
1 x| 1 10
3 z| 2 20
'))
>>> pykx.q.pj(df1, df2)
pykx.Table(q('
a b c  d
---------
1 x 11 10
2 y 20 0
3 z 32 20
'))
```

### [uj](https://code.kx.com/q/ref/uj/)

Union join. Where the first table and the second table are both keyed or both unkeyed tables, returns the union of the columns, filled with nulls where necessary. If the tables have matching key columns then the records in the second table will be used to update the first table, if the tables are not keyed then the records from the second table will be joined onto the end of the first table.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'sym':['IBM', 'FDP', 'FDP', 'FDP', 'IBM', 'MSFT'], 'p': pykx.LongVector([7, 8, 6, 5, 2, 5])})
>>> df2 = pd.DataFrame({'sym':['IBM', 'MSFT'], 'ex': ['N', 'CME'], 'MC': pykx.LongVector([1000, 250])})
>>> df1
    sym  p
0   IBM  7
1   FDP  8
2   FDP  6
3   FDP  5
4   IBM  2
5  MSFT  5
>>> df2
    sym   ex    MC
0   IBM    N  1000
1  MSFT  CME   250
>>> pykx.q.uj(df1, df2)
pykx.Table(q('
sym  p ex  MC
---------------
IBM  7
FDP  8
FDP  6
FDP  5
IBM  2
MSFT 5
IBM    N   1000
MSFT   CME 250
'))
```

### [ujf](https://code.kx.com/q/ref/uj/)

Union join nulls filled. Where the first table and the second table are both keyed or both unkeyed tables, returns the union of the columns, filled with nulls where necessary. If the tables have matching key columns then the records in the second table will be used to update the first table, if the tables are not keyed then the records from the second table will be joined onto the end of the first table.

```python
>>> import pandas as pd
>>> df1 = pd.DataFrame({'sym':['IBM', 'FDP', 'FDP', 'FDP', 'IBM', 'MSFT'], 'p': pykx.LongVector([7, 8, 6, 5, 2, 5])})
>>> df2 = pd.DataFrame({'sym':['IBM', 'MSFT'], 'ex': ['N', 'CME'], 'MC': pykx.LongVector([1000, 250])})
>>> df1
    sym  p
0   IBM  7
1   FDP  8
2   FDP  6
3   FDP  5
4   IBM  2
5  MSFT  5
>>> df2
    sym   ex    MC
0   IBM    N  1000
1  MSFT  CME   250
>>> pykx.q.ujf(df1, df2)
pykx.Table(q('
sym  p ex  MC
---------------
IBM  7
FDP  8
FDP  6
FDP  5
IBM  2
MSFT 5
IBM    N   1000
MSFT   CME 250
'))
```

### [wj](https://code.kx.com/q/ref/wj/)

Window join. Returns for each record in the table, a record with additional columns `c0` and `c1`, which contain the results of the aggregation functions applied to values over the matching intervals defined in the first parameter of the function.

```python
>>> import pandas as pd
>>> import numpy as np
>>> pykx.q('t: ([]sym:3#`ibm;time:10:01:01 10:01:04 10:01:08;price:100 101 105)')
pykx.Table(pykx.q('
sym time     price
------------------
ibm 10:01:01 100
ibm 10:01:04 101
ibm 10:01:08 105
'))
>>> df_t = pd.DataFrame({
        'sym': ['ibm', 'ibm', 'ibm'],
        'time': np.array([36061, 36064, 36068], dtype='timedelta64[s]'),
        'price': pykx.LongVector([100, 101, 105])
    })
   sym            time  price
0  ibm 0 days 10:01:01    100
1  ibm 0 days 10:01:04    101
2  ibm 0 days 10:01:08    105
>>> pykx.q('q:([]sym:`ibm; time:10:01:01+til 9; ask: (101 103 103 104 104 107 108 107 108); bid: (98 99 102 103 103 104 106 106 107))')
pykx.Table(pykx.q('
sym time     ask bid
--------------------
ibm 10:01:01 101 98
ibm 10:01:02 103 99
ibm 10:01:03 103 102
ibm 10:01:04 104 103
ibm 10:01:05 104 103
ibm 10:01:06 107 104
ibm 10:01:07 108 106
ibm 10:01:08 107 106
ibm 10:01:09 108 107
'))
>>> f = pykx.SymbolVector(['sym', 'time'])
>>> w = pykx.q('-2 1+\:t.time')
>>> pykx.q.wj(w, f, df_t, pykx.q('(q;(max;`ask);(min;`bid))'))
pykx.Table(pykx.q('
sym time     price ask bid
--------------------------
ibm 10:01:01 100   103 98
ibm 10:01:04 101   104 99
ibm 10:01:08 105   108 104
'))
```

### [wj1](https://code.kx.com/q/ref/wj/)

Window join. Returns for each record in the table, a record with additional columns `c0` and `c1`, which contain the results of the aggregation functions applied to values over the matching intervals defined in the first parameter of the function.

```python
>>> import pandas as pd
>>> import numpy as np
>>> pykx.q('t: ([]sym:3#`ibm;time:10:01:01 10:01:04 10:01:08;price:100 101 105)')
pykx.Table(pykx.q('
sym time     price
------------------
ibm 10:01:01 100
ibm 10:01:04 101
ibm 10:01:08 105
'))
>>> df_t = pd.DataFrame({
...     'sym': ['ibm', 'ibm', 'ibm'],
...     'time': np.array([36061, 36064, 36068], dtype='timedelta64[s]'),
...     'price': pykx.LongVector([100, 101, 105])
... })
   sym            time  price
0  ibm 0 days 10:01:01    100
1  ibm 0 days 10:01:04    101
2  ibm 0 days 10:01:08    105
>>> pykx.q('q:([]sym:`ibm; time:10:01:01+til 9; ask: (101 103 103 104 104 107 108 107 108); bid: (98 99 102 103 103 104 106 106 107))')
pykx.Table(pykx.q('
sym time     ask bid
--------------------
ibm 10:01:01 101 98
ibm 10:01:02 103 99
ibm 10:01:03 103 102
ibm 10:01:04 104 103
ibm 10:01:05 104 103
ibm 10:01:06 107 104
ibm 10:01:07 108 106
ibm 10:01:08 107 106
ibm 10:01:09 108 107
'))
>>> f = pykx.SymbolVector(['sym', 'time'])
>>> w = pykx.q('-2 1+\:t.time')
>>> pykx.q.wj(w, f, df_t, pykx.q('(q;(max;`ask);(min;`bid))'))
pykx.Table(pykx.q('
sym time     price ask bid
--------------------------
ibm 10:01:01 100   103 98
ibm 10:01:04 101   104 99
ibm 10:01:08 105   108 104
'))
```

## List

### [count](https://code.kx.com/q/ref/count/)

Count the items of a list or dictionary.

```python
>>> pykx.q.count([1, 2, 3])
pykx.LongAtom(q('3'))
```

### [cross](https://code.kx.com/q/ref/cross/)

Returns all possible combinations of x and y.

```python
>>> pykx.q.cross([1, 2, 3], [4, 5, 6])
pykx.List(q('
1 4
1 5
1 6
2 4
2 5
2 6
3 4
3 5
3 6
'))
```

### [cut](https://code.kx.com/q/ref/cut/)

Cut a list or table into sub-arrays.

```python
>>> pykx.q.cut(3, range(10))
pykx.List(q('
0 1 2
3 4 5
6 7 8
,9
'))
```

### [enlist](https://code.kx.com/q/ref/enlist/)

Returns a list with its arguments as items.

```python
>>> pykx.q.enlist(1, 2, 3, 4)
pykx.LongVector(q('1 2 3 4'))
```

### [fills](https://code.kx.com/q/ref/fills/)

Replace nulls with preceding non-nulls.

```python
>>> a = pykx.q('0N 1 2 0N 0N 2 3 4 5 0N 4')
>>> pykx.q.fills(a)
pykx.LongVector(q('0N 1 2 2 2 2 3 4 5 5 4'))
```

### [first](https://code.kx.com/q/ref/first/)

First item of a list
```python
>>> pykx.q.first([1, 2, 3, 4, 5])
pykx.LongAtom(q('1'))
```

### [flip](https://code.kx.com/q/ref/flip/)

Returns x transposed, where x may be a list of lists, a dictionary or a table.

```python
>>> pykx.q.flip([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
pykx.List(q('
1 6
2 7
3 8
4 9
5 10
'))
```

### [group](https://code.kx.com/q/ref/group/)

Returns a dictionary in which the keys are the distinct items of x, and the values the indexes where the distinct items occur.

The order of the keys is the order in which they appear in x.

```python
>>> pykx.q.group(b'mississippi')
pykx.Dictionary(q('
m| ,0
i| 1 4 7 10
s| 2 3 5 6
p| 8 9
'))
```

### [inter](https://code.kx.com/q/ref/inter/)

Intersection of two lists or dictionaries.

```python
>>> pykx.q.inter([1, 2, 3], [2, 3, 4])
pykx.LongVector(q('2 3'))
```

### [last](https://code.kx.com/q/ref/first/#last)

Last item of a list

```python
>>> pykx.q.last([1, 2, 3])
pykx.LongAtom(q('3'))
```

### [mcount](https://code.kx.com/q/ref/count/#mcount)

Returns the x-item moving counts of the non-null items of y. The first x items of the result are the counts so far, and thereafter the result is the moving count.

```python
>>> pykx.q.mcount(3, pykx.q('1 2 3 4 5 0N 6 7 8'))
pykx.IntVector(q('1 2 3 3 3 2 2 2 3i'))
```

### [next](https://code.kx.com/q/ref/next/)

Next items in a list.

```python
>>> pykx.q.next([1, 2, 3, 4])
pykx.LongVector(q('2 3 4 0N'))
```

### [prev](https://code.kx.com/q/ref/next/#prev)

Immediately preceding items in a list.

```python
>>> pykx.q.prev([1, 2, 3, 4])
pykx.LongVector(q('0N 1 2 3'))
```

### [raze](https://code.kx.com/q/ref/raze/)

Return the items of x joined, collapsing one level of nesting.

```python
>>> pykx.q.raze([[1, 2], [3, 4]])
pykx.LongVector(q('1 2 3 4'))
```

### [reverse](https://code.kx.com/q/ref/reverse/)

Reverse the order of items of a list or dictionary.

```python
>>> pykx.q.reverse([1, 2, 3, 4, 5])
pykx.List(q('
5
4
3
2
1
'))
```

### [rotate](https://code.kx.com/q/ref/rotate/)

Shift the items of a list to the left or right.

```python
>>> pykx.q.rotate(2, [1, 2, 3, 4, 5])
pykx.LongVector(q('3 4 5 1 2'))
```

### [sublist](https://code.kx.com/q/ref/sublist/)

Select a sublist of a list.

```python
>>> pykx.q.sublist(2, [1, 2, 3, 4, 5])
pykx.LongVector(q('1 2'))
```

### [sv](https://code.kx.com/q/ref/sv/)

"Scalar from vector"

- join strings, symbols, or filepath elements
- decode a vector to an atom

```python
>>> pykx.q.sv(10, [1, 2, 3, 4])
pykx.LongAtom(q('1234'))
```

### [til](https://code.kx.com/q/ref/til/)

First x natural numbers.

```python
>>> pykx.q.til(10)
pykx.LongVector(q('0 1 2 3 4 5 6 7 8 9'))
```

### [union](https://code.kx.com/q/ref/union/)

Union of two lists.

```python
>>> pykx.q.union([1, 2, 3, 3, 5], [2, 4, 6, 8])
pykx.LongVector(q('1 2 3 5 4 6 8'))
```

### [vs](https://code.kx.com/q/ref/vs/)

"Vector from scalar"

- partition a symbol, string, or bytestream
- encode a vector from an atom, or a matrix from a vector

```python
>>> pykx.q.vs(b',', b'one,two,three')
pykx.List(q('
"one"
"two"
"three"
'))
```

### [where](https://code.kx.com/q/ref/where/)

Copies of indexes of a list or keys of a dictionary.

```python
>>> pykx.q.where(pykx.BooleanVector([True, False, True, True, False]))
pykx.LongVector(q('0 2 3'))
>>> pykx.q.where(pykx.q('1 0 0 1 0 1 1'))
pykx.LongVector(q('0 3 5 6'))
```

### [xprev](https://code.kx.com/q/ref/next/#xprev)

Nearby items in a list.

```python
>>> pykx.q.xprev(2, [1, 2, 3, 4, 5, 6])
pykx.LongVector(q('0N 0N 1 2 3 4'))
```

There is no `xnext` function, but `xprev` with a negative number as its first argument can achieve this.

```python
>>> pykx.q.xprev(-2, [1, 2, 3, 4, 5, 6])
pykx.LongVector(q('3 4 5 6 0N 0N'))
```

## Logic

### [all](https://code.kx.com/q/ref/all-any/#all/)

Everything is true.

```python
>>> pykx.q.all([True, True, True, True])
pykx.BooleanAtom(q('1b'))
>>> pykx.q.all([True, True, False, True])
pykx.BooleanAtom(q('0b'))
```

### [any](https://code.kx.com/q/ref/all-any/#any)

Something is true.

```python
>>> pykx.q.any([False, False, True, False])
pykx.BooleanAtom(q('1b'))
>>> pykx.q.any([False, False])
pykx.BooleanAtom(q('0b'))
```

## Math

### [abs](https://code.kx.com/q/ref/abs/)

Where x is a numeric or temporal, returns the absolute value of x. Null is returned if x is null.

```python
>>> pykx.q.abs(-5)
pykx.LongAtom(q('5'))
```

### [acos](https://code.kx.com/q/ref/cos/)

The arccosine of x; that is, the value whose cosine is x. The result is in radians and lies between 0 and π.

```python
>>> pykx.q.acos(0.5)
pykx.FloatAtom(q('1.047198'))
```

### [asin](https://code.kx.com/q/ref/sin/)

The arcsine of x; that is, the value whose sine is x. The result is in radians and lies between -π / 2 and π / 2. (The range is approximate due to rounding errors). Null is returned if the argument is not between -1 and 1.

```python
>>> pykx.q.asin(0.5)
pykx.FloatAtom(q('0.5235988'))
```

### [atan](https://code.kx.com/q/ref/tan/)

The arctangent of x; that is, the value whose tangent is x. The result is in radians and lies between -π / 2 and π / 2.
```python
>>> pykx.q.atan(0.5)
pykx.FloatAtom(q('0.4636476'))
```

### [avg](https://code.kx.com/q/ref/avg/#avg)

Arithmetic mean.

```python
>>> pykx.q.avg([1, 2, 3, 4, 7])
pykx.FloatAtom(q('3.4'))
```

### [avgs](https://code.kx.com/q/ref/avg/#avgs)

Running mean.

```python
>>> pykx.q.avgs([1, 2, 3, 4, 7])
pykx.FloatVector(q('1 1.5 2 2.5 3.4'))
```

### [ceiling](https://code.kx.com/q/ref/ceiling/)

Round up.

```python
>>> pykx.q.ceiling([-2.7, -1.1, 0, 1.1, 2.7])
pykx.LongVector(q('-2 -1 0 2 3'))
```

### [cor](https://code.kx.com/q/ref/cor/)

Correlation.

```python
>>> pykx.q.cor(pykx.LongVector([29, 10, 54]), pykx.LongVector([1, 3, 9]))
pykx.FloatAtom(q('0.7727746'))
```

### [cos](https://code.kx.com/q/ref/cos/)

The cosine of x, taken to be in radians. The result is between -1 and 1, or null if the argument is null or infinity.

```python
>>> pykx.q.cos(0.2)
pykx.FloatAtom(q('0.9800666'))
```

### [cov](https://code.kx.com/q/ref/cov/)

Where x and y are conforming numeric lists returns their covariance as a floating-point number. Applies to all numeric data types and signals an error with temporal types, char and sym.

```python
>>> pykx.q.cov(pykx.LongVector([29, 10, 54]), pykx.LongVector([1, 3, 9]))
pykx.FloatAtom(q('47.33333'))
```

### [deltas](https://code.kx.com/q/ref/deltas/)

Where x is a numeric or temporal vector, returns differences between consecutive pairs of its items.

```python
>>> pykx.q.deltas(pykx.LongVector([1, 4, 9, 16]))
pykx.LongVector(q('1 3 5 7'))
```

### [dev](https://code.kx.com/q/ref/dev/)

Standard deviation.

```python
>>> pykx.q.dev(pykx.LongVector([10, 343, 232, 55]))
pykx.FloatAtom(q('134.3484'))
```

### [div](https://code.kx.com/q/ref/div/)

Integer division.

```python
>>> pykx.q.div(7, 3)
pykx.LongAtom(q('2'))
```

### [ema](https://code.kx.com/q/ref/ema/)

The cosine of x, taken to be in radians. The result is between -1 and 1, or null if the argument is null or infinity.

```python
>>> pykx.q.ema(0.5, [1, 2, 3, 4, 5])
pykx.FloatVector(q('1 1.5 2.25 3.125 4.0625'))
```

### [exp](https://code.kx.com/q/ref/exp/)

Raise *e* to a power.

```python
>>> pykx.q.exp(1)
pykx.FloatAtom(q('2.718282'))
```

### [floor](https://code.kx.com/q/ref/floor/)

Round down.

```python
>>> pykx.q.floor([-2.7, -1.1, 0, 1.1, 2.7])
pykx.LongVector(q('-3 -2 0 1 2'))
```

### [inv](https://code.kx.com/q/ref/inv/)

Matrix inverse.

```python
>>> a = pykx.q('3 3# 2 4 8 3 5 6 0 7 1f')
pykx.List(q('
2 4 8
3 5 6
0 7 1
'))
>>> pykx.q.inv(a)
pykx.List(q('
-0.4512195  0.6341463  -0.195122
-0.03658537 0.02439024 0.1463415
0.2560976   -0.1707317 -0.02439024
'))
```

### [log](https://code.kx.com/q/ref/log/)

Natural logarithm.

```python
>>> pykx.q.log([1, 2, 3])
pykx.FloatVector(q('0 0.6931472 1.098612'))
```

### [lsq](https://code.kx.com/q/ref/lsq/)

Least squares, matrix divide.

```python
>>> a = pykx.q('1f+3 4#til 12')
pykx.List(q('
1 2  3  4
5 6  7  8
9 10 11 12
'))
>>> b = pykx.q('4 4#2 7 -2 5 5 3 6 1 -2 5 2 7 5 0 3 4f')
pykx.List(q('
2  7 -2 5
5  3 6  1
-2 5 2  7
5  0 3  4
'))
>>> pykx.q.lsq(a, b)
pykx.List(q('
-0.1233333 0.16      0.4766667 0.28
0.07666667 0.6933333 0.6766667 0.5466667
0.2766667  1.226667  0.8766667 0.8133333
'))
```

### [mavg](https://code.kx.com/q/ref/avg/#mavg)

Moving averages.

```python
>>> pykx.q.mavg(3, [1, 2, 3, 5, 7, 10])
pykx.FloatVector(q('1 1.5 2 3.333333 5 7.333333'))
```

### [max](https://code.kx.com/q/ref/max/)

Maximum.

```python
>>> pykx.q.max([0, 7, 2, 4 , 1, 3])
pykx.LongAtom(q('7'))
```

### [maxs](https://code.kx.com/q/ref/max/#maxs)

Maximums.

```python
>>> pykx.q.maxs([1, 2, 5, 4, 7, 1, 2])
pykx.LongVector(q('1 2 5 5 7 7 7'))
```

### [mdev](https://code.kx.com/q/ref/dev/#mdev)

Moving deviations.

```python
>>> pykx.q.mdev(3, [1, 2, 5, 4, 7, 1, 2])
pykx.FloatVector(q('0 0.5 1.699673 1.247219 1.247219 2.44949 2.624669'))
```

### [med](https://code.kx.com/q/ref/med/)

Median.

```python
>>> pykx.q.med([1, 2, 3, 4, 4, 1, 2, 4, 5])
pykx.FloatAtom(q('3f'))
```

### [min](https://code.kx.com/q/ref/min/)

Minimum.

```python
>>> pykx.q.min([7, 5, 2, 4, 6, 5, 1, 4])
pykx.LongAtom(q('1'))
```

### [mins](https://code.kx.com/q/ref/min/#mins)

Minimums.

```python
>>> pykx.q.mins([7, 5, 2, 4, 6, 5, 1, 4])
pykx.LongVector(q('7 5 2 2 2 2 1 1'))
```

### [mmax](https://code.kx.com/q/ref/max/#mmax)

Moving maximums.

```python
>>> pykx.q.mmax(4, [7, 5, 2, 4, 6, 5, 1, 4])
pykx.LongVector(q('7 7 7 7 6 6 6 6'))
```

### [mmin](https://code.kx.com/q/ref/min/#mmin)

Moving minimums.

```python
>>> pykx.q.mmin(4, pykx.LongVector([7, 5, 2, 4, 6, 5, 1, 4]))
pykx.LongVector(q('7 5 2 2 2 2 1 1'))
```

### [mmu](https://code.kx.com/q/ref/mmu/)

Matrix multiply, dot product.

```python
>>> a = pykx.q('2 4#2 4 8 3 5 6 0 7f')
>>> a
pykx.List(q('
2 4 8 3
5 6 0 7
'))
>>> b = pykx.q('4 3#"f"$til 12')
>>> b
pykx.List(q('
0 1  2
3 4  5
6 7  8
9 10 11
'))
>>> pykx.q.mmu(a, b)
pykx.List(q('
87 104 121
81 99  117
'))
```

### [mod](https://code.kx.com/q/ref/mod/)

Modulus.

```python
>>> pykx.q.mod([1, 2, 3, 4, 5, 6, 7], 4)
pykx.LongVector(q('1 2 3 0 1 2 3'))
```

### [msum](https://code.kx.com/q/ref/sum/#msum)

Moving sums.

```python
>>> pykx.q.msum(3, [1, 2, 3, 4, 5, 6, 7])
pykx.LongVector(q('1 3 6 9 12 15 18'))
```

### [neg](https://code.kx.com/q/ref/neg/)

Negate.

```python
>>> pykx.q.neg([2, 0, -1, 3, -5])
pykx.LongVector(q('-2 0 1 -3 5'))
```

### [prd](https://code.kx.com/q/ref/prd/)

Product.

```python
>>> pykx.q.prd([1, 2, 3, 4, 5])
pykx.LongAtom(q('120'))
```

### [prds](https://code.kx.com/q/ref/prd/#prds)

Cumulative products.

```python
>>> pykx.q.prds([1, 2, 3, 4, 5])
pykx.LongVector(q('1 2 6 24 120'))
```

### [rand](https://code.kx.com/q/ref/rand/)

Pick randomly.

```python
>>> pykx.q.rand([1, 2, 3, 4, 5])
pykx.LongAtom(q('2'))
```

### [ratios](https://code.kx.com/q/ref/ratios/)

Ratios between items.

```python
>>> pykx.q.ratios([1, 2, 3, 4, 5])
pykx.FloatVector(q('0n 2 1.5 1.333333 1.25'))
```

### [reciprocal](https://code.kx.com/q/ref/reciprocal/)

Reciprocal of a number.

```python
>>> pykx.q.reciprocal([1, 0, 3])
pykx.FloatVector(q('1 0w 0.3333333'))
```

### [scov](https://code.kx.com/q/ref/cov/#scov)

Sample covariance.

```python
>>> pykx.q.scov(pykx.LongVector([2, 3, 5, 7]), pykx.LongVector([4, 3, 0, 2]))
pykx.FloatAtom(q('-2.416667'))
```

### [sdev](https://code.kx.com/q/ref/dev/#sdev)

Sample standard deviation.

```python
>>> pykx.q.sdev(pykx.LongVector([10, 343, 232, 55]))
pykx.FloatAtom(q('155.1322'))
```

### [signum](https://code.kx.com/q/ref/signum/)

Where x (or its underlying value for temporals) is

- null or negative, returns `-1i`
- zero, returns `0i`
- positive, returns `1i`

```python
>>> pykx.q.signum([-2, 0, 1, 3])
pykx.IntVector(q('-1 0 1 1i'))
```

### [sin](https://code.kx.com/q/ref/sin/)

Sine.

```python
>>> pykx.q.sin(0.5)
pykx.FloatAtom(q('0.4794255'))
```

### [sqrt](https://code.kx.com/q/ref/sqrt/)

Square root.

```python
>>> pykx.q.sqrt([-1, 0, 25, 50])
pykx.FloatVector(q('0n 0 5 7.071068'))
```

### [sum](https://code.kx.com/q/ref/sum/)

Total.

```python
>>> pykx.q.sum(pykx.LongVector([2, 3, 5, 7]))
pykx.LongAtom(q('17'))
```

### [sums](https://code.kx.com/q/ref/sum/#sums)

Cumulative total.

```python
>>> pykx.q.sums(pykx.LongVector([2, 3, 5, 7]))
pykx.LongVector(q('2 5 10 17'))
```

### [svar](https://code.kx.com/q/ref/var/#svar)

Sample variance.

```python
>>> pykx.q.svar(pykx.LongVector([2, 3, 5, 7]))
pykx.FloatAtom(q('4.916667'))
```

### [tan](https://code.kx.com/q/ref/tan/)

Tangent.

```python
>>> pykx.q.tan(0.5)
pykx.FloatAtom(q('0.5463025'))
```

### [var](https://code.kx.com/q/ref/var/)

Variance.

```python
>>> pykx.q.var(pykx.LongVector([2, 3, 5, 7]))
pykx.FloatAtom(q('3.6875'))
```

### [wavg](https://code.kx.com/q/ref/avg/#wavg)

Weighted average.

```python
>>> pykx.q.wavg([2, 3, 4], [1, 2 ,4])
pykx.FloatAtom(q('2.666667'))
```

### [within](https://code.kx.com/q/ref/within/)

Check bounds.

```python
>>> pykx.q.within([1, 3, 10, 6, 4], [2, 6])
pykx.BooleanVector(q('01011b'))
```

### [wsum](https://code.kx.com/q/ref/sum/#wsum)

Weighted sum.

```python
>>> pykx.q.wsum([2, 3, 4], [1, 2, 4]) # equivalent to 2 * 1 + 3 * 2 + 4 * 4
pykx.LongAtom(q('24'))
```

### [xexp](https://code.kx.com/q/ref/exp/#xepx)

Raise x to a power.

```python
>>> pykx.q.xexp(2, 8)
pykx.FloatAtom(q('256f'))
```

### [xlog](https://code.kx.com/q/ref/log/#xlog)

Logarithm base x.

```python
>>> pykx.q.xlog(2, 8)
pykx.FloatAtom(q('3f'))
```

## Meta

### [attr](https://code.kx.com/q/ref/attr/)

[Attributes](../../user-guide/advanced/attributes.md) of an object, returns a Symbol Atom or Vector.

The possible attributes are:

| code | attribute             |
|------|-----------------------|
| s	   | sorted                |
| u	   | unique (hash table)   |
| p	   | partitioned (grouped) |
| g	   | true index (dynamic attribute): enables constant time update and access for real-time tables |

```python
>>> pykx.q.attr([1,2,3])
pykx.SymbolAtom(q('`'))
>>> pykx.q.attr(pykx.q('asc 1 2 3'))
pykx.SymbolAtom(q('`s'))
```

### [null](https://code.kx.com/q/ref/null/)

Is null.

```python
>>> pykx.q.null(1)
pykx.BooleanAtom(q('0b'))
>>> pykx.q.null(float('NaN'))
pykx.BooleanAtom(q('1b'))
>>> pykx.q.null(None)
pykx.BooleanAtom(q('1b'))
```

### [tables](https://code.kx.com/q/ref/tables/)

List of tables in a namespace.

```python
>>> pykx.q('exampleTable: ([] a: til 10; b: 10?10)')
pykx.Identity(pykx.q('::'))
>>> pykx.q('exampleTable: ([] a: til 10; b: 10?10)')
pykx.Table(q('
a b
---
0 8
1 1
2 9
3 5
4 4
5 6
6 6
7 1
8 8
9 5
'))
>>> pykx.q.tables('.')
pykx.SymbolVector(q(',`exampleTable'))
```

### [type](https://code.kx.com/q/ref/type/)

Underlying [k type](https://code.kx.com/q/ref/#datatypes) of an [object](../pykx-q-data/wrappers.md).

```python
>>> pykx.q.type(1)
pykx.ShortAtom(q('-7h'))
>>> pykx.q.type([1, 2, 3])
pykx.ShortAtom(q('0h'))
>>> pykx.q.type(pykx.LongVector([1, 2, 3]))
pykx.ShortAtom(q('7h'))
```

### [view](https://code.kx.com/q/ref/view/)

Expression defining a view.

```python
>>> pykx.q('v::2+a*3')
>>> pykx.q('a:5')
>>> pykx.q('v')
pykx.LongAtom(q('17'))
>>> pykx.q.view('v')
pykx.CharVector(q('"2+a*3"'))
```

### [views](https://code.kx.com/q/ref/view/#views)

List views defined in the default namespace.

```python
>>> pykx.q('v::2+a*3')
>>> pykx.q('a:5')
>>> pykx.q('v')
pykx.LongAtom(q('17'))
>>> pykx.q.views()
pykx.SymbolVector(q(',`v'))
```

## Queries

### [fby](https://code.kx.com/q/ref/fby/)

Apply an aggregate to groups.

```python
>>> d = pykx.q('data: 10?10')
pykx.LongVector(pykx.q('4 9 2 7 0 1 9 2 1 8'))
>>> group = pykx.SymbolVector(['a', 'b', 'a', 'b', 'c', 'd', 'c', 'd', 'd', 'c'])
pykx.SymbolVector(pykx.q('`a`b`a`b`c`d`c`d`d`c'))
>>> >>> pykx.q.fby(pykx.q('(sum; data)'), group)
pykx.LongVector(pykx.q('6 16 6 16 17 4 17 4 4 17'))
```

## Sort

### [asc](https://code.kx.com/q/ref/asc/)

Ascending sort.

```python
>>> pykx.q.asc([4, 2, 5, 1, 0])
pykx.LongVector(q('`s#0 1 2 4 5'))
```

### [bin](https://code.kx.com/q/ref/bin/)

Binary search.

```python
>>> pykx.q.bin([0, 2, 4, 6, 8, 10], 5)
pykx.LongAtom(q('2'))
>>> pykx.q.bin([0, 2, 4, 6, 8, 10], [-10, 0, 4, 5, 6, 20])
pykx.LongVector(q('-1 0 2 2 3 5'))
```

### [binr](https://code.kx.com/q/ref/bin/#binr)

Binary search right.

```python
>>> pykx.q.binr([0, 2, 4, 6, 8, 10], 5)
pykx.LongAtom(q('3'))
>>> pykx.q.binr([0, 2, 4, 6, 8, 10], [-10, 0, 4, 5, 6, 20])
pykx.LongVector(q('0 0 2 3 3 6'))
```

### [desc](https://code.kx.com/q/ref/desc/)

Descending sort.

```python
>>> pykx.q.desc([4, 2, 5, 1, 0])
pykx.LongVector(q('5 4 2 1 0'))
```

### [differ](https://code.kx.com/q/ref/differ/)

Find where list items change value.

```python
>>> pykx.q.differ([1, 1, 2, 3, 4, 4])
pykx.BooleanVector(q('101110b'))
```

### [distinct](https://code.kx.com/q/ref/distinct/)

Unique items of a list.

```python
>>> pykx.q.distinct([1, 3, 1, 4, 5, 1, 2, 3])
pykx.LongVector(q('1 3 4 5 2'))
```

### [iasc](https://code.kx.com/q/ref/asc/#iasc)

Ascending grade.

```python
>>> pykx.q.iasc([4, 2, 5, 1, 0])
pykx.LongVector(q('4 3 1 0 2'))
```

### [idesc](https://code.kx.com/q/ref/desc/#idesc)

Descending grade.

```python
>>> pykx.q.idesc([4, 2, 5, 1, 0])
pykx.LongVector(q('2 0 1 3 4'))
```

### [rank](https://code.kx.com/q/ref/rank/)

Position in the sorted list.

Where x is a list or dictionary, returns for each item in x the index of where it would occur in the sorted list or dictionary.

```python
>>> pykx.q.rank([4, 2, 5, 1, 0])
pykx.LongVector(q('3 2 4 1 0'))
>>> pykx.q.rank({'c': 3, 'a': 4, 'b': 1})
pykx.LongVector(q('2 0 1'))
```

### [xbar](https://code.kx.com/q/ref/xbar/)

Round y down to the nearest multiple of x.

```python
>>> pykx.q.xbar(5, 3)
pykx.LongAtom(q('0'))
>>> pykx.q.xbar(5, 5)
pykx.LongAtom(q('5'))
>>> pykx.q.xbar(5, 7)
pykx.LongAtom(q('5'))
>>> pykx.q.xbar(3, range(16))
pykx.LongVector(q('0 0 0 3 3 3 6 6 6 9 9 9 12 12 12 15'))
```

### [xrank](https://code.kx.com/q/ref/xrank/)

Group by value.

```python
>>> pykx.q.xrank(3, range(6))
pykx.LongVector(q('0 0 1 1 2 2'))
>>> pykx.q.xrank(4, range(9))
pykx.LongVector(q('0 0 0 1 1 2 2 3 3'))
```

## Table

### [cols](https://code.kx.com/q/ref/cols/#cols)

Column names of a table.

```python
>>> import pandas as pd
>>> import numpy as np
>>> df = pd.DataFrame({
...     'time': numpy.array([1, 2, 3, 4], dtype='timedelta64[s]'),
...     'sym':['a', 'a', 'b', 'b'],
...     'p': pykx.LongVector([2, 4, 6, 8])
...  })
>>> pykx.q.cols(df)
pykx.SymbolVector(q('`time`sym`p'))
```

### [csv](https://code.kx.com/q/ref/csv/)

CSV delimiter.

A synonym for "," for use in preparing text for CSV files, or reading them.

```python
>>> pykx.q.csv
pykx.CharAtom(q('","'))
```

### [fkeys](https://code.kx.com/q/ref/fkeys/)

Foreign-key columns of a table.

```python
>>> pykx.q('f:([x:1 2 3]y:10 20 30)')
pykx.Identity(q('::'))
>>> pykx.q('t: ([]a:`f$2 2 2; b: 0; c: `f$1 1 1)')
pykx.Identity(q('::'))
>>> pykx.q.fkeys('t')
pykx.Dictionary(q('
a| f
c| f
'))
```

### [insert](https://code.kx.com/q/ref/insert/)

Insert or append records to a table.

```python
>>> pykx.q('t: ([] a: `a`b`c; b: til 3)')
>>> pykx.q('t')
pykx.Table(q('
a b
---
a 0
b 1
c 2
'))
>>> pykx.q.insert('t', ['d', 3])
>>> pykx.q('t')
pykx.Table(q('
a b
---
a 0
b 1
c 2
d 3
'))
```

### [key](https://code.kx.com/q/ref/key/)

Where x is a dictionary (or the name of one), returns its keys.

```python
>>> pykx.q.key({'a': 1, 'b': 2})
pykx.SymbolVector(q('`a`b'))
```

### [keys](https://code.kx.com/q/ref/keys/)

Get the names of the key columns of a table.

```python
>>> pykx.q['v'] = pykx.KeyedTable(data={'x': [4, 5, 6]}, index=[1, 2, 3])
>>> pykx.q('v')
pykx.KeyedTable(pykx.q('
idx| x
---| -
1  | 4
2  | 5
3  | 6
'))
>>> pykx.q.keys('v')
pykx.SymbolVector(q(',`idx'))
```

### [meta](https://code.kx.com/q/ref/meta/)

Metadata for a table.

| Column | Information |
|--------|-------------|
| c      | column name |
| t      | [data type](https://code.kx.com/q/ref/#datatypes) |
| f      | foreign key (enums) |
| a      | [attribute](#attr) |

```python
>>> import pandas as pd
>>> import numpy as np
>>> df = pd.DataFrame({
...     'time': np.array([1, 2, 3, 4], dtype='timedelta64[s]'),
...     'sym': ['a', 'a', 'b', 'b'],
...     'p': pykx.LongVector([2, 4, 6, 8])
... })
>>> pykx.q.meta(df)
pykx.KeyedTable(q('
c   | t f a
----| -----
time| n
sym | s
p   | j
'))
```

### [ungroup](https://code.kx.com/q/ref/ungroup/)

Where x is a table, in which some cells are lists, but for any row, all lists are of the same length, returns the normalized table, with one row for each item of a lists.

```python
>>> a = pykx.Table([['a', [2, 3], 10], ['b', [5, 6, 7], 20], ['c', [11], 30]], columns=['s', 'x', 'q'])
>>> a
pykx.Table(pykx.q('
s x       q
------------
a (2;3)   10
b (5;6;7) 20
c ,11     30
'))
>>> pykx.q.ungroup(a)
pykx.Table(q('
s x  q
-------
a 2  10
a 3  10
b 5  20
b 6  20
b 7  20
c 11 30
'))
```

### [upsert](https://code.kx.com/q/ref/upsert/)

Add new records to a table.

```python
>>> import pandas as pd
>>> df = pd.DataFrame({'sym':['a', 'a', 'b', 'b'], 'p': pykx.LongVector([2, 4, 6, 8])})
>>> pykx.Table(df)
pykx.Table(q('
sym p
-----
a   2
a   4
b   6
b   8
'))
>>> pykx.q.upsert(df, ['c', 10])
>>> pykx.Table(q('
sym p
------
a   2
a   4
b   6
b   8
c   10
'))
```

### [xasc](https://code.kx.com/q/ref/asc/#xasc)

Sort a table in ascending order of specified columns.

```python
>>> import pandas as pd
>>> df = pd.DataFrame({'sym':['a', 'a', 'b', 'b', 'c', 'c'], 'p': pykx.LongVector([10, 4, 6, 2, 0, 8])})
>>> pykx.Table(df)
pykx.Table(q('
sym p
------
a   10
a   4
b   6
b   2
c   0
c   8
'))
>>> pykx.q.xasc('p', df)
pykx.Table(q('
sym p
------
c   0
b   2
a   4
b   6
c   8
a   10
'))
```

### [xcol](https://code.kx.com/q/ref/cols/#xcol)

Rename table columns.

```python
>>> import pandas as pd
>>> df = pd.DataFrame({'sym':['a', 'a', 'b', 'b', 'c', 'c'], 'p': pykx.LongVector([10, 4, 6, 2, 0, 8])})
>>> pykx.Table(df)
pykx.Table(q('
sym p
------
a   10
a   4
b   6
b   2
c   0
c   8
'))
>>> pykx.q.xcol(pykx.SymbolVector(['Sym', 'Qty']), df)
pykx.Table(q('
Sym Qty
-------
a   10
a   4
b   6
b   2
c   0
c   8
'))
>>> pykx.q.xcol({'p': 'Qty'}, df)
pykx.Table(q('
sym Qty
-------
a   10
a   4
b   6
b   2
c   0
c   8
'))
```

### [xcols](https://code.kx.com/q/ref/cols/#xcols)

Reorder table columns.

```python
>>> import pandas as pd
>>> import numpy as np
>>> df = pd.DataFrame({
...     'time': np.array([1, 2, 3, 4], dtype='timedelta64[s]'),
...     'sym':['a', 'a', 'b', 'b'],
...     'p': pykx.LongVector([2, 4, 6, 8])
... })
>>> pykx.Table(df)
pykx.Table(q('
time                 sym p
--------------------------
0D00:00:01.000000000 a   2
0D00:00:02.000000000 a   4
0D00:00:03.000000000 b   6
0D00:00:04.000000000 b   8
'))
>>> pykx.q.xcols(pykx.SymbolVector(['p', 'sym', 'time']), df)
pykx.Table(q('
p sym time
--------------------------
2 a   0D00:00:01.000000000
4 a   0D00:00:02.000000000
6 b   0D00:00:03.000000000
8 b   0D00:00:04.000000000
'))
```

### [xdesc](https://code.kx.com/q/ref/desc/#xdesc)

Sorts a table in descending order of specified columns. The sort is by the first column specified, then by the second column within the first, and so on.

```python
>>> import pandas as pd
>>> df = pd.DataFrame({'sym':['a', 'a', 'b', 'b', 'c', 'c'], 'p': pykx.LongVector([10, 4, 6, 2, 0, 8])})
>>> pykx.Table(df)
pykx.Table(q('
sym p
------
a   10
a   4
b   6
b   2
c   0
c   8
'))
>>> pykx.q.xdesc('p', df)
pykx.Table(q('
sym p
------
a   10
c   8
b   6
a   4
b   2
c   0
'))
```

### [xgroup](https://code.kx.com/q/ref/xgroup/)

Groups a table by values in selected columns.

```python
>>> import pandas as pd
>>> df = pd.DataFrame({'sym':['a', 'a', 'b', 'b', 'c', 'c'], 'p': pykx.LongVector([10, 4, 6, 2, 0, 8])})
>>> pykx.Table(df)
pykx.Table(q('
sym p
------
a   10
a   4
b   6
b   2
c   0
c   8
'))
>>> pykx.q.xgroup('sym', df)
pykx.KeyedTable(q('
sym| p
---| ----
a  | 10 4
b  | 6  2
c  | 0  8
'))
```

### [xkey](https://code.kx.com/q/ref/keys/#xkey)

Set specified columns as primary keys of a table.

```python
>>> import pandas as pd
>>> df = pd.DataFrame({'sym':['a', 'a', 'b', 'b', 'c', 'c'], 'p': pykx.LongVector([10, 4, 6, 2, 0, 8])})
>>> pykx.Table(df)
pykx.Table(q('
sym p
------
a   10
a   4
b   6
b   2
c   0
c   8
'))
>>> pykx.q.xkey('p', df)
pykx.KeyedTable(q('
p | sym
--| ---
10| a
4 | a
6 | b
2 | b
0 | c
8 | c
'))
```

## Text

### [like](https://code.kx.com/q/ref/like/)

Whether text matches a pattern.

```python
>>> pykx.q.like('quick', b'qu?ck')
pykx.BooleanAtom(q('1b'))
>>> pykx.q.like('brown', b'br[ao]wn')
pykx.BooleanAtom(q('1b'))
>>> pykx.q.like('quick', b'quickish')
pykx.BooleanAtom(q('0b'))
```

### [lower](https://code.kx.com/q/ref/lower/)

Shift case to lower case.

```python
>>> pykx.q.lower('HELLO')
pykx.SymbolAtom(q('`hello'))
>>> pykx.q.lower(b'HELLO')
pykx.CharVector(q('"hello"'))
```

### [ltrim](https://code.kx.com/q/ref/trim/#ltrim)

Remove leading nulls from a list.

```python
>>> pykx.q.ltrim(b'    pykx    ')
pykx.CharVector(q('"pykx    "'))
```

### [md5](https://code.kx.com/q/ref/md5/)

Message digest hash.

```python
>>> pykx.q.md5(b'pykx')
pykx.ByteVector(q('0xfba0532951f022133f8e8b14b6ddfced'))
```

### [rtrim](https://code.kx.com/q/ref/trim/#rtrim)

Remove trailing nulls from a list.

```python
>>> pykx.q.rtrim(b'    pykx    ')
pykx.CharVector(q('"    pykx"'))
```

### [ss](https://code.kx.com/q/ref/ss/)

String search.

```python
>>> pykx.q.ss(b'a cat and a dog', b'a')
pykx.LongVector(q('0 3 6 10'))
```

### [ssr](https://code.kx.com/q/ref/ss/#ssr)

String search and replace.

```python
>>> pykx.q.ssr(b'toronto ontario', b'ont', b'x')
pykx.CharVector(q('"torxo xario"'))
```

### [string](https://code.kx.com/q/ref/string/)

Cast to string.

```python
>>> pykx.q.string(2)
pykx.CharVector(q(',"2"'))
>>> pykx.q.string([1, 2, 3, 4, 5])
pykx.List(q('
,"1"
,"2"
,"3"
,"4"
,"5"
'))
```

### [trim](https://code.kx.com/q/ref/trim/)

Remove leading and trailing nulls from a list.

```python
>>> pykx.q.trim(b'    pykx    ')
pykx.CharVector(q('"pykx"'))
```

### [upper](https://code.kx.com/q/ref/lower/#upper)

Shift case to upper case.

```python
>>> pykx.q.upper('hello')
pykx.SymbolAtom(q('`HELLO'))
>>> pykx.q.upper(b'hello')
pykx.CharVector(q('"HELLO"'))
```

## Operators

### [drop](https://code.kx.com/q/ref/drop/)

Drop items from a list, entries from a dictionary or rows from a table.

Examples:

Drop the first 3 items from a list

```python
>>> import pykx as kx
>>> kx.q.drop(3, kx.q('1 2 3 4 5 6'))
pykx.LongVector(pykx.q('4 5 6'))
```

Drop the last 10 rows from a table
        
```python
>>> import pykx as kx
>>> tab = kx.Table(data={
...     'x': kx.q.til(100),
...     'y': kx.random.random(100, 10.0)
... })
>>> kx.q.drop(-10, tab)
pykx.Table(pykx.q('
x  y        
------------
0  3.927524 
1  5.170911 
2  5.159796 
3  4.066642 
4  1.780839
..
'))
>>> len(kx.q.drop(-10, tab))
90
```

### [coalesce](https://code.kx.com/q/ref/coalesce/)

Merge two keyed tables ignoring null objects

Example:

Coalesce two keyed tables one containing nulls

```python
>> tab1 = kx.Table(data={
...     'x': kx.q.til(10),
...     'y': kx.random.random(10, 10.0)
...     }).set_index('x')
>>> tab2 = kx.Table(data={
...     'x': kx.q.til(10),
...     'y':kx.random.random(10, [1.0, kx.FloatAtom.null, 10.0])
...     }).set_index('x')
>>> kx.q.coalesce(tab1, tab2)
pykx.KeyedTable(pykx.q('
x| y         z
-| ------------
0| 9.006991  10
1| 8.505909
2| 8.196014  10
3| 0.9982673 1
4| 8.187707
..
'))
```

### [fill](https://code.kx.com/q/ref/fill)

Replace nulls in lists, dictionaries or tables

Examples:

Replace null values in a list

```python
>>> null_list = kx.random.random(10, [10, kx.LongAtom.null, 100])
>>> kx.q.fill(0, null_list)
```

Replace all null values in a table

```python
>>> table = kx.Table(data={
...     'x': kx.random.random(10, [10.0, kx.FloatAtom.null, 100.0]),
...     'y': kx.random.random(10, [10.0, kx.FloatAtom.null, 100.0])
...     })
>>> kx.q.fill(10.0, table)
```

### [take](https://code.kx.com/q/ref/take)

Select leading or trailing items from a list or dictionary, named entries from a dictionary, or named columns from a table

Examples:

Retrieve the last 3 items from a list

```python
>>> lst = kx.q.til(100)
>>> kx.q.take(-3, lst)
pykx.LongVector(pykx.q('97 98 99'))
```

Retrieve named columns from a table using take

```python
>>> table = kx.Table(data={
...     'x': kx.random.random(5, 10.0),
...     'y': kx.random.random(5, 10.0),
...     'z': kx.random.random(5, 10.0),
...     })
>>> kx.q.take(['x', 'y'], table)
pykx.Table(pykx.q('
x        y       
-----------------
6.916099 9.672398
2.296615 2.306385
6.919531 9.49975 
4.707883 4.39081 
6.346716 5.759051
'))
```

### [set_attribute](https://code.kx.com/q/ref/set-attribute/)

Set an attribute for a supplied list or dictionary, the supplied attribute must be one of: 's', 'u', 'p' or 'g'.

```python
>>> kx.q.set_attribute('s', kx.q.til(10))
pykx.LongVector(pykx.q('`s#0 1 2 3 4 5 6 7 8 9'))
>>> kx.q.set_attribute('g', [2, 1, 2, 1])
pykx.LongVector(pykx.q('`g#2 1 2 1'))
```

### [join](https://code.kx.com/q/ref/join/)

Join atoms, lists, dictionaries or tables

```python
>>> kx.q.join([1, 2, 3], [4, 5, 6])
pykx.LongVector(pykx.q('1 2 3 4 5 6'))
```

Join multiple dictionaries together

```python
>>> kx.q.join({'x': 1, 'y': 2}, {'z': 3})
pykx.Dictionary(pykx.q('
x| 1
y| 2
z| 3
'))
```

Join multiple columns row wise

```python
>>> t = kx.q('([]a:1 2 3;b:`a`b`c)')
>>> s = kx.q('([]a:10 11;b:`d`e)')
>>> kx.q.join(t, s)
pykx.Table(pykx.q('
a  b
----
1  a
2  b
3  c
10 d
11 e
'))
```

### [find](https://code.kx.com/q/ref/find/)

Find the first occurrence of an item(s) in a list

```python
>>> lst = [10, -8, 3, 5, -1, 2, 3]
>>> kx.q.find(lst, -8)
pykx.LongAtom(pykx.q('1'))
>>> kx.q.find(lst, [10, 3])
pykx.LongVector(pykx.q('0 2'))
```

### [enum_extend](https://code.kx.com/q/ref/enum-extend/)

Extend a defined variable enumeration

```python
>>> kx.q['foo'] = ['a', 'b']
>>> kx.q.enum_extend('foo', ['a', 'b', 'c', 'a', 'b'])
pykx.EnumVector(pykx.q('`foo$`a`b`c`a`b'))
>>> kx.q['foo']
pykx.SymbolVector(pykx.q('`a`b`c'))
```

Extend a filepath enumeration

```python
>>> import os
>>> from pathlib import Path
>>> kx.q['bar'] = ['c', 'd']    # about to be overwritten
>>> kx.q.enum_extend(Path('bar'), ['a', 'b', 'c', 'b', 'b', 'a'])
pykx.EnumVector(pykx.q('`bar$`a`b`c`b`b`a'))
>>> os.system('ls -l bar')
-rw-r--r--  1 username  staff  14 20 Aug 09:34 bar
>>> kx.q['bar']
pykx.SymbolVector(pykx.q('`a`b`c'))
```

### [roll](https://code.kx.com/q/ref/roll/)

Generate a random list of values with duplicates, for this the first parameter must be positive.

```python
>>> kx.q.roll(3, 10.0)
pykx.FloatVector(pykx.q('3.927524 5.170911 5.159796'))
>>> kx.q.roll(4, [1, 'a', 10.0])
pykx.List(pykx.q('
`a
1
`a
10f
'))
```

### [deal](https://code.kx.com/q/ref/deal/)

Generate a random list of values without duplicates, for this the first parameter must be negative.

```python
>>> kx.q.deal(-5, 5)
pykx.LongVector(pykx.q('1 3 2 0 4'))
>>> kx.q.deal(-3, ['the', 'quick', 'brown', 'fox'])
pykx.SymbolVector(pykx.q('`the`brown`quick'))
```

### [dict](https://code.kx.com/q/ref/dict/)

Generate a dictionary by passing two lists of equal lengths

```python
>>> kx.q.dict(['a', 'b', 'c'], [1, 2, 3])
pykx.Dictionary(pykx.q('
a| 1
b| 2
c| 3
'))
```

### [enkey](https://code.kx.com/q/ref/enkey/)

Create a keyed table by passing an integer to a simple table. This is similar to `set_index`

```python
>>> simple_tab = kx.Table(data = {
...     'x': [1, 2, 3],
...     'y': [4, 5, 6]
...     })
>>> kx.q.dict(1, simple_tab)
pykx.KeyedTable(pykx.q('
x| y
-| -
1| 4
2| 5
3| 6
'))
```

### [unkey](https://code.kx.com/q/ref/unkey/)

Remove the keys from a keyed table returning a simple table, this is similar to `reset_index`
with no arguments

```python
>>> keyed_tab = kx.Table(data = {
...     'x': [1, 2, 3],
...     'y': [4, 5, 6]
...     }).set_index(1)
>>> kx.q.unkey(0, keyed_tab)
pykx.Table(pykx.q('
x y
---
1 4
2 5
3 6
'))
```

### [enumeration](https://code.kx.com/q/ref/enumeration/)

Enumerate a symbol list
- First argument is a variable in q memory denoting a symbol list
- Second argument is a vector of integers in the domain 0-length(first argument)

```python
>>> kx.q['x'] = ['a', 'b', 'c', 'd']
>>> kx.q.enumeration('x', [1, 2, 3])
pykx.EnumVector(pykx.q('`x$`b`c`d'))
```

### [enumerate](https://code.kx.com/q/ref/enumerate/)

Enumerate a list of symbols based on the symbols in a global q variable

```python
>>> kx.q['d'] = ['a', 'b', 'c']
>>> y = ['a', 'b', 'c', 'b', 'a', 'b', 'c']
>>> kx.q.enumerate('d', y)
pykx.EnumVector(pykx.q('`d$`a`b`c`b`a`b`c'))
```

### [pad](https://code.kx.com/q/ref/pad/)

Pad a supplied PyKX string (Python bytes) to the length supplied by the user.
In the case that you are padding the front of a string use a negative value.

```python
>>> kx.q.pad(-5, b'abc')
pykx.CharVector(pykx.q('"  abc"'))
>>> kx.q.pad(10, [b'test', b'string', b'length'])
pykx.List(pykx.q('
"test      "
"string    "
"length    "
'))
```

### [cast](https://code.kx.com/q/ref/cast/)

Convert to another datatype, this should be a single lower case character byte, or name of the type.
See https://code.kx.com/q/ref/cast/ for the accepted list.

```python
>>> long_vec = kx.q('til 10')
>>> kx.q.cast('short', long_vec)
pykx.ShortVector(pykx.q('0 1 2 3 4 5 6 7 8 9h'))
>>> kx.q.cast(b'b', long_vec)
pykx.BooleanVector(pykx.q('0111111111b'))
```

### [tok](https://code.kx.com/q/ref/tok/)

Interpret a PyKX string as a data value(s), this should use a single upper case character byte or
a non-positive PyKX short value.
See https://code.kx.com/q/ref/tok/ for more information on accepted lists for casting

```python
>>> kx.q.tok(b'F', b'3.14')
pykx.FloatAtom(pykx.q('3.14'))
>>> float_int = kx.toq(-9, kx.ShortAtom)
>>> kx.qkx.toq(int(1), kx.ShortAtom)
```

### [compose](https://code.kx.com/q/ref/compose/)

Compose a unary value function with another.

```python
>>> f = kx.q('{2*x}')
>>> ff = kx.q('{[w;x;y;z]w+x+y+z}')
>>> d = kx.q.compose(f, ff)
>>> d(1, 2, 3, 4)
pykx.LongAtom(pykx.q('20'))
```

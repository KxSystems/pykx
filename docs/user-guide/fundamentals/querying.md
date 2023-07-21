# Querying PyKX data

There are a number of ways to query tables using PyKX.

1.  Directly using q.
2.  The qSQL API.
3.  The ANSI SQL API.

Each of these methods can be used both locally using [`Embedded q`][pykx.EmbeddedQ] and over IPC
using a [`QConnection`][pykx.ipc.QConnection] instance.

## Directly using q

For users that have previous knowledge of `kdb+/q` and wish to directly write their queries in pure q
they can directly write and execute queries in the same way they would in a q process.

For example you can do this to run qSQL queries directly, where q is either `Embedded q` or an instance
of a `pykx.QConnection`.

```Python
>>> q('select from t') # where t is a table in q's memory
>>> q('{[t] select from t}', tab) # where tab is a PyKX Table object
```

## Query APIs

PyKX has two main APIs to help query local tables as well as tables over IPC. The first API is the
[`qSQL API`][pykx.query.QSQL] which can be used to generate functional q SQL queries on tables. The
second API is the [`ANSI SQL API`][pykx.query.SQL] which supports a large
[`subset of ANSI SQL`](https://code.kx.com/insights/core/sql/sql-compliance.html).

### qSQL API

The [`qSQL API`][pykx.query.QSQL] provides various helper functions around generating selecting, updating, deleteing and
executing various [`functional qSQL`](https://code.kx.com/q4m3/9_Queries_q-sql/#912-functional-forms)
queries.

For example you can do this to execute a functional `qSQL` select, where q is `Embedded q` or a `pykx.QConnection`
instance.

```python
# select from table object
>>> pykx.q.qsql.select(qtab, columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
# or by name
>>> pykx.q.qsql.select('qtab', columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
```

Or you can use this to run a functional `qSQL` execute.

```python
>>> pykx.q.qsql.exec(qtab, columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'})
```

You can also update rows within tables using `qSQL` for example.

```python
>>> pykx.q.qsql.update(qtab, {'eye': ['blue']}, where='hair=`fair')
```

You can also delete rows of a table based on vairious conditions using `qSQL`.

```python
>>> pykx.q.qsql.delete('qtab', where=['hair=`fair', 'age=28'])
```

### ANSI SQL API

The [`ANSI SQL API`][pykx.query.SQL] can be used to prepare and execute `SQL` queries,
on `q` tables. The use of this API also requires an extra feature flag to be present on your / the
servers license to be used.

For example you can do this to execute a `SQL` query against a table named `trades` in `q`'s memory
using either `Embedded q` or over IPC using a `pykx.QConnection`.

```python
>>> q.sql('select * from trades where date = $1 and price < $2', date(2022, 1, 2), 500.0)
```

You can also directly pass a [`pykx.Table`][] object in as a variable to `SQL` queries.

```python
>>> q.sql('select * from $1', trades) # where `trades` is a `pykx.Table` object
```

Finally, you can prepare a `SQL` query and then when it is used later the types will be forced to
match in order for the query to run.

```Python
>>> p = q.sql.prepare('select * from trades where date = $1 and price < $2',
    kx.DateAtom,
    kx.FloatAtom
)
>>> q.sql.execute(p, date(2022, 1, 2), 500.0)
```

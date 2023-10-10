# Attributes

Attributes are metadata that you attach to lists of special forms. They are also used on table
columns to speed retrieval for some operations. PyKX can make certain optimizations
based on the structure of the list implied by the attribute.

Attributes (other than `` `g#``) are descriptive rather than prescriptive. By this we mean that
by applying an attribute you are asserting that the list has a special form, which PyKX will check.
It does not instruct PyKX to (re)make the list into the special form; that is your job. A list
operation that respects the form specified by the attribute leaves the attribute intact
(other than `` `p#``), while an operation that breaks the form results in the attribute being
removed in the result.

## Applying Attributes

Attributes can be applied on the various `Vector`/`List` types as well as `Tables` and `KeyedTable`'s.
These attributes can be applied to their supported types by directly calling the `sorted`, `unique`,
`grouped`, and `parted` methods on these objects.

Examples: Applying the sorted attribute to a `Vector` can be done by calling the `sorted` method on
the `Vector`.

```Python
>>> a = kx.q.til(10)
>>> a
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
>>> a.sorted()
pykx.LongVector(pykx.q('`s#0 1 2 3 4 5 6 7 8 9'))
```

Applying the unique attribute to the first column of the table.

```Python
>>> a = kx.q('([] a: til 10; b: `a`b`c`d`e`f`g`h`i`j)')
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s
'))
>>> a = a.unique()
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j   u
b| s
'))
```

Applying the grouped attribute to a specified column of a table.

```Python
>>> a = kx.q('([] a: til 10; b: `a`a`b`b`c`c`d`d`e`e)')
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s
'))
>>> a = a.grouped('b')
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s   g
'))
```

Applying the parted attribute to multiple columns on a table.

```Python
>>> a = kx.q('([] a: til 10; b: `a`a`b`b`c`c`d`d`e`e)')
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s
'))
>>> a = a.parted(['a', 'b'])
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j   p
b| s   p
'))
```

### Sorted

The sorted attribute ensures that all items in the `Vector` / `Table` column are sorted in ascending
order. This attribute will be removed if you append to the list with an item that is not in sorted
order.

### Unique

The unique attribute ensures that all items in the `Vector` / `Table` column are unique (there are
no duplicated values). This attribute will be removed if you append to the list with an item that
is not unique.

### Grouped

The grouped attribute ensures that all items in the `Vector` / `Table` column are stored in a
different format to help reduce memory usage, it creates a backing dictionary to store the value and
indexes that each value has within the list. Unlike other attributes the grouped attribute will be
kept on all insert operations to the list.

For example this is how a grouped list would be stored.

```q
// The list
`g#`a`b`c`a`b`b`c
// The backing dictionary
a| 0 3
b| 1 4 5
c| 2 6
```

### Parted

The parted attribute is similar to the grouped attribute with the additional requirement that each
unique value must be adjacent to its other copies, where the grouped attribute allows them to be
dispersed throughout the `Vector` / `Table`. When possible the parted attribute will result in a
larger performance gain than using the grouped attribute.
This attribute will be removed if you append to the list with an item that is not in the parted
order.

```q
// Can be parted
`p#`a`a`a`e`e`b`b`c`c`c`d
// Has to be grouped as the `d symbols are not all contiguous within the vector
`g#`a`a`d`e`e`b`b`c`c`c`d
```

## Performance

When attributes are set on PyKX objects various functions can use these attributes to speed up their
execution, by using different algorithms. For example searching through a list without an attribute
requires checking every single value, however setting the sorted attribute allows a search algorithm
to use a binary search in stead and then only a fraction of the values actually need to be checked.

Examples of some functions that can use attributes to speed up execution.

- Where clauses in `select` and `exec` templates run faster with `where =`, `where in` and `where within`.
- Searching with [`bin`](../../api/pykx-execution/q.md#bin), [`distinct`](../../api/pykx-execution/q.md#distinct),
    [`Find`](https://code.kx.com/q/ref/find/) and [`in`](https://code.kx.com/q/ref/in/).
- Sorting with [`iasc`](../../api/pykx-execution/q.md#iasc) or [`idesc`](../../api/pykx-execution/q.md#idesc).

!!!Note
    Setting attributes consumes resources and is likely to improve performance on large lists.

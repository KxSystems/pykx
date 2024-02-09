# PyKX under q known issues

PyKX aims to make q and Python operate as seamlessly as possible together.
However due to differences in the languages there are some special cases to be aware of when using the interface.

## Passing special values to PyKX objects

PyKX under q uses certain special values to control how objects are returned/converted. When you need to pass these special values as parameters some specific steps must be followed.

### Return control values `<`, `>`, and `*`

Using the [PyKX function API](intro.md#pykx-function-api), PyKX objects can be called directly (returning PyKX objects) or declared callable returning q or `foreign` data.

Users explicitly specify the return type as q or foreign, the default is as a PyKX object.

Given `func`, a object representing a callable Python function or method, we can carry out the following operations:

```q
func                   / func is callable by default (returning PyKX)
func arg               / call func(arg) (returning PyKX)
func[*]                / declare func callable (returning PyKX)
func[*]arg             / call func(arg) (returning PyKX)
func[*;arg]            / equivalent
func[<]                / declare func callable (returning q)
func[<]arg             / call func(arg) (returning q)
func[<;arg]            / equivalent
func[>]                / declare func callable (returning foreign)
func[>]arg             / call func(arg) (returning foreign)
func[>;arg]            / equivalent
```

**Chaining operations** Returning another PyKX object from a function or method call, allows users to chain together sequences of operations.
We can also chain these operations together with calls to `.pykx.import`, `.pykx.get` and `.pykx.eval`.

Due to this usage of `<`, `>`, and `*` as control characters passing them as arguments to functions must be managed more carefully.

```q
func // Avoid passing the function without specifying a return type if you need to pass *,<,> as possible arguments 
func arg // Avoid passing the argument without specifying a return type if you need to pass *,<,> as possible arguments 
```

Do attach a return type to the function as you define it:

```q
q)f:.pykx.eval["lambda x: x";<] // Specify < to return output as q object
q)f[*] // *,<,> can now be passed as a arguments successfully 
*
```

### Conversion control values `` ` `` and `` `. ``

When [converting data](intro.md#converting-data), given a PyKX object `obj` representing Python data, we can get the underlying data (as foreign or q) using:

```q
obj`. / get data as foreign
obj`  / get data as q
```

For example:

```q
q)x:.pykx.eval"(1,2,3)"
q)x
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q)x`.
foreign
q)x`
1 2 3
```

Due to this usage of `` ` `` and `` `. `` as control characters passing them as arguments to functions must be managed more carefully:

.i.e

```q
q).pykx.eval["lambda x: x"][`]`
'Provided foreign object is not a Python object
```

To avoid this you can define the return type using `<` or `>` in advance:

```q
q).pykx.eval["lambda x: x";<][`]
`
```

Or wrap the input in `.pykx.tok`:

```q
q).pykx.eval["lambda x: x"][.pykx.tok[`]]`
`
```

### Default parameter `::`

In q, functions take between 1-8 parameters. This differs from Python.

When one calls a q function with empty brackets `[]` a default value is still passed.
This value is `::` the generic null.

```q
q)(::)~{x}[] //Showing x parameter receives the generic null ::
1b
```

Due to this difference with Python, using `::` as an argument to PyKX functions has some difficulties:

```q
q)f:.pykx.eval["lambda x: x";<]
q)f[::] // The Python cannot tell the difference between f[] and f[::] as they resolve to the same input
'TypeError("<lambda>() missing 1 required positional argument: 'x'")
  [0]  f[::]
```

You can avoid this by wrapping the input in `.pykx.tok`:

```q
q)(::)~f[.pykx.tok[::]]
1b
```

Note Python functions with 0 parameters run without issue as they ignore the passed `(::)`:

```q
p)def noparam():return 7
q)f:.pykx.get[`noparam;<]
q)f[]
7
q)f[::] / equivalent
7
```
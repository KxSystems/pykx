# PyKX Conversion Considerations

PyKX attempts to make conversions between q and Python as seamless as possible.
However due to differences in their underlying implementations there are cases where 1 to 1 mappings are not possible.

## Data types and conversions

The key PyKX APIs around data types and conversions are outlined under:

* [Convert Pythonic data to PyKX](../../api/pykx-q-data/toq.md)
* [PyKX type wrappers](../../api/pykx-q-data/wrappers.md)
* [PyKX to Pythonic data type mapping](../../api/pykx-q-data/type_conversions.md)
* [Registering Custom Conversions](../../api/pykx-q-data/register.md)

## Text representation in PyKX

[Text representation in PyKX](../fundamentals/text.md) requires consideration as there are some key differences between the `Symbol` and `Char` data types.

## Nulls and Infinites

Most q datatypes have the concepts of null, negative infinity, and infinity. Python does not have the concept of infinites and its null behaviour differs in implementation. The page [handling nulls and infinities](./nulls_and_infinities.md) details the needed considerations when dealing with these special values.

## Temporal types

### Timestamp/Datetime types

Particular care is needed when converting temporal types as Python and q use different [epoch](https://en.wikipedia.org/wiki/Epoch_(computing)) values:

* q 2000
* Python 1970

__Note:__ The following details focus on `NumPy` but similar considerations should be taken in to account when converting Python, Pandas, and PyArrow objects.

The 30 year epoch offset means there are times which are unreachable in one or the other language:

|               | TimestampVector                 | datetime64[ns]                  |
|---------------|---------------------------------|---------------------------------|
| Minimum value | `1707.09.22D00:12:43.145224194` | `1677-09-21T00:12:43.145224194` |
| Maximum value | `2292.04.10D23:47:16.854775806` | `2262-04-11T23:47:16.854775807` |

As such the range of times which can be directly converted should be considered:

* Minimum value: `1707-09-22T00:12:43.145224194`
* Maximum value: `2262-04-11T23:47:16.854775807`

As mentioned [above](#nulls-and-infinites) most q data types have null, negative infinity, and infinity values.

|                   | q representation | datetime64[ns]                  |
|-------------------|------------------|---------------------------------|
| Null              | `0Np`            | `NaT`                           |
| Negative Infinity | `-0Wp`           | `1707-09-22T00:12:43.145224193` |
| Infinity          | `0Wp`            | Overflow cannot be represented  |

Converting from q to NumPy using `.np()`, `0Np` and `-0Wp` convert to meaningful values but `0Wp` overflows:

```q
>>> kx.q('0N -0W 0Wp').np()
array(['NaT', '1707-09-22T00:12:43.145224193', '1707-09-22T00:12:43.145224191'], dtype='datetime64[ns]')
```

Converting to q using `toq` by default only the NumPy maximum values converts to a meaningful value:

```q
>>> arr = np.array(['NaT', '1677-09-21T00:12:43.145224194', '2262-04-11T23:47:16.854775807'], dtype='datetime64[ns]')
>>> kx.toq(arr)
pykx.TimestampVector(pykx.q('2262.04.11D23:47:16.854775808 2262.04.11D23:47:16.854775810 2262.04.11D23:47:16.854775807'))
```

To additionally handle `NaT` being converted the `handle_nulls` keyword can be used:

```q
>>> arr = np.array(['NaT', '1677-09-21T00:12:43.145224194', '2262-04-11T23:47:16.854775807'], dtype='datetime64[ns]', handle_nulls=True)
>>> kx.toq(arr)
pykx.TimestampVector(pykx.q('0N 2262.04.11D23:47:16.854775810 2262.04.11D23:47:16.854775807'))
```

Using `raw=True` we can request that the epoch offset is not applied. This allows for the underlying numeric values to be accessed directly:

```python
>>> kx.q('0N -0W 0Wp').np(raw=True)
array([-9223372036854775808, -9223372036854775807,  9223372036854775807])
```

Passing back to q with `toq` these are then presented as the long null, negative infinity, and infinity:

```python
>>> kx.toq(kx.q('0N -0W 0Wp').np(raw=True))
pykx.LongVector(pykx.q('0N -0W 0W'))
```

`ktype` can be passed during `toq` to specify desired types:

```python
>>> kx.toq(pd.DataFrame(data= {'d':np.array(['2020-09-08T07:06:05'], dtype='datetime64[s]')}), ktype={'d':kx.DateVector})
pykx.Table(pykx.q('
d         
----------
2020.09.08
'))
```

Note that:

* Dictionary based conversion is only supported when operating in [licensed mode](../../user-guide/advanced/modes.md).
* Data is first converted to the default type and then cast to the desired type.

Other items of note:

* In NumPy further data types exist `datetime64[us]`, `datetime64[ms]`, `datetime64[s]` which due to their lower precision have a wider range of dates they can represent. When converted using to q using `toq` these all present as q `Timestamp` type and as such only dates within the range this data type can represent should be converted.
* Pandas 2.* changes behavior and conversions should be reviewed as  part of an upgrade of this package. [PyKX to Pythonic data type mapping](../../api/pykx-q-data/type_conversions.md) includes examples showing differences seen when calling `.pd()`.

### Duration types

Duration types do not have the issue of epoch offsets but some range limitations exist when converting between Python and PyKX.

`kx.SecondVector` and `kx.MinuteVector` convert to `timedelta64[s]`:

|                                     | q representation | timedelta64[s]            |
|-------------------------------------|------------------|---------------------------|
| `kx.SecondVector` Null              | `0Nv`            | `NaT`                     |
| `kx.SecondVector` Negative Infinity | `-0Wv`           | `-24856 days +20:45:53`   |
| `kx.SecondVector` Infinity          | `0Wv`            | `24855 days 03:14:07`     |
| `kx.MinuteVector` Null              | `0Nu`            | `NaT`                     |
| `kx.MinuteVector` Negative Infinity | `-0Wu`           | `-1491309 days +21:53:00` |
| `kx.MinuteVector` Infinity          | `0Wu`            | `1491308 days 02:07:00`   |

When converting Python to q using `toq` care must be taken as `timedelta64[s]` is 64 bit and converts to `kx.SecondVector` which is 32 bit:

|               | SecondVector | timedelta64[s]                    |
|---------------|--------------|-----------------------------------|
| Minimum value | `**:14:06`   | `106751991167300 days 15:30:07`   |
| Maximum value | `-**:14:06`  | `-106751991167301 days +08:29:53` |

As such the range of times which can be directly converted should be considered:

* Minimum value: `-24856 days +20:45:54`
* Maximum value: `24855 days 03:14:06`

q does not display values of second type over `99:59:59`, beyond this `**` is displayed in the hour field.
The data is still stored correctly and will display when converted:

```python
>>> kx.q('99:59:59 +1')
pykx.SecondAtom(pykx.q('**:00:00'))
>>> kx.q('99:59:59 +1').pd()
Timedelta('4 days 04:00:00')
```

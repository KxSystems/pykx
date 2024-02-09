# Pandas Like API for PyKX Tables

The aim of this page is to demonstrate PyKX functionality that aligns with the approach and principles of the Pandas API for DataFrame interactions. Not all operations supported by Pandas are covered, only the operations on PyKX tables that follow Pandas API conventions. In particular, it focuses on areas where PyKX/q has the potential to provide a performance advantage over the use of Pandas. This advantage may be in the memory footprint of the operations and/or in the execution time of the analytic.

A full breakdown of the the available functionality and examples of its use can be found [here](Pandas_API.ipynb).

## Covered sections of the Pandas API

Coverage in this instance refers here to functionality covered by the PyKX API for Tables which has equivalent functionality to the methods and attributes supported by the Pandas DataFrame API. This does not cover the functionality supported by Pandas for interactions with Series objects or for reading/writing CSV/JSON files etc.


If there's any functionality you would like to see added to this library, please open an issue [here](https://github.com/KxSystems/pykx/issues) or open a pull request [here](https://github.com/KxSystems/pykx/pulls).

### Property/metadata type information

| DataFrame Properties | PyKX Supported? | PyKX API Documentation Link | Additional Information |
|----------------------|-----------------|-----------------------------|------------------------|
| [columns](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html) | :material-check: | [link](Pandas_API.ipynb#tablecolumns) | |
| [dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html) | :material-check: | [link](https://code.kx.com/pykx/2.2/user-guide/advanced/Pandas_API.html#tabledtypes) | |
| [empty](https://https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.empty.html) | :material-check: | [link](https://code.kx.com/pykx/2.2/user-guide/advanced/Pandas_API.html#tableempty) | |
| [ndim](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ndim.html) | :material-check: | [link](https://code.kx.com/pykx/2.2/user-guide/advanced/Pandas_API.html#tablendim) | |
| [shape](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html) | :material-check: | [link](https://code.kx.com/pykx/2.2/user-guide/advanced/Pandas_API.html#tableshape) | |
| [size](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.size.html) | :material-check: | [link](https://code.kx.com/pykx/2.2/user-guide/advanced/Pandas_API.html#tablesize) | |

### Analytic functionality

| DataFrame Method     | PyKX Supported? | PyKX API Documentation Link | Additional Information |
|----------------------|-----------------|-----------------------------|------------------------|
| [abs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.abs.html) | :material-check: | [link](Pandas_API.ipynb#tableabs) | |
| [agg](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) | :material-check: | [link](Pandas_API.ipynb#tableagg) | |
| [apply](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html) | :material-check: | [link](Pandas_API.ipynb#tableapply) | |
| [groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) | :material-check: | [link](Pandas_API.ipynb#tablegroupby) | |
| [max](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.max.html) | :material-check: | [link](Pandas_API.ipynb#tablemax) | |
| [mean](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html) | :material-check: | [link](Pandas_API.ipynb#tablemean) | |
| [median](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.median.html) | :material-check: | [link](Pandas_API.ipynb#tablemedian) | |
| [min](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.min.html) | :material-check: | [link](Pandas_API.ipynb#tablemin) | |
| [mode](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mode.html) | :material-check: | [link](Pandas_API.ipynb#tablemode) | |
| [sum](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html) | :material-check: | [link](Pandas_API.ipynb#tablesum) | |
| [skew](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.skew.html) | :material-check: | [link](Pandas_API.ipynb#tableskew) | |
| [std](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.std.html) | :material-check: | [link](Pandas_API.ipynb#tablestd) | |
| [prod](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.prod.html) | :material-check: | [link](Pandas_API.ipynb#tableprod) | |

### Querying and data interrogation

| DataFrame Method     | PyKX Supported? | PyKX API Documentation Link | Additional Information |
|----------------------|-----------------|-----------------------------|------------------------|
| [all](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.all.html) | :material-check: | [link](Pandas_API.ipynb#tableall) | |
| [any](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.any.html) | :material-check: | [link](Pandas_API.ipynb#tableany) | |
| [at](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html) | :material-check: | [link](Pandas_API.ipynb#tableat) | |
| [count](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.count.html) | :material-check: | [link](Pandas_API.ipynb#tablecount) | |
| [get](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.get.html) | :material-check: | [link](Pandas_API.ipynb#tableget) | |
| [head](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) | :material-check: | [link](Pandas_API.ipynb#tablehead) | |
| [iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html) | :material-check: | [link](Pandas_API.ipynb#tableiloc) | |
| [loc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html) | :material-check: | [link](Pandas_API.ipynb#tableloc) | |
| [sample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html) | :material-check: | [link](Pandas_API.ipynb#tablesample) | |
| [select_dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html) | :material-check: | [link](Pandas_API.ipynb#tableselect_dtypes) | |
| [tail](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html) | :material-check: | [link](Pandas_API.ipynb#tabletail) | |

### Data Preprocessing

| DataFrame Method     | PyKX Supported? | PyKX API Documentation Link | Additional Information |
|----------------------|-----------------|-----------------------------|------------------------|
| [add_prefix](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.add_prefix.html) | :material-check: | [link](Pandas_API.ipynb#tableas_prefix) | |
| [add_suffix](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.add_suffix.html) | :material-check: | [link](Pandas_API.ipynb#tableas_suffix) | |
| [astype](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html) | :material-check: | [link](Pandas_API.ipynb#tableastype) | |
| [drop](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html) | :material-check: | [link](Pandas_API.ipynb#tabledrop) | |
| [drop_duplicates](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html) | :material-check: | [link](Pandas_API.ipynb#tabledrop_duplicates) | |
| [pop](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pop.html) | :material-check: | [link](Pandas_API.ipynb#tablepop) | |
| [rename](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html) | :material-check: | [link](Pandas_API.ipynb#tablerename) | |
| [set_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html) | :material-check: | [link](Pandas_API.ipynb#tableset_index) | |

### Data Joins/Merge

| DataFrame Method     | PyKX Supported? | PyKX API Documentation Link | Additional Information |
|----------------------|-----------------|-----------------------------|------------------------|
| [merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html) | :material-check: | [link](Pandas_API.ipynb#tablemerge) | |
| [merge_asof](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge_asof.html) | :material-check: | [link](Pandas_API.ipynb#tablemerge_asof) | |

### Unsupported Functionality

| DataFrame Methods    | PyKX Supported?  | PyKX API Documentation Link | Additional Information |
|----------------------|------------------|-----------------------------|------------------------|
| `*from*`             | :material-close: | | Functionality for the creation of PyKX Tables from alternative data sources is not supported at this time. |
| `*plot*`             | :material-close: | | Functionality for the plotting of columns/tables is not supported at this time. |
| `*sparse*`           | :material-close: | | Sparse data like interactions presently not supported. |
| `to_*`               | :material-close: | | Functionality for the conversion/persistence of PyKX Tables to other formats is not supported at this time. |

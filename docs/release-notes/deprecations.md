# Deprecations

A list of deprecated behaviors and the version in which they were removed.

| Feature                                           | Alternative               | Deprecated    | Removed |
|---------------------------------------------------|---------------------------|---------------|------------|
| `kx.q.system.console_size`                        | `kx.q.system.display_size`| 3.1.3         |            |
| `.pykx.console[]` on Windows                      |                           | 3.1.3         | 3.1.3      |
| `labels` keyword for `rename` method              | `mapper`                  | 2.5.0         | 3.1.0      |
| `type` from dtypes for `kx.Table`                 |                           | 2.5.0         | 3.0.1      |
| `modify` keyword for operations on `kx.Table`     | `inplace`                 | 2.3.1         | 3.0.0      |
| `replace_self` keyword for overwriting `kx.Table` | `inplace`                 | 2.3.1         | 3.0.0      |
| `PYKX_NO_SIGINT`                                  | `PYKX_NO_SIGNAL`          | 2.2.1         | 3.0.0      |
| `IGNORE_QHOME`                                    | `PYKX_IGNORE_QHOME`       | 3.0.0         | 3.0.0      |
| `KEEP_LOCAL_TIMES`                                | `PYKX_KEEP_LOCAL_TIMES`   | 3.0.0         | 3.0.0      |
| `SKIP_UNDERQ`                                     | `PYKX_SKIP_UNDERQ`        | 3.0.0         | 3.0.0      |
| `UNDER_PYTHON`                                    | `PYKX_UNDER_PYTHON`       | 2.2.1         | 3.0.0      |
| `UNSET_PYKX_GLOBALS`                              |                           | 3.0.0         | 3.0.0      |
| `PYKX_UNSET_GLOBALS`                              |                           | 3.0.0         | 3.0.0      |
| `PYKX_ENABLE_PANDAS_API`                          |                           | 3.0.0         | 3.0.0      |
| `.pd(raw_guids)`                                  |                           | 2.5.0         | 2.5.0      |


## PyKX 3.1.3

Release Date: 2025-06-12

- Deprecated `kx.q.system.console_size`, use `kx.q.system.display_size` instead.
- `.pykx.console[]` has been removed on Windows due to incompatibility. Will now error with `'.pykx.console is not available on Windows` if called.


## PyKX 3.1.0

Release Date: 2025-02-11

- Removal of previously deprecated use of keyword `labels` when using the `rename` method for table objects. Users should use the `mapper` keyword to maintain the same behavior.
- Error message when checking a license referenced a function `pykx.util.install_license` which is deprecated, this has now been updated to reference `pykx.license.install`


## PyKX 3.0.1

Release Date: 2024-12-04

- Removal of column `type` from the return of `#!python dtypes` method for `#!python kx.Table` objects, previously this had raised a deprecation warning

## PyKX 3.0.0

Release Date: 2024-11-12

- Removal of various deprecated keywords used in table operations:
	- `#!python modify` keyword for `#!python select`, `#!python exec`, `#!python update` and `#!python delete` operations on `#!python pykx.Table` and `#!python pykx.KeyedTable`. This has been permanently changed to use `#!python inplace`.
	- `#!python replace_self` keyword when attempting to overwrite a `#!python pykx.Table` or `#!python KeyedTable` using insert/upsert functionality. This has been permanently changed to use `#!python inplace`.

- The following table outlines environment variables/configuration options which are now fully deprecated and the updated name for these values if they exist.

	| **Deprecated option**    | **Supported option**    |
	| :----------------------- | :---------------------- |
	| `PYKX_NO_SIGINT`         | `PYKX_NO_SIGNAL`        |
	| `IGNORE_QHOME`           | `PYKX_IGNORE_QHOME`     |
	| `KEEP_LOCAL_TIMES`       | `PYKX_KEEP_LOCAL_TIMES` |
	| `SKIP_UNDERQ`            | `PYKX_SKIP_UNDERQ`      |
	| `UNDER_PYTHON`           | `PYKX_UNDER_PYTHON`     |
	| `UNSET_PYKX_GLOBALS`     | No longer applicable    |
	| `PYKX_UNSET_GLOBALS`     | No longer applicable    |
	| `PYKX_ENABLE_PANDAS_API` | No longer applicable    |


## PyKX 2.5.0

Release Date: 2024-05-15

- Deprecated `.pd(raw_guids)` keyword.
- Renamed `labels` parameter in `Table.rename()` to `mapper` to match Pandas. Added deprecation warning to `labels`.
- Deprecation of `type` column in `dtypes` output as it is a reserved keyword. Use new `datatypes` column instead.


## PyKX 2.3.1

Release Date: 2024-02-07

- To align with other areas of PyKX the `upsert` and `insert` methods for PyKX tables and keyed tables now support the keyword argument `inplace`, this change will deprecate usage of `replace_self` with the next major release of PyKX.
  

## PyKX 2.2.1

Release Date: 2023-11-30

- Deprecation of internally used environment variable `UNDER_PYTHON` which has been replaced by `PYKX_UNDER_PYTHON` to align with other internally used environment variables.
- Addition of deprecation warning for environmental configuration option `PYKX_NO_SIGINT` which is to be replaced by `PYKX_NO_SIGNAL`. This is used when users require no signal handling logic overwrites and now covers `SIGTERM`, `SIGINT`, `SIGABRT` signals amongst others.

## PyKX 1.6.1

Release Date: 2023-07-19

- Added deprecation warning around the discontinuing of support for Python 3.7.


## PyKX 1.0.1

Release Date: 2022-03-18

- The `sync` parameter for `pykx.QConnection` and `pykx.QConnection.__call__` has been renamed to the less confusing name `wait`. The `sync` parameter remains, but its usage will result in a `DeprecationWarning` being emitted. The `sync` parameter will be removed in a future version.


## PyKX 1.0.0

Release Date: 2022-02-14

- The `pykdb.q.ipc` attribute has been removed. The IPC module can be accessed directly instead at `pykx.ipc`, but generally one will only need to access the `QConnection` class, which can be accessed at the top-level: `pykx.QConnection`.
- The `pykdb.q.K` attribute has been removed. Instead, `K` types can be used as constructors for that type by leveraging the `toq` module. For example, instead of `pykdb.q.K(x)` one should write `pykx.K(x)`. Instead of `pykx.q.K(x, k_type=pykx.k.SymbolAtom)` one should write `pykx.SymbolAtom(x)` or `pykx.toq(x, ktype=pykx.SymbolAtom)`.
- Most `KdbError`/`QError` subclasses have been removed, as identifying them is error prone, and we are unable to provide helpful error messages for most of them.
- The `pykx.kdb` singleton class has been removed.
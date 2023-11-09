# PyKX Configurable Behavior

The following document outlines how users can modify the underlying behavior of PyKX based on their specific use-case. The [options](#options) presented are provided for use-case/performance tuned optimisations of the library itself.

Setting of these configuration options is supported via a [configuration file](#configuration-file) or [environment variables](#environment-variables) as described below. In all cases environment variable definitions will take precedence over definitions within the configuration file.

## Configuration File

Users can use a configuration file `.pykx-config` to define configuration options for PyKX initialization. The following provides an example of a `.pykx-config` file which operates according to `*.toml` syntax:

```bash
[default]
PYKX_IGNORE_QHOME="true"
PYKX_KEEP_LOCAL_TIMES="true"

[test]
PYKX_GC="true"
PYKX_RELEASE_GIL="true"
```

On import of PyKX the file `.pykx-config` will be searched for according to the following path ordering, the first location containing a `.pykx-config` file will be used for definition of the :

| Order | Location      |
|-------|---------------|
| 1.    | `Path('.')`   |
| 2.    | `Path(os.getenv('PYKX_CONFIGURATION_LOCATION'))` |
| 3.    | `Path.home()` |

When loading this file unless otherwise specified PyKX will use the profile `default`. Use of non default profiles from within this file can be configured through the setting of an environment variable `PYKX_PROFILE` prior to loading of PyKX, for example using the above configuration file.

=== "default"

	```python
	>>> import pykx as kx
	>>> kx.config.ignore_qhome
        True
	```

=== "test"

	```python
	>>> import os
	>>> os.environ['PYKX_PROFILE'] = "test"
	>>> import pykx as kx
	>>> kx.config.k_gc
	True
	```

## Environment variables

For users wishing to make use of the provided [options](#options) as environment variables this is also supported, for example a user can define the environment variables to use before import of PyKX as follows.

```python
>>> import os
>>> os.environ['PYKX_RELEASE_GIL'] = '1'
>>> os.environ['PYKX_GC'] = '1'
>>> import pykx as kx
>>> kx.config.k_gc
True
```

## Options

The options can be used to tune PyKX behavior at run time. These variables need to be set before attempting to import PyKX and will take effect for the duration of the execution of the PyKX process.

### General

The following variables can be used to enable or disable advanced features of PyKX across all modes of operation:

| Option                          | Default | Values                                                                | Description                                                                                                                                                                                                                                                                                                                                                                                       | Status                                           |
|---------------------------------|---------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| `PYKX_IGNORE_QHOME`             | `False` | `1` or `true`                                                         | When loading PyKX on a machine with an existing q installation (and the environment variable `QHOME` set to the installation folder), PyKX will look within this directory for q scripts their dependencies. It will then symlink these files to make them available to load under PyKX. This variable instructs PyKX to not perform this symlinking.                                             |                                                  |
| `PYKX_KEEP_LOCAL_TIMES`         | `False` | `1` or `true`                                                         | When converting a Python datetime object to q, PyKX will translate the Python datetime into UTC before the conversion. This variable instructs PyKX to convert the Python datetime using the local time zone.                                                                                                                                                                                     |                                                  |
| `PYKX_ALLOCATOR`                | `False` | `1` or `true`                                                         | When converting a Numpy array to q, PyKX implements a full data copy in order to translate the Numpy array to q representation in memory. When this is set PyKX implements [NEP-49](https://numpy.org/neps/nep-0049.html) which allows q to handle memory allocation of all Numpy arrays so they can be converted more efficiently to q. This avoids the need to resort to a copy where possible. |                                                  |
| `PYKX_GC`                       | `False` | `1` or `true`                                                         | When PYKX_ALLOCATOR is enabled, PyKX can trigger q garbage collector when Numpy arrays allocated by PyKX are deallocated. This variable enables this behavior which will release q memory to the OS following deallocation of the Numpy array at the cost of a small overhead.                                                                                                                    |                                                  |
| `PYKX_LOAD_PYARROW_UNSAFE`      | `False` | `1` or `true`                                                         | By default, PyKX uses a subprocess to import pyarrow as it can result in a crash when the version of pyarrow is incompatible. This variable will trigger a normal import of pyarrow and importing PyKX should be slightly faster.                                                                                                                                                                 |                                                  |
| `PYKX_MAX_ERROR_LENGTH`         | `256`   | size in characters                                                    | By default, PyKX reports IPC connection errors with a message buffer of size 256 characters. This allows the length of these error messages to be modified reducing the chance of excessive error messages polluting logs.                                                                                                                                                                        |                                                  |
| `PYKX_NOQCE`                    | `False` | `1` or `true`                                                         | On Linux, PyKX comes with q Cloud Edition features from Insights Core (https://code.kx.com/insights/1.2/core/). This variable allows a user to skip the loading of q Cloud Edition functionality, saving some time when importing PyKX but removing access to possibly supported additional functionality.                                                                                        |                                                  |
| `PYKX_Q_LIB_LOCATION`           | `UNSET` | Path to a directory containing q libraries necessary for loading PyKX | See [here](../release-notes/changelog.md#pykx-131) for detailed information. This allows a user to centralise the q libraries, `q.k`, `read.q`, `libq.so` etc to a managed location within their environment which is decentralised from the Python installation. This is required for some enterprise use-cases.                                                                                 |                                                  |
| `PYKX_RELEASE_GIL`              | `False` | `1` or `true`                                                         | When PYKX_RELEASE_GIL is enabled the Python Global Interpreter Lock will not be held when calling into q.                                                                                                                                                                                                                                                                                         |                                                  |
| `PYKX_Q_LOCK`                   | `False` | `1` or `true`                                                         | When PYKX_Q_LOCK is enabled a re-entrant lock is added around calls into q, this lock will stop multiple threads from calling into q at the same time. This allows embedded q to be thread safe even when using PYKX_RELEASE_GIL.                                                                                                                                                                 |                                                  |
| `PYKX_DEBUG_INSIGHTS_LIBRARIES` | `False` | `1` or `true`                                                         | If the insights libraries failed to load this variable can be used to print out the full error output for debugging purposes.                                                                                                                                                                                                                                                                     |                                                  |
| `PYKX_UNLICENSED`               | `False` | `1` or `true`                                                         | Set PyKX to make use of the library in `unlicensed` mode at all times.                                                                                                                                                                                                                                                                                                                            |                                                  |
| `PYKX_LICENSED`                 | `False` | `1` or `true`                                                         | Set PyKX to make use of the library in `licensed` mode at all times.                                                                                                                                                                                                                                                                                                                              |                                                  |
| `IGNORE_QHOME`                  | `True`  | `1` or `true`                                                         | When loading PyKX on a machine with an existing q installation (and the environment variable `QHOME` set to the installation folder), PyKX will look within this directory for q scripts their dependencies. It will then symlink these files to make them available to load under PyKX. This variable instructs PyKX to not perform this symlinking.                                             | `DEPRECATED`, please use `PYKX_IGNORE_QHOME`     |
| `KEEP_LOCAL_TIMES`              | `False` | `1` or `true`                                                         | When converting a Python datetime object to q, PyKX will translate the Python datetime into UTC before the conversion. This variable instructs PyKX to convert the Python datetime using the local time zone.                                                                                                                                                                                     | `DEPRECATED`, please use `PYKX_KEEP_LOCAL_TIMES` |


The variables below can be used to set the environment for q (embedded in PyKX, in licensed mode):

| Variable | Values   | Description |
|----------|----------|-------------|
| `QARGS`  | See link | Command-line flags to pass to q, see [here](https://code.kx.com/q/basics/cmdline/) for more information. |
| `QHOME`  | Path to the users q installation folder | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information. |
| `QLIC`   | Path to the folder where the q license should be found | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information. |
| `QINIT`  | Path to an additional `*.q` file loaded after `PyKX` has initialized | See [here](https://code.kx.com/q4m3/14_Introduction_to_Kdb%2B/#1481-the-environment-variables) for more information. |

#### PyKX QARGS Supported Additions

When using PyKX users can use the following values when defining `QARGS` to modify the behaviour of PyKX at initialisation when running within a Linux environment.

| Input          | Description                                                                     |
|----------------|---------------------------------------------------------------------------------|
| `--no-qce`     | Ensure that no kdb Insights libraries are loaded at initialisation of PyKX.     |
| `--no-kurl`    | Ensure that the kdb Insights `kurl` library is not loaded at initialisation.    |
| `--no-objstor` | Ensure that the kdb Insights `objstor` library is not loaded at initialisation. |
| `--no-qlog`    | Ensure that the kdb Insights `qlog` library is not loaded at initialisation.    |
| `--no-sql`     | Ensure that the kdb Insights `sql` library is not loaded at initialisation.     |

### PyKX under q

PyKX can be loaded and used from a q session (see [here](../pykx-under-q/intro.md) for more information). The following variables are specific to this mode of operation.

| Variable                  | Values                        | Description | Status |
|---------------------------|-------------------------------|-------------|--------|
| `PYKX_DEFAULT_CONVERSION` | `py`, `np`, `pd`, `pa` or `k` | Default conversion to apply when passing q objects to Python. Converting to Numpy (`np`) by default. | |
| `PYKX_SKIP_UNDERQ`        | `1` or `true`                 | When importing PyKX from Python, PyKX will also load `pykx.q` under its embedded q. This variable skips this step. | |
| `PYKX_UNSET_GLOBALS`      | `1` or `true`                 | By default "PyKX under q" will load some utility functions into the global namespace (eg. `print`). This variable prevents this. | |
| `PYKX_EXECUTABLE`         | File path                     | The path to use for the Python executable | |             
| `PYKX_PYTHON_LIB_PATH`    | File path                     | The path to use for loading libpython. | |
| `PYKX_PYTHON_BASE_PATH`   | File path                     | The path to use for the base directory of your Python installation. | |
| `PYKX_PYTHON_HOME_PATH`   | File path                     | The path to use for the base Python home directory (used to find site packages). | |
| `SKIP_UNDERQ`             | `1` or `true`                 | When importing PyKX from Python, PyKX will also load `pykx.q` under its embedded q. This variable skips this step. | `DEPRECATED`, please use `PYKX_SKIP_UNDERQ` |
| `UNSET_PYKX_GLOBALS`      | `1` or `true`                 | By default "PyKX under q" will load some utility functions into the global namespace (eg. `print`). This variable prevents this. | `DEPRECATED`, please use `PYKX_UNSET_GLOBALS` |


### q Cloud Edition features with Insights Core (Linux only)

On Linux, the q Cloud Edition features, coming with Insights Core, can be used to read data from Cloud Storage (AWS S3, Google Cloud Storage, Azure Blob Storage). Credentials to access the Cloud Storage can be passed using specific environment variables. For more information, see the two following links:

- https://code.kx.com/insights/core/objstor/main.html#environment-variables
- https://code.kx.com/insights/1.2/core/kurl/kurl.html#automatic-registration-using-credential-discovery


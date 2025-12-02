---
title: Configure PyKX 
description: How to configure PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, data, convert
---

# Configure PyKX 

_This page provides details on how to configure PyKX using a configuration file and/or environment variables._

To modify the underlying behavior of PyKX based on your specific use-case, check out your [options](#options) for use-case/performance tuned optimizations of the library. You can configure them using a [configuration file](#configuration-file) or [environment variables](#environment-variables) as described below. 

!!! warning "Important: In all cases, environment variable definitions take precedence over definitions within the configuration file."

## Configuration file

If you choose to use a configuration file `#!python .pykx-config` to define your options for PyKX initialization, here's an example of a `#!python .pykx-config` file which operates according to `#!python *.toml` syntax:

```bash
[default]
PYKX_IGNORE_QHOME="true"
PYKX_KEEP_LOCAL_TIMES="true"

[test]
PYKX_GC="true"
PYKX_RELEASE_GIL="true"

[beta]
PYKX_BETA_FEATURES="true"
```

On import of PyKX, the file `#!python .pykx-config` is searched for according to the following path ordering. The first location containing a `#!python .pykx-config` file is used for definition of the PyKX configuration:

| **Order** | **Location**                                     |
|-----------|--------------------------------------------------|
| 1.        | `Path('.')`                                      |
| 2.        | `Path(os.getenv('PYKX_CONFIGURATION_LOCATION'))` |
| 3.        | `Path.home()`                                    |

When loading this file, unless otherwise specified, PyKX uses the profile `#!python default`. To configure non-default profiles from within this file, set an environment variable `#!python PYKX_PROFILE` prior to loading of PyKX, for example using the above configuration file.

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

To add values to your configuration file you can modify the file directly or alternatively use the helper function `#!python kx.util.add_to_config` as follows for example

```python
>>> import pykx as kx
>>> kx.util.add_to_config({'PYKX_GC': 'True', 'PYKX_BETA_FEATURES': 'True'})

Configuration updated at: /Users/conormccarthy/.pykx-config.
Profile updated: default.
Successfully added:
	- PYKX_GC = True
	- PYKX_BETA_FEATURES = True
```

## Environment variables

If you wish to configure the [options](#options) as environment variables, before importing PyKX, you can, for example, define the environment variables to use: 

```python
>>> import os
>>> os.environ['PYKX_RELEASE_GIL'] = '1'
>>> os.environ['PYKX_GC'] = '1'
>>> import pykx as kx
>>> kx.config.k_gc
True
```

## Options

You have various options to tune PyKX behavior at run time. You must set these variables before importing PyKX. They remain effective throughout the execution of the PyKX process.

### General

To enable or disable advanced features of PyKX across all modes of operation, use the following variables:

| **Option**                      | **Default** | **Values**                                                            | **Description**                                                                                                                                                                                                                                                                                                                                                                                   |
|---------------------------------|-------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PYKX_BETA_FEATURES`            | `False`     | `1` or `true`                                                         | Enable all Beta features supplied with PyKX allowing users to test and prototype code slated for later releases.                                                                                                                                                                                                                                                                                  |
| `PYKX_QDEBUG`                   | `False`     | `1` or `true`                                                         | Enable retrieval of backtrace information on error being raised when executing q functions, this can alternatively be enabled by setting `debug=True` as a keyword in calls to `kx.q`.                                                                                                                                                                                                            |
| `PYKX_IGNORE_QHOME`             | `False`     | `1` or `true`                                                         | When loading PyKX on a machine with an existing q installation (and the environment variable `QHOME` set to the installation folder), PyKX will look within this directory for q scripts their dependencies. It will then symlink these files to make them available to load under PyKX. This variable instructs PyKX to not perform this symlinking.                                             |
| `PYKX_KEEP_LOCAL_TIMES`         | `False`     | `1` or `true`                                                         | When converting a Python datetime object to q, PyKX will translate the Python datetime into UTC before the conversion. This variable instructs PyKX to convert the Python datetime using the local time zone.                                                                                                                                                                                     |
| `PYKX_ALLOCATOR`                | `False`     | `1` or `true`                                                         | When converting a Numpy array to q, PyKX implements a full data copy in order to translate the Numpy array to q representation in memory. When this is set PyKX implements [NEP-49](https://numpy.org/neps/nep-0049.html) which allows q to handle memory allocation of all Numpy arrays so they can be converted more efficiently to q. This avoids the need to resort to a copy where possible. |
| `PYKX_GC`                       | `False`     | `1` or `true`                                                         | When PYKX_ALLOCATOR is enabled, PyKX can trigger q garbage collector when Numpy arrays allocated by PyKX are deallocated. This variable enables this behavior which will release q memory to the OS following deallocation of the Numpy array at the cost of a small overhead.                                                                                                                    |
| `PYKX_LOAD_PYARROW_UNSAFE`      | `False`     | `1` or `true`                                                         | By default, PyKX uses a subprocess to import pyarrow as it can result in a crash when the version of pyarrow is incompatible. This variable will trigger a normal import of pyarrow and importing PyKX should be slightly faster.                                                                                                                                                                 |
| `PYKX_MAX_ERROR_LENGTH`         | `256`       | size in characters                                                    | By default, PyKX reports IPC connection errors with a message buffer of size 256 characters. This allows the length of these error messages to be modified reducing the chance of excessive error messages polluting logs.                                                                                                                                                                        |
| `PYKX_NOQCE`                    | `False`     | `1` or `true`                                                         | On Linux, PyKX comes with q Cloud Edition features from [Insights Core](https://code.kx.com/insights/core/). This variable allows a user to skip the loading of q Cloud Edition functionality, saving some time when importing PyKX but removing access to possibly supported additional functionality.                                                                                           |
| `PYKX_Q_LIB_LOCATION`           | `UNSET`     | Path to a directory containing q libraries necessary for loading PyKX | See [here](../release-notes/changelog.md#pykx-131) for detailed information. This allows a user to store the PyKX libraries: `q.so`, `q.k` etc. separately from their Python installation. This is required for some enterprise use-cases.                                                                                                                                                        |
| `PYKX_RELEASE_GIL`              | `False`     | `1` or `true`                                                         | When PYKX_RELEASE_GIL is enabled the Python Global Interpreter Lock will not be held when calling into q.                                                                                                                                                                                                                                                                                         |
| `PYKX_Q_LOCK`                   | `False`     | `1` or `true`                                                         | When PYKX_Q_LOCK is enabled a re-entrant lock is added around calls into q, this lock will stop multiple threads from calling into q at the same time. This allows embedded q to be thread safe even when using PYKX_RELEASE_GIL.                                                                                                                                                                 |
| `PYKX_DEBUG_INSIGHTS_LIBRARIES` | `False`     | `1` or `true`                                                         | If the insights libraries failed to load this variable can be used to print out the full error output for debugging purposes.                                                                                                                                                                                                                                                                     |
| `PYKX_UNLICENSED`               | `False`     | `1` or `true`                                                         | Set PyKX to make use of the library in `unlicensed` mode at all times.                                                                                                                                                                                                                                                                                                                            |
| `PYKX_LICENSED`                 | `False`     | `1` or `true`                                                         | Set PyKX to make use of the library in `licensed` mode at all times. If licensed initialisation fails the import will error rather than bringing up the interactive license helper. Fallback to unlicensed mode is blocked.                                                                                                                                                                       |
| `PYKX_THREADING`                | `False`     | `1` or `true`                                                         | When importing PyKX start EmbeddedQ within a background thread. This allows calls into q from any thread to modify state, this environment variable is only supported for licensed users.                                                                                                                                                                                                         |
| `PYKX_NO_SIGNAL`                | `False`     | `1` or `true`                                                         | Skip overwriting of [signal](https://docs.python.org/3/library/signal.html) definitions by PyKX, these are presently overwritten by default to reset Pythonic default definitions with are reset by PyKX on initialisation in licensed modality.                                                                                                                                                  |
| `PYKX_4_1_ENABLED`              | `False`     | `1` or `true`                                                         | Load version 4.1 of `libq` when starting `PyKX` in licensed mode, this environment variable does not work without a valid `q` license.                                                                                                                                                                                                                                                            |
| `PYKX_JUPYTERQ`                 | `False`     | `1` or `true`                                                         | When enabled, any Jupyter Notebook will start in q first mode by default when PyKX is imported.                                                                                                                                                                                                                                                                                                   |
| `PYKX_Q_EXECUTABLE`             | `q`         | string denoting path to q executable                                  | This allows users to specify the location of the q executable which should be called when using making use of the `tick` module for defining streaming infrastructures                                                                                                                                                                                                                            |
| `PYKX_SUPPRESS_WARNINGS`        | `False`     | `1` or `true`                                                         | This allows the user to suppress warnings that have been suggested as sensible to be raised by users for PyKX in situations where edge cases can result in unexpected behaviour. Warnings in scenarios where a decision has been made to not support behaviour explicitly rather than where user discretion is required are still maintained.                                                     |
| `PYKX_CONFIGURATION_LOCATION`   | `.`         | The path to the folder containing the `.pykx-config` file.            | This allows users to specify a location other than the `.` or a users `home` directory to store their configuration file outlined [here](#configuration-file)                                                                                                                                                                                                                                     |
| `PYKX_CONFIGURATION_PROFILE`    | `default`   | The "profile" defined in `.pykx-config` file to be used.              | Users can specify which set of configuration variables are to be used by modifying the `PYKX_CONFIGURATION_PROFILE` variable see [here](#configuration-file) for more details. Note that this configuration can only be used as an environment variable.                                                                                                                                          |

To set the environment for q (embedded in PyKX, in licensed mode), use the variables below:

| **Variable** | **Values**                                                               | **Description**                                                                                                          |
|--------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `QARGS`      | See link                                                                 | Command-line flags to pass to q, see [here](https://code.kx.com/q/basics/cmdline/) for more information.                 |
| `QHOME`      | Path to the users q installation folder                                  | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information.                          |
| `QLIC`       | Path to the folder where the q license should be found                   | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information.                          |
| `QINIT`      | Path to an additional `*.q` file loaded after `PyKX` has initialized     | See [here](https://code.kx.com/q4m3/14_Introduction_to_Kdb%2B/#1481-the-environment-variables) for more information.     |

If no license is found, set the following variables either in configuration or as environment variables to define the `kc.lic` or `k4.lic` license used by PyKX:

| **Variable**        | **Description**                                                                                |
|---------------------|------------------------------------------------------------------------------------------------|
| `KDB_LICENSE_B64`   | This should contain the base-64 encoded contents of a valid `kc.lic` file with `pykx` enabled. |
| `KDB_K4LICENSE_B64` | This should contain the base-64 encoded contents of a valid `k4.lic` file with `pykx` enabled. |

#### PyKX QARGS supported additions

When using PyKX, you can define `#!python QARGS` to modify its behavior during initialization in a Linux environment. Here are some of the values you can use for `#!python QARGS`:

| **Input**      | **Description**                                                                 |
|----------------|---------------------------------------------------------------------------------|
| `--no-qce`     | Ensure that no kdb Insights libraries are loaded at initialization of PyKX.     |
| `--no-kurl`    | Ensure that the kdb Insights `kurl` library is not loaded at initialization.    |
| `--no-objstor` | Ensure that the kdb Insights `objstor` library is not loaded at initialization. |
| `--no-qlog`    | Ensure that the kdb Insights `qlog` library is not loaded at initialization.    |
| `--no-sql`     | Ensure that the kdb Insights `sql` library is not loaded at initialization.     |

### PyKX under q

You can load PyKX and [use it from a q session](../pykx-under-q/intro.md). The following variables are specific to this mode of operation:

| **Variable**              | **Values**                    | **Description**                                                                                                                                                                                                              |
|---------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PYKX_DEFAULT_CONVERSION` | `py`, `np`, `pd`, `pa` or `k` | Default conversion to apply when passing q objects to Python. Converting to Numpy (`np`) by default.                                                                                                                         |
| `PYKX_SKIP_UNDERQ`        | `1` or `true`                 | When importing PyKX from Python, PyKX also loads `pykx.q` under its embedded q. This variable skips this step.                                                                                                               |
| `PYKX_EXECUTABLE`         | File path                     | The path to use for the Python executable                                                                                                                                                                                    |
| `PYKX_USE_FIND_LIBPYTHON` | `1` or `true`                 | Should the Python package [`find-libpython`](https://pypi.org/project/find-libpython/) be used to determine the location of `libpython.[so|dll]`, this manually could be done by setting the location `PYKX_PYTHON_LIB_PATH` |
| `PYKX_PYTHON_LIB_PATH`    | File path                     | The path to use for loading libpython.                                                                                                                                                                                       |
| `PYKX_PYTHON_BASE_PATH`   | File path                     | The path to use for the base directory of your Python installation.                                                                                                                                                          |
| `PYKX_PYTHON_HOME_PATH`   | File path                     | The path to use for the base Python home directory (used to find site packages).                                                                                                                                             |


### q Cloud Edition features with Insights Core (Linux only)

On Linux, the q Cloud Edition features coming with Insights Core can be used to read data from Cloud Storage (AWS S3, Google Cloud Storage, Azure Blob Storage). Credentials to access the Cloud Storage can be passed using specific environment variables. For more information, go to:

- [kdb Insights SDK environment variables](https://code.kx.com/insights/core/objstor/main.html#environment-variables)
- [kdb Insights SDK automatic registration using credential discovery](https://code.kx.com/insights/core/kurl/kurl.html#automatic-registration-using-credential-discovery)


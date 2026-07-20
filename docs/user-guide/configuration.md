---
title: Configure KDB-X Python 
description: How to configure KDB-X Python
date: July 2024
author: KX Systems, Inc.,
tags: KDB-X Python, data, convert
---

# Configure KDB-X Python 

_This page provides details on how to configure KDB-X Python using a configuration file and/or environment variables._

To modify the underlying behavior of KDB-X Python based on your specific use-case, check out your [options](#options) for use-case/performance tuned optimizations of the library. You can configure them using a [configuration file](#configuration-file) or [environment variables](#environment-variables) as described below. 

!!! warning "Important: In all cases, environment variable definitions take precedence over definitions within the configuration file."

## Configuration file

If you choose to use a configuration file `#!python config-pykx` to define your options for KDB-X Python initialization, here's an example of a `#!python config-pykx` file which operates according to `#!python *.toml` syntax:

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

On import of KDB-X Python, the file `#!python config-pykx` is searched for according to the following path ordering. The first found file is used for definition of the KDB-X Python configuration:

| **Order** | **Location**                                     |
|-----------|--------------------------------------------------|
| 1.        | `Path(os.getenv('PYKX_CONFIGURATION_LOCATION'))` |
| 2.        | `Path.home()/'.kx/config-pykx'`                  |

When loading this file, unless otherwise specified, KDB-X Python uses the profile `#!python default`. To configure non-default profiles from within this file, set an environment variable `#!python PYKX_PROFILE` prior to loading of KDB-X Python, for example using the above configuration file.

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

Configuration updated at: /Users/user/config-pykx.
Profile updated: default.
Successfully added:
	- PYKX_GC = True
	- PYKX_BETA_FEATURES = True
```

## Environment variables

If you wish to configure the [options](#options) as environment variables, before importing KDB-X Python, you can, for example, define the environment variables to use: 

```python
>>> import os
>>> os.environ['PYKX_RELEASE_GIL'] = '1'
>>> os.environ['PYKX_GC'] = '1'
>>> import pykx as kx
>>> kx.config.k_gc
True
```

## Options

You have various options to tune KDB-X Python behavior at run time. You must set these variables before importing KDB-X Python. They remain effective throughout the execution of the KDB-X Python process.

### General

To enable or disable advanced features of KDB-X Python across all modes of operation, use the following variables:

| **Option**                      | **Default** | **Values**                                                            | **Description**                                                                                                                                                                                                                                                                                                                                                                                   |
|---------------------------------|-------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PYKX_BETA_FEATURES`            | `False`     | `1` or `true`                                                         | Enable all Beta features supplied with KDB-X Python allowing users to test and prototype code slated for later releases.                                                                                                                                                                                                                                                                                  |
| `PYKX_QDEBUG`                   | `False`     | `1` or `true`                                                         | Enable retrieval of backtrace information on error being raised when executing q functions, this can alternatively be enabled by setting `debug=True` as a keyword in calls to `kx.q`.                                                                                                                                                                                                            |
| `PYKX_IGNORE_QHOME`             | `False`     | `1` or `true`                                                         | When loading KDB-X Python on a machine with an existing q installation (and the environment variable `QHOME` set to the installation folder), KDB-X Python will look within this directory for q scripts their dependencies. It will then symlink these files to make them available to load under KDB-X Python. This variable instructs KDB-X Python to not perform this symlinking.                                             |
| `PYKX_KEEP_LOCAL_TIMES`         | `False`     | `1` or `true`                                                         | When converting a Python datetime object to q, KDB-X Python will translate the Python datetime into UTC before the conversion. This variable instructs KDB-X Python to convert the Python datetime using the local time zone.                                                                                                                                                                                     |
| `PYKX_NO_ALLOCATOR`             | `False`     | `1` or `true`                                                         | KDB-X Python implements [NEP-49](https://numpy.org/neps/nep-0049.html) which allows q to handle memory allocation of all Numpy arrays so they can be converted more efficiently to q. This avoids the need to resort to a copy where possible. Use `PYKX_NO_ALLOCATOR` to turn this behaviour off, resulting in full data copies.                                                                         |
| `PYKX_GC`                       | `False`     | `1` or `true`                                                         | KDB-X Python can trigger q garbage collector when Numpy arrays allocated by KDB-X Python are deallocated. This variable enables this behavior which will release q memory to the OS following deallocation of the Numpy array at the cost of a small overhead. Only usable when `PYKX_NO_ALLOCATOR` is not set.                                                                                                   |
| `PYKX_LOAD_PYARROW_UNSAFE`      | `False`     | `1` or `true`                                                         | By default, KDB-X Python uses a subprocess to import pyarrow as it can result in a crash when the version of pyarrow is incompatible. This variable will trigger a normal import of pyarrow and importing KDB-X Python should be slightly faster.                                                                                                                                                                 |
| `PYKX_MAX_ERROR_LENGTH`         | `256`       | size in characters                                                    | By default, KDB-X Python reports IPC connection errors with a message buffer of size 256 characters. This allows the length of these error messages to be modified reducing the chance of excessive error messages polluting logs.                                                                                                                                                                        |
| `PYKX_QCE`                      | `False`     | `1` or `true`                                                         | On Linux, KDB-X Python comes with q Cloud Edition features from [Insights Core](https://code.kx.com/insights/core/). This variable allows a user to opt in the loading of q Cloud Edition functionality.                                                                                                                                                                                                  |
| `PYKX_Q_LIB_LOCATION`           | `UNSET`     | Path to a directory containing q libraries necessary for loading KDB-X Python | See [here](../release-notes/changelog.md#pykx-131) for detailed information. This allows a user  to store the KDB-X Python libraries: `q.so`, `q.k` etc. separately from their Python installation. This is required for some enterprise use-cases. **Note:** If using this override you must manage the folder to ensure its contents match the version of KDB-X Python you are using                            |
| `PYKX_RELEASE_GIL`              | `False`     | `1` or `true`                                                         | When PYKX_RELEASE_GIL is enabled the Python Global Interpreter Lock will not be held when calling into q.                                                                                                                                                                                                                                                                                         |
| `PYKX_Q_LOCK`                   | `False`     | `1` or `true`                                                         | When PYKX_Q_LOCK is enabled a re-entrant lock is added around calls into q, this lock will stop multiple threads from calling into q at the same time. This allows embedded q to be thread safe even when using PYKX_RELEASE_GIL.                                                                                                                                                                 |
| `PYKX_DEBUG_INSIGHTS_LIBRARIES` | `False`     | `1` or `true`                                                         | If the insights libraries failed to load this variable can be used to print out the full error output for debugging purposes.                                                                                                                                                                                                                                                                     |
| `PYKX_UNLICENSED`               | `False`     | `1` or `true`                                                         | Set KDB-X Python to make use of the library in `unlicensed` mode at all times.                                                                                                                                                                                                                                                                                                                            |
| `PYKX_LICENSED`                 | `False`     | `1` or `true`                                                         | Set KDB-X Python to make use of the library in `licensed` mode at all times. If licensed initialisation fails the import will error rather than bringing up the interactive license helper. Fallback to unlicensed mode is blocked.                                                                                                                                                                       |
| `PYKX_THREADING`                | `False`     | `1` or `true`                                                         | When importing KDB-X Python start EmbeddedQ within a background thread. This allows calls into q from any thread to modify state, this environment variable is only supported for licensed users.                                                                                                                                                                                                         |
| `PYKX_NO_SIGNAL`                | `False`     | `1` or `true`                                                         | Skip overwriting of [signal](https://docs.python.org/3/library/signal.html) definitions by KDB-X Python, these are presently overwritten by default to reset Pythonic default definitions with are reset by KDB-X Python on initialisation in licensed modality.                                                                                                                                                  |
| `PYKX_JUPYTERQ`                 | `False`     | `1` or `true`                                                         | When enabled, any Jupyter Notebook will start in q first mode by default when KDB-X Python is imported.                                                                                                                                                                                                                                                                                                   |
| `PYKX_Q_EXECUTABLE`             | `q`         | string denoting path to q executable                                  | This allows users to specify the location of the q executable which should be called when using making use of the `tick` module for defining streaming infrastructures                                                                                                                                                                                                                            |
| `PYKX_SUPPRESS_WARNINGS`        | `False`     | `1` or `true`                                                         | This allows the user to suppress warnings that have been suggested as sensible to be raised by users for KDB-X Python in situations where edge cases can result in unexpected behavior. Warnings in scenarios where a decision has been made to not support behavior explicitly rather than where user discretion is required are still maintained.                                                     |
| `PYKX_CONFIGURATION_LOCATION`   | `.`         | The path to the folder containing the `config-pykx` file.            | This allows users to specify a location other than the `.` or a users `home` directory to store their configuration file outlined [here](#configuration-file)                                                                                                                                                                                                                                      |
| `PYKX_CONFIGURATION_PROFILE`    | `default`   | The "profile" defined in `config-pykx` file to be used.              | Users can specify which set of configuration variables are to be used by modifying the `PYKX_CONFIGURATION_PROFILE` variable see [here](#configuration-file) for more details. Note that this configuration can only be used as an environment variable.                                                                                                                                           |

To set the environment for q (embedded in KDB-X Python, in licensed mode), use the variables below:

| **Variable** | **Values**                                                               | **Description**                                                                                                          |
|--------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `QARGS`      | See link                                                                 | Command-line flags to pass to q, see [here](https://code.kx.com/q/basics/cmdline/) for more information.                 |
| `QHOME`      | Path to the users q installation folder                                  | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information.                          |
| `QLIC`       | Path to the folder where the q license should be found                   | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information.                          |
| `QINIT`      | Path to an additional `*.q` file loaded after KDB-X Python has initialized     | See [here](https://code.kx.com/q4m3/14_Introduction_to_Kdb%2B/#1481-the-environment-variables) for more information.     |
| `QCFG`       | Location of KX config file. Defaults to `~/.kx/config`                   |                                                                                                                          |

If no license is found, set the following variables either in configuration or as environment variables to define the `kc.lic` or `k4.lic` license used by KDB-X Python:

| **Variable**        | **Description**                                                                                |
|---------------------|------------------------------------------------------------------------------------------------|
| `KDB_LICENSE_B64`   | This should contain the base-64 encoded contents of a valid `kc.lic` file with `pykx` enabled. |
| `KDB_K4LICENSE_B64` | This should contain the base-64 encoded contents of a valid `k4.lic` file with `pykx` enabled. |

#### KDB-X Python QARGS supported additions

When using KDB-X Python, you can define `#!python QARGS` to modify its behavior during initialization in a Linux environment. Here are some of the values you can use for `#!python QARGS`:

| **Input**   | **Description**                                                                 |
|-------------|---------------------------------------------------------------------------------|
| `--qce`     | Loads  KDB-X libraries at initialization of KDB-X.                           |
| `--kurl`    | Ensure that the KDB-X `kurl` library is not loaded at initialization.    |
| `--objstor` | Ensure that the KDB-X `objstor` library is not loaded at initialization. |
| `--qlog`    | Ensure that the KDB-X `qlog` library is not loaded at initialization.    |
| `--sql`     | Ensure that the KDB-X `sql` library is not loaded at initialization.     |

### KDB-X Python under q

You can load KDB-X Python and [use it from a q session](../pykx-under-q/intro.md). The following variables are specific to this mode of operation:

| **Variable**              | **Values**                    | **Description**                                                                                                                                                                                                              |
|---------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PYKX_DEFAULT_CONVERSION` | `py`, `np`, `pd`, `pa` or `k` | Default conversion to apply when passing q objects to Python. Converting to Numpy (`np`) by default.                                                                                                                         |
| `PYKX_SKIP_UNDERQ`        | `1` or `true`                 | When importing KDB-X Python from Python, KDB-X Python also loads `pykx.q` under its embedded q. This variable skips this step.                                                                                                               |
| `PYKX_EXECUTABLE`         | File path                     | The path to use for the Python executable                                                                                                                                                                                    |
| `PYKX_USE_FIND_LIBPYTHON` | `1` or `true`                 | Should the Python package [`find-libpython`](https://pypi.org/project/find-libpython/) be used to determine the location of `libpython.[so|dll]`, this manually could be done by setting the location `PYKX_PYTHON_LIB_PATH` |
| `PYKX_PYTHON_LIB_PATH`    | File path                     | The path to use for loading libpython.                                                                                                                                                                                       |
| `PYKX_PYTHON_BASE_PATH`   | File path                     | The path to use for the base directory of your Python installation.                                                                                                                                                          |
| `PYKX_PYTHON_HOME_PATH`   | File path                     | The path to use for the base Python home directory (used to find site packages).                                                                                                                                             |


### q Cloud Edition features with Insights Core (Linux only)

On Linux, the q Cloud Edition features coming with Insights Core can be used to read data from Cloud Storage (AWS S3, Google Cloud Storage, Azure Blob Storage). Credentials to access the Cloud Storage can be passed using specific environment variables. For more information, go to:

- [kdb Insights SDK environment variables](https://code.kx.com/insights/core/objstor/main.html#environment-variables)
- [kdb Insights SDK automatic registration using credential discovery](https://code.kx.com/insights/core/kurl/kurl.html#automatic-registration-using-credential-discovery)


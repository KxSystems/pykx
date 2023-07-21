# Modifying PyKX using environment variables

The following environment variables can be used to tune PyKX behavior at run time. These variables need to be set before attempting to import PyKX and will take effect for the duration of the execution of the PyKX process.


## General

The following variables can be used to enable or disable advanced features of PyKX:

| Variable                   | Values                                                                | Description                                                                                                                                                                                                                                                                                                                                                                                       |
|----------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `IGNORE_QHOME`             | `1` or `true`                                                         | When loading PyKX on a machine with an existing q installation (and the environment variable `QHOME` set to the installation folder), PyKX will look within this directory for q scripts and dependencies to load. This variable instructs PyKX to ignore the existing q installation and not load from this directory.                                                                           |
| `KEEP_LOCAL_TIMES`         | `1` or `true`                                                         | When converting a Python datetime object to q, PyKX will translate the Python datetime into UTC before the conversion. This variable instructs PyKX to convert the Python datetime using the local timezone.                                                                                                                                                                                      |
| `PYKX_ALLOCATOR`           | `1` or `true`                                                         | When converting a Numpy array to q, PyKX implements a full data copy in order to translate the Numpy array to q representation in memory. When this is set PyKX implements [NEP-49](https://numpy.org/neps/nep-0049.html) which allows q to handle memory allocation of all Numpy arrays so they can be converted more efficiently to q. This avoids the need to resort to a copy where possible. |
| `PYKX_ENABLE_PANDAS_API`   | `1` or `true`                                                         | Enable the Pandas API for `pykx.Table` objects                                                                                                                                                                                                                                                                                                                                                    |
| `PYKX_GC`                  | `1` or `true`                                                         | When PYKX_ALLOCATOR is enabled, PyKX can trigger q garbage collector when Numpy arrays allocated by PyKX are deallocated. This variable enables this behavior which will release q memory to the OS following deallocation of the numpy array at the cost of a small overhead.                                                                                                                    |
| `PYKX_LOAD_PYARROW_UNSAFE` | `1` or `true`                                                         | By default, PyKX uses a subprocess to import pyarrow as it can result in a crash when the version of pyarrow is incompatible. This variable will trigger a normal import of pyarrow and importing PyKX should be slightly faster.                                                                                                                                                                 |
| `PYKX_MAX_ERROR_LENGTH`    | size in characters                                                    | By default, PyKX reports IPC connection errors with a message buffer of size 256 characters. This allows the length of these error messages to be modified reducing the chance of excessive error messages polluting logs.                                                                                                                                                                        |
| `PYKX_NOQCE`               | `1` or `true`                                                         | On Linux, PyKX comes with q Cloud Edition features from Insights Core (https://code.kx.com/insights/1.2/core/). This variable allows a user to skip the loading of q Cloud Edition functionality, saving some time when importing PyKX but removing access to possibly supported additional functionality.                                                                                        |
| `PYKX_Q_LIB_LOCATION`      | Path to a directory containing q libraries necessary for loading PyKX | See [here](https://code.kx.com/pykx/changelog.html#pykx-131) for detailed information. This allows a user to centralise the q libraries, `q.k`, `read.q`, `libq.so` etc to a managed location within their environment which is decentralised from the Python installation. This is required for some enterprise use-cases.                                                                       |
| `PYKX_RELEASE_GIL`         | `1` or `true`                                                         | When PYKX_RELEASE_GIL is enabled the Python Global Interpreter Lock will not be held when calling into q.                                                                                                                                                                                                                                                                                         |
| `PYKX_Q_LOCK`              | `1` or `true`                                                         | When PYKX_Q_LOCK is enabled a reentrant lock is added around calls into q, this lock will stop multiple threads from calling into q at the same time. This allows embedded q to be threadsafe even when using PYKX_RELEASE_GIL.                                                                                                                                                                   |
| `PYKX_DEBUG_INSIGHTS_LIBRARIES` | `1` or `true`                                                    | If the insights libraries failed to load this variable can be used to print out the full error output for debugging purposes.                                                                                                                                                                                                                                                                     |

The variables below can be used to set the environment for q (embedded in PyKX, in licensed mode):

| Variable | Values   | Description |
|----------|----------|-------------|
| `QARGS`  | See link | Command-line flags to pass to q, see [here](https://code.kx.com/q/basics/cmdline/) for more information. |
| `QHOME`  | Path to the users q installation folder | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information. |
| `QLIC`   | Path to the folder where the q license should be found | See [here](https://code.kx.com/q/learn/install/#step-5-edit-your-profile) for more information. |


## PyKX under q

PyKX can be loaded and used from a q session (see [here](running_under_q.md) for more information). The following variables are specific to this mode of operation.

| Variable | Values | Description |
|----------|--------|-------------|
| `PYKX_DEFAULT_CONVERSION` | `py`, `np`, `pd`, `pa` or `k` | Default conversion to apply when passing q objects to Python. Converting to Numpy (`np`) by default. |
| `SKIP_UNDERQ` | `1` or `true` | When importing PyKX from Python, PyKX will also load `pykx.q` under its embedded q. This variable skip this step. |
| `UNSET_PYKX_GLOBALS` | `1` or `true` | By default "PyKX under q" will load some utility functions into the global namespace (eg. `print`). This variable prevents this. |
| `PYKX_PYTHON_LIB_PATH` | File path | The path to use for loading libpython. |
| `PYKX_PYTHON_BASE_PATH` | File path | The path to use for the base directory of your Python installation. |
| `PYKX_PYTHON_HOME_PATH` | File path | The path to use for the base Python home directory (used to find site packages). |

## q Cloud Edition features with Insights Core (Linux only)

On Linux, the q Cloud Edition features, coming with Insights Core, can be used to read data from Cloud Storage (AWS S3, Google Cloud Storage, Azure Blob Storage). Credentials to access the Cloud Storage can be passed using specific environment variables. For more information, see the two following links:

- https://code.kx.com/insights/core/objstor/main.html#environment-variables
- https://code.kx.com/insights/1.2/core/kurl/kurl.html#automatic-registration-using-credential-discovery


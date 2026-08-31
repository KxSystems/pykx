---
title: Configure KDB-X Python
description: Set environment variables, customize runtime behavior, and configure advanced options for KDB-X Python.
last_updated: August 2026
author: KX Systems, Inc., a subsidiary of KX Software Limited
tags: KDB-X Python, data, convert
---

# Configure KDB-X Python

_Configure KDB-X Python with a configuration file or environment variables, and look up every option it accepts._

Set the [configuration options](#options) in a [configuration file](#configuration-file) or with [environment variables](#environment-variables).

!!! warning "Configuration precedence"

	Environment variables take precedence over values in the configuration file.

## Prerequisites

Before you start, make sure you have:

- [KDB-X Python installed](../getting-started/installing.md#install-from-pypi)
- A [KDB-X license](../getting-started/installing.md#install-a-kdb-x-license), if you intend to use options that require [licensed mode](advanced/modes.md)

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

When you import KDB-X Python, it searches these locations in order and uses the first `#!python config-pykx` file it finds:

| **Order** | **Location**                                     |
|-----------|--------------------------------------------------|
| 1.        | `Path(os.getenv('PYKX_CONFIGURATION_LOCATION'))` |
| 2.        | `Path.home()/'.kx/config-pykx'`                  |

KDB-X Python loads the `#!python default` profile unless you name another one. To select a different profile, set `#!python PYKX_PROFILE` before you import KDB-X Python – for example, using the configuration file above.

Two variables control which file and profile KDB-X Python loads. Because KDB-X Python reads them before it reads the file, set them as environment variables only:

| **Variable** | **Default** | **Description** |
| --- | --- | --- |
| `PYKX_CONFIGURATION_LOCATION` | Unset | Path to the configuration file to load, checked before `~/.kx/config-pykx`. |
| `PYKX_PROFILE` | `default` | Name of the profile within the configuration file to load. |

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

To add values, edit the file directly or call the helper function `#!python kx.util.add_to_config`:

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

To set the [options](#options) as environment variables, define them before you import KDB-X Python:

```python
>>> import os
>>> os.environ['PYKX_RELEASE_GIL'] = '1'
>>> os.environ['PYKX_GC'] = '1'
>>> import pykx as kx
>>> kx.config.k_gc
True
```

## Check the active configuration

Because environment variables override the configuration file, the value in effect is not always the one you set most recently. To see what KDB-X Python resolved, call `#!python kx.util.debug_environment`:

```python
>>> import pykx as kx
>>> kx.util.debug_environment()
```

It reports which file and profile KDB-X Python loaded, along with the value of every option:

```text
**** KDB-X Python Configuration File ****
File location: /usr/local/.kx/config-pykx
Used profile: default
Profile content: {'PYKX_Q_EXECUTABLE': '/usr/local/anaconda3/envs/qenv/q/m64/q'}

**** KDB-X Python Configuration Variables ****
PYKX_IGNORE_QHOME: False
PYKX_KEEP_LOCAL_TIMES: False
PYKX_NO_ALLOCATOR: False
PYKX_GC: False
```

The function also reports your Python, platform, license, and q details, which makes its full output a useful attachment for a support request. Refer to [`pykx.util.debug_environment`](../api/util.md#pykxutildebug_environment) for the complete output and the `detailed` and `return_info` arguments.

!!! tip "Check the variable, not the attribute"

	`#!python kx.config` also exposes resolved settings as attributes, but their names differ from the variables and `PYKX_NO_ALLOCATOR` is inverted: `#!python kx.config.k_allocator` reads `#!python True` when `PYKX_NO_ALLOCATOR` is unset. `#!python kx.util.debug_environment` prints the variable names themselves, so prefer it when confirming a setting.

## Configuration tasks

### Configure q for Real-Time Capture

[Real-Time Capture](advanced/streaming/index.md) starts q subprocesses. If the Python process cannot resolve the `q` command, set these values in the [configuration file](#configuration-file) or as [environment variables](#environment-variables):

| **Variable** | **Value** |
| --- | --- |
| `PYKX_Q_EXECUTABLE` | Full path to `QHOME/[lmw]64/q[.exe]` |
| `QHOME` | Directory where q is installed |

Set the values before importing KDB-X Python.

Those q subprocesses also inherit `QHOME` from the Python process. If they fail to start – particularly under kdb+ 4.0/4.1 – refer to [QHOME, symlinking, and subprocesses](#qhome-symlinking-and-subprocesses) and use `PYKX_IGNORE_QHOME` or [`PyKXReimport`](../api/reimporting.md).

### QHOME, symlinking, and subprocesses

By default, KDB-X Python takes over the `QHOME` environment variable when running in licensed mode, pointing it at an internal directory bundled with the `pykx` package rather than your own q installation. So the embedded q can still see the scripts and data in your original `QHOME`, KDB-X Python symlinks its contents into that internal directory. It deliberately excludes the bootstrap files `q.k` and `s.k_` from this symlinking.

This affects any q subprocess started after importing KDB-X Python – it inherits the internal `QHOME`, not your original one:

| **q subprocess** | **Result** |
|---|---|
| Python that imports KDB-X Python | Works – your original `QHOME` is restored automatically on import. |
| KDB-X 5.0 `q` | Works – KDB-X 5.0 doesn't need `q.k` and can find its own installation automatically. Note that `QHOME` is still *respected* if set, so pointing it at a directory with no license can still cause a license error. |
| kdb+ 4.0/4.1 `q` | Fails to start – kdb+ requires `QHOME` to point at an installation containing `q.k`, with no auto-discovery fallback. |

On Windows, symlinking often requires elevated privileges. If it fails, the internal `QHOME` may be missing your license too, so either q flavor can fail to start.

#### Run q subprocesses

If you need to start a q subprocess – particularly kdb+ 4.0/4.1 – choose one of the following:

- **`PYKX_IGNORE_QHOME="true"`** – Set this before importing KDB-X Python to stop it from modifying `QHOME`. If you've set `QHOME` yourself, KDB-X Python leaves it untouched for this process and any q subprocess.

	!!! warning
		If you *haven't* set `QHOME`, KDB-X Python still writes a fallback value (`~/.kx` if it exists, otherwise its own internal directory) into the environment. On a machine with neither a `QHOME` nor a `~/.kx`, a kdb+ subprocess can still inherit the internal `QHOME`. To guarantee a real `QHOME` for a kdb+ subprocess, set `QHOME` explicitly.

- **[`PyKXReimport`](../api/reimporting.md)** – Restores your original `QHOME` around a single q subprocess launch, then reverts to KDB-X Python's internal `QHOME`. Use this if you want KDB-X Python's default behavior everywhere except for specific q subprocess launches.

!!! note
	With `PYKX_IGNORE_QHOME="true"`, the embedded q no longer has access to scripts and data in your `QHOME` (that's what the symlinking normally provides). Put any q scripts you need on `QPATH` instead.

## Dependencies and bundled assets

### Required Python packages

`pip` installs these packages with KDB-X Python and selects versions compatible with the active Python version:

| **Package** | **Purpose** |
| --- | --- |
| [NumPy](https://pypi.org/project/numpy) | Converts KDB-X Python objects to NumPy arrays and supports NumPy functions on KDB-X Python data. |
| [pandas](https://pypi.org/project/pandas) | Converts data to pandas `Series` and `DataFrame` objects and supports some PyArrow conversions. |
| [pytz](https://pypi.org/project/pytz/) | Applies time-zone offsets during temporal data conversion. |
| [toml](https://pypi.org/project/toml/) | Parses the `config-pykx` configuration file. |
| [dill](https://pypi.org/project/dill) | Serializes Python objects for [remote functions](advanced/remote-functions.md) and [Real-Time Capture](advanced/streaming/index.md). |
| [requests](https://pypi.org/project/requests/) | Provides HTTP client functionality. |

### Optional Python packages

Install an optional dependency group with `python -m pip install "pykx[<extra>]"`.

| **Extra** | **Package** | **Purpose** |
| --- | --- | --- |
| `pyarrow` | `pyarrow>=3.0.0` | Converts KDB-X Python objects to and from PyArrow tables and arrays. |
| `debug` | `find-libpython~=0.2` | Locates the `libpython` library that [KDB-X Python under q](../pykx-under-q/intro.md) requires. |
| `streaming` | `psutil>=5.0.0` | Manages the q subprocesses that Real-Time Capture starts. |
| `dashboards` | `ast2json~=0.3` | Supports KX Dashboards Direct integration. |
| `streamlit` | `streamlit~=1.28` | Supports Streamlit integration. |
| `torch` | `torch>2.1` | Converts between `torch.Tensor` objects and KDB-X Python objects on supported platforms. |

### Optional system libraries

- `libssl` supports TLS on [IPC connections](../api/ipc.md).
- `libpthread` supports `PYKX_THREADING` on Linux and macOS.

### Bundled assets

KDB-X Python wheels include these native libraries:

| **Platform** | **Mode** | **File** | **Version** |
| --- | --- | --- | --- |
| Linux ARM | KDB-X | `libq.so` | 5.0.20260706 |
| Linux x86 | KDB-X | `libq.so` | 5.0.20260706 |
| macOS ARM | KDB-X | `libq.dylib` | 5.0.20260706 |
| macOS x86 | KDB-X | `libq.dylib` | 5.0.20260706 |
| Windows | KDB-X | `q.dll`, `q.lib` | 5.0.20260706 |
| Linux ARM | Unlicensed | `libe.so` | 2023.11.22 |
| Linux x86 | Unlicensed | `libe.so` | 2023.11.22 |
| macOS ARM | Unlicensed | `libe.so` | 2023.11.22 |
| macOS x86 | Unlicensed | `libe.so` | 2023.11.22 |
| Windows | Unlicensed | `e.dll`, `e.lib` | 2024.08.21 |

## Options

Use these options to tune KDB-X Python behavior at run time. Set them before you import KDB-X Python. Each setting then stays in effect until the process ends.

To turn on any option whose default is `False`, set it to `1` or `true`.

### Examples

These three routes are equivalent. Pick whichever suits how you start your application:

=== "Configuration file"

	```bash
	[default]
	PYKX_GC="true"
	PYKX_RELEASE_GIL="true"
	PYKX_MAX_ERROR_LENGTH="1024"
	```

	Refer to [Configuration file](#configuration-file) for where KDB-X Python looks for this file.

=== "Environment variables"

	```python
	>>> import os
	>>> os.environ['PYKX_GC'] = '1'
	>>> os.environ['PYKX_RELEASE_GIL'] = '1'
	>>> os.environ['PYKX_MAX_ERROR_LENGTH'] = '1024'
	>>> import pykx as kx
	```

=== "QARGS"

	```sh
	QARGS="--licensed --qce" python my_application.py
	```

	`#!python QARGS` passes flags to the embedded q. Refer to [KDB-X Python QARGS supported additions](#kdb-x-python-qargs-supported-additions).

### Licensing and startup

| **Variable** | **Default** | **Description** |
| --- | --- | --- |
| `PYKX_LICENSED` | `False` | Always run in licensed mode. Refer to [Modes of operation](advanced/modes.md). |
| `PYKX_UNLICENSED` | `False` | Always run in unlicensed mode. Refer to [Modes of operation](advanced/modes.md). |
| `PYKX_BETA_FEATURES` | `False` | Enable all beta features, so you can test and prototype code slated for later releases. |
| `PYKX_NO_SIGNAL` | `False` | Leave Python [signal](https://docs.python.org/3/library/signal.html) definitions untouched. By default, KDB-X Python restores Python's definitions, which embedded q replaces when it initializes in licensed mode. |

### q environment and libraries

| **Variable** | **Default** | **Description** |
| --- | --- | --- |
| `PYKX_IGNORE_QHOME` | `False` | Stop KDB-X Python taking over the `QHOME` environment variable: skip symlinking your `QHOME` contents into its internal directory, and leave `QHOME` pointing at your original installation, for this process and any q subprocess. Matters most when you start q subprocesses (especially kdb+ 4.0/4.1) or work on Windows. Refer to [QHOME, symlinking, and subprocesses](#qhome-symlinking-and-subprocesses). |
| `PYKX_Q_LIB_LOCATION` | Unset | Directory holding the q libraries KDB-X Python loads. Set it to store those libraries separately from your Python installation, which some enterprise deployments require. The directory must mirror the `lib` directory of your installed `pykx` package: the `*.q` and `*.k` scripts at the top level, plus the platform subdirectory (`l64`, `l64arm`, `m64`, `m64arm`, or `w64`) holding the native libraries. Maintain it yourself so its contents match your KDB-X Python version. |
| `PYKX_QCE` | `False` | On Linux, load the q Cloud Edition features that ship with [Insights Core](https://code.kx.com/insights/core/). |
| `PYKX_Q_EXECUTABLE` | `q` | Path to the q executable that the [`tick`](../api/tick.md) module calls when building streaming infrastructures. |

### Performance and concurrency

| **Variable** | **Default** | **Description** |
| --- | --- | --- |
| `PYKX_NO_ALLOCATOR` | `False` | Copy NumPy array data in full instead of letting q allocate it. By default, KDB-X Python implements [NEP-49](https://numpy.org/neps/nep-0049.html) so that q handles NumPy memory allocation and converts arrays more efficiently, avoiding a copy where it can. |
| `PYKX_GC` | `False` | Trigger the q garbage collector when Python deallocates a NumPy array that KDB-X Python allocated. This returns q memory to the OS at the cost of a small overhead. Requires `PYKX_NO_ALLOCATOR` to be unset. |
| `PYKX_RELEASE_GIL` | `False` | Release the Python Global Interpreter Lock when calling into q. |
| `PYKX_Q_LOCK` | `False` | Add a re-entrant lock around calls into q, which stops two threads calling into q at once. This keeps embedded q thread safe even when you set `PYKX_RELEASE_GIL`. |
| `PYKX_THREADING` | `False` | Start embedded q in a background thread, so a call into q from any thread can modify state. Licensed mode only. |
| `PYKX_LOAD_PYARROW_UNSAFE` | `False` | Import PyArrow directly, which speeds up importing KDB-X Python slightly. By default, KDB-X Python imports PyArrow in a subprocess, because an incompatible PyArrow version can crash the process. |

### Behavior and diagnostics

| **Variable** | **Default** | **Description** |
| --- | --- | --- |
| `PYKX_KEEP_LOCAL_TIMES` | `False` | Convert Python datetime objects using the local time zone. By default, KDB-X Python translates them to UTC first. |
| `PYKX_QDEBUG` | `False` | Return backtrace information when a q function raises an error. Alternatively, pass `debug=True` to a `kx.q` call. |
| `PYKX_DEBUG_INSIGHTS_LIBRARIES` | `False` | Print the full error output when the Insights libraries fail to load. |
| `PYKX_MAX_ERROR_LENGTH` | `256` | Message buffer length, in characters, that KDB-X Python uses to report IPC connection errors. Lower it to stop long error messages polluting your logs. |
| `PYKX_SUPPRESS_WARNINGS` | `False` | Suppress the warnings KDB-X Python raises where an edge case can cause unexpected behavior. KDB-X Python still warns about behavior it explicitly does not support. |
| `PYKX_JUPYTERQ` | `False` | Start every Jupyter notebook in q-first mode when you import KDB-X Python. |

### q environment variables

These variables set the environment for the q that KDB-X Python embeds in licensed mode:

| **Variable** | **Values** | **Description** |
| --- | --- | --- |
| `QARGS` | Command-line flags | Flags to pass to q. Refer to the [q command-line reference](https://code.kx.com/q/basics/cmdline/). |
| `QHOME` | Path to a directory | Your q installation folder. KDB-X Python manages this variable in licensed mode – refer to [QHOME, symlinking, and subprocesses](#qhome-symlinking-and-subprocesses). |
| `QLIC` | Path to a directory | The folder holding your q license. Refer to [Install a KDB-X license](../getting-started/installing.md#install-a-kdb-x-license). |
| `QINIT` | Path to a file | An extra `*.q` file that KDB-X Python loads once it finishes initializing. Refer to [Introduction to kdb+ environment variables](https://code.kx.com/q4m3/14_Introduction_to_Kdb%2B/#1481-the-environment-variables). |
| `QCFG` | Path to a file | Location of the KX config file. Defaults to `~/.kx/config`. |

### License variables

If KDB-X Python finds no license, set one of these variables to supply a base64-encoded license, either in configuration or as an environment variable. To obtain and install a license, refer to [Install a KDB-X license](../getting-started/installing.md#install-a-kdb-x-license); to renew or upgrade one, refer to [Manage your license](advanced/license.md).

| **Variable** | **Values** | **Description** |
| --- | --- | --- |
| `KDB_LICENSE_B64` | Base64-encoded `kc.lic` contents | Supplies a `kc.lic` license with `pykx` enabled. |
| `KDB_K4LICENSE_B64` | Base64-encoded `k4.lic` contents | Supplies a `k4.lic` license with `pykx` enabled. |

### KDB-X Python QARGS supported additions

When using KDB-X Python, you can define `#!python QARGS` to modify its behavior during initialization in a Linux environment. Here are some of the values you can use for `#!python QARGS`:

| **Input**   | **Description**                                                          |
|-------------|--------------------------------------------------------------------------|
| `--qce`     | Loads all the KDB-X libraries below at initialization.                   |
| `--kurl`    | Loads the KDB-X `kurl` library at initialization.                        |
| `--objstor` | Loads the KDB-X `objstor` library at initialization.                     |
| `--qlog`    | Loads the KDB-X `qlog` library at initialization.                        |
| `--sql`     | Loads the KDB-X `sql` library at initialization.                         |

!!! note "These flags are opt-in"

	KDB-X Python loads a library only when you set its flag. Earlier versions loaded them all by default and used `--no-qce`, `--no-kurl`, `--no-objstor`, `--no-qlog`, and `--no-sql` to opt out; those flags no longer exist. Refer to [Migrating from PyKX 3.\* to KDB-X Python 4.\*](../upgrades/3040.md#qargs-flag-updates).

### KDB-X Python under q

You can load KDB-X Python and [use it from a q session](../pykx-under-q/intro.md). These variables apply only to that mode:

| **Variable** | **Values** | **Description** |
| --- | --- | --- |
| `PYKX_DEFAULT_CONVERSION` | `py`, `np`, `pd`, `pa`, or `k` | Conversion to apply when passing q objects to Python. Defaults to NumPy (`np`). |
| `PYKX_SKIP_UNDERQ` | `1` or `true` | Skip loading `pykx.q` under embedded q, which KDB-X Python otherwise does when you import it from Python. |
| `PYKX_EXECUTABLE` | File path | Path to the Python executable. |
| `PYKX_USE_FIND_LIBPYTHON` | `1` or `true` | Use [`find-libpython`](https://pypi.org/project/find-libpython/) to locate `libpython.so` or `libpython.dll`. To set that location yourself, use `PYKX_PYTHON_LIB_PATH` instead. |
| `PYKX_PYTHON_LIB_PATH` | File path | Path KDB-X Python loads `libpython` from. |
| `PYKX_PYTHON_BASE_PATH` | File path | Base directory of your Python installation. |
| `PYKX_PYTHON_HOME_PATH` | File path | Base Python home directory, which KDB-X Python uses to find site packages. |

!!! note "These variables no longer reach child processes"

	Loading KDB-X Python under q no longer exports `PYKX_SKIP_UNDERQ`, and `#!python .pykx.setdefault` no longer exports `PYKX_DEFAULT_CONVERSION`. A Python process started from that q session imports with the full `#!python .pykx` API and its own default conversion. Refer to the [KDB-X Python under q changelog](../release-notes/underq-changelog.md).

### q Cloud Edition features with Insights Core (Linux only)

On Linux, the q Cloud Edition features from Insights Core read data from cloud storage: AWS S3, Google Cloud Storage, and Azure Blob Storage. Load these features with `PYKX_QCE` or the [QARGS flags](#kdb-x-python-qargs-supported-additions), then pass your cloud storage credentials in environment variables. For more information, go to:

- [kdb Insights SDK environment variables](https://code.kx.com/insights/core/objstor/main.html#environment-variables)
- [kdb Insights SDK automatic registration using credential discovery](https://code.kx.com/insights/core/kurl/kurl.html#automatic-registration-using-credential-discovery)

## Related topics

- [Compare the modes of operation](advanced/modes.md)
- [Use KDB-X Python in a Python subprocess](advanced/subprocess.md)
- [Install a KDB-X license](../getting-started/installing.md#install-a-kdb-x-license)
- [Manage your license](advanced/license.md)
- [Enable multithreading](advanced/threading.md)
- [Set up Real-Time Capture](advanced/streaming/index.md)
- [Review deprecated configuration options](../release-notes/deprecations.md)
- [Troubleshoot errors](../help/troubleshooting.md)

## Next steps

- [Run the quickstart](../getting-started/quickstart.md)
- [Explore KDB-X Python fundamentals](../examples/interface-overview.md)


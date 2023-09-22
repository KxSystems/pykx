# PyKX

## Introduction

PyKX is a Python first interface to the worlds fastest time-series database kdb+ and it's underlying vector programming language q. PyKX takes a Python first approach to integrating q/kdb+ with Python following 10+ years of integrations between these two languages. Fundamentally it provides users with the ability to efficiently query and analyze huge amounts of in-memory and on-disk time-series data.

This interface exposes q as a domain-specific language (DSL) embedded within Python, taking the approach that q should principally be used for data processing and management of databases. This approach does not diminish the ability for users familiar with q or those wishing to learn more about it from making the most of advanced analytics and database management functionality but rather empowers those who want to make use of the power of kdb+/q who lack this expertise to get up and running fast.

PyKX supports three principal use cases:

- It allows users to store, query, manipulate and use q objects within a Python process.
- It allows users to query external q processes via an IPC interface.
- It allows users to embed Python functionality within a native q session using it's under q functionality.

Users wishing to install the library can do so following the instructions [here](https://code.kx.com/pykx/getting-started/installing.html).

Once you have the library installed you can get up and running with PyKX following the quickstart guide [here](https://code.kx.com/pykx/getting-started/quickstart.html).

### What is q/kdb+?

Mentioned throughout the documentation q and kdb+ are respectively a highly efficient vector programming language and highly optimised time-series database used to analyse streaming, real-time and historical data. Used throughout the financial sector for 25+ years this technology has been a cornerstone of modern financial markets providing a storage mechanism for historical market data and tooling to make the analysis of this vast data performant.

Kdb+ is a high-performance column-oriented database designed to process and store large amounts of data. Commonly accessed data is available in RAM which makes it faster to access than disk stored data. Operating with temporal data types as a first class entity the use of q and it's query language qsql against this database creates a highly performant time-series analysis tool.

q is the vector programming language which is used for all interactions with kdb+ databases and which is known both for its speed and expressiveness.

For more information on using q/kdb+ and getting started with see the following links:

- [An introduction to q/kdb+](https://code.kx.com/q/learn/tour/)
- [Tutorial videos introducing kdb+/q](https://code.kx.com/q/learn/q-for-all/)

## Installation

### Installing PyKX using `pip`

Ensure you have a recent version of pip:

pip install --upgrade pip


Then install the latest version of PyKX with the following command:

```
pip install pykx
```

To install a specific version of PyKX run the following command replacing <INSERT_VERSION> with a specific released semver version of the interface

```
pip install pykx==<INSERT_VERSION>
```

**Warning:** Python packages should typically be installed in a virtual environment. [This can be done with the venv package from the standard library](https://docs.python.org/3/library/venv.html).

### PyKX License access and enablement

Installation of PyKX via pip provides users with access to the library with limited functional scope, full details of these limitations can be found [here](docs/user-guide/advanced/modes.md). To access the full functionality of PyKX you must first download and install a kdb+ license, this can be achieved either through use of a personal evaluation license or receipt of a commercial license.

#### Personal Evaluation License

The following steps outline the process by which a user can gain access to an install a kdb Insights license which provides access to PyKX

1. Visit https://kx.com/kdb-insights-personal-edition-license-download/ and fill in the attached form following the instructions provided.
2. On receipt of an email from KX providing access to your license download this file and save to a secure location on your computer.
3. Set an environment variable on your computer pointing to the folder containing the license file (instructions for setting environment variables on PyKX supported operating systems can be found [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/).
	* Variable Name: `QLIC`
	* Variable Value: `/user/path/to/folder`

#### Commercial Evaluation License

The following steps outline the process by which a user can gain access to an install a kdb Insights license which provides access to PyKX 

1. Visit https://kx.com/kdb-insights-commercial-evaluation-license-download/ and fill in the attached form following the instructions provided.
2. On receipt of an email from KX providing access to your license download this file and save to a secure location on your computer.
3. Set an environment variable on your computer pointing to the folder containing the license file (instructions for setting environment variables on PyKX supported operating systems can be found [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/).
	* Variable Name: `QLIC`
	* Variable Value: `/user/path/to/folder`

__Note:__ PyKX will not operate with a vanilla or legacy kdb+ license which does not have access to specific feature flags embedded within the license. In the absence of a license with appropriate feature flags PyKX will fail to initialise with full feature functionality.

### Supported Environments

KX only officially supports versions of PyKX built by KX, i.e. versions of PyKX installed from wheel files. Support for user-built installations of PyKX (e.g. built from the source distribution) is only provided on a best-effort basis. Currently, PyKX provides wheels for the following environments:

- Linux (`manylinux_2_17_x86_64`) with CPython 3.8-3.11
- macOS (`macosx_10_10_x86_64`) with CPython 3.8-3.11
- Windows (`win_amd64`) with CPython 3.8-3.11

### Dependencies

#### Python Dependencies

PyKX depends on the following third-party Python packages:

- `pandas~=1.2`
- `numpy~=1.22`

They are installed automatically by `pip` when PyKX is installed.

PyKX also has an optional Python dependency of `pyarrow>=3.0.0`, which can be included by installing the `pyarrow` extra, e.g. `pip install pykx[pyarrow]`

**Warning:** Trying to use the `pa` conversion methods of `pykx.K` objects or the `pykx.toq.from_arrow` method when PyArrow is not installed (or could not be imported without error) will raise a `pykx.PyArrowUnavailable` exception. `pyarrow` is supported Python 3.8-3.10 but remains in Beta for Python 3.11.

#### Optional Non-Python Dependencies

- `libssl` for TLS on [IPC connections](docs/api/ipc.md).

#### Windows Dependencies

To run q or PyKX on Windows, `msvcr100.dll` must be installed. It is included in the [Microsoft Visual C++ 2010 Redistributable](https://www.microsoft.com/en-ca/download/details.aspx?id=26999).

## Building from source

### Installing Dependencies

The full list of supported environments is detailed [here](https://code.kx.com/pykx/getting-started/installing.html#supported-environments). Installation of dependencies will vary on different platforms.

`apt` example:

```bash
apt-install python3 python3-venv build-essential python3-dev
```

`yum` example:

```bash
yum install python3 gcc gcc-c++ python3-devel.x86_64
```

Windows:

* [Python](https://www.python.org/downloads/windows/)
* [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/?q=build+tools).
* [dlfcn-win32](https://github.com/dlfcn-win32/dlfcn-win32). Can be installed using [Vcpkg](https://github.com/microsoft/vcpkg).
* `msvcr100.dll`. Available in [Microsoft Visual C++ 2010 Redistributable](https://www.microsoft.com/en-ca/download/details.aspx?id=26999).

### Building

Using a Python virtual environment is recommended:

```bash
python3 -m venv pykx-dev
source pykx-dev/bin/activate
```

Build and install PyKX:

```bash
cd pykx
pip3 install -U '.[all]'
```

To run PyKX in licensed mode ensure to follow the steps to receive a [Personal Evaluation License](https://code.kx.com/pykx/getting-started/installing.html#personal-evaluation-license)

Now you can run/test PyKX:

```bash
(pykx-dev) /data/pykx$ python
Python 3.10.6 (main, May 29 2023, 11:10:38) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pykx
>>> pykx.q('1+1')
pykx.LongAtom(pykx.q('2'))
```

### Testing

Contributions to the project must pass a linting check:

```bash
pflake8
```

Contributions to the project must include tests. To run tests:

```bash
export PATH="$PATH:/location/of/your/q/l64" # q must be on PATH for tests
export QHOME=/location/of/your/q #q needs QHOME available
python -m pytest -vvv -n 0 --no-cov --junitxml=report.xml
```

## PyKX License access and enablement

This work is dual licensed under [Apache 2.0](https://code.kx.com/pykx/license.html#apache-2-license) and the [Software License for q.so](https://code.kx.com/pykx/license.html#qso-license) and users are required to abide by the terms of both licenses in their entirety.

## Community Help

If you have any issues or questions you can post them to [community.kx.com](https://community.kx.com/). Also available on Stack Overflow are the tags [pykx](https://stackoverflow.com/questions/tagged/pykx) and [kdb](https://stackoverflow.com/questions/tagged/kdb).

## Customer Support

* Inquires or feedback: [`pykx@kx.com`](mailto:pykx@kx.com)
* Support for Licensed Subscribers: [support.kx.com](https://support.kx.com/support/home)

# Installing

## Installing PyKX Using `pip`

Ensure you have a recent version of `pip`:

```
pip install --upgrade pip
```

Then install the latest version of PyKX with the following command:

```
pip install pykx
```

To install a specific version of PyKX run the following command replacing `<INSERT_VERSION>` with a specific released [semver](https://semver.org/) version of the interface

```
pip install pykx==<INSERT_VERSION>
```

!!! Warning

    Python packages should typically be installed in a virtual environment. [This can be done with the venv package from the standard library](https://docs.python.org/3/library/venv.html).

## PyKX License access and enablement

Installation of PyKX via pip provides users with access to the library with limited functional scope, full details of these limitations can be found [here](../user-guide/advanced/modes.md). To access the full functionality of PyKX you must first download and install a kdb+ license, this can be achieved either through use of a personal evaluation license or receipt of a commercial license.

### Personal Evaluation License

The following steps outline the process by which a user can gain access to an install a kdb Insights license which provides access to PyKX

1. Visit https://kx.com/kdb-insights-personal-edition-license-download/ and fill in the attached form following the instructions provided.
2. On receipt of an email from KX providing access to your license download this file and save to a secure location on your computer.
3. Set an environment variable on your computer pointing to the folder containing the license file (instructions for setting environment variables on PyKX supported operating systems can be found [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/).
	* Variable Name: `QLIC`
	* Variable Value: `/user/path/to/folder`

### Commercial Evaluation License

The following steps outline the process by which a user can gain access to an install a kdb Insights license which provides access to PyKX 

1. Visit https://kx.com/kdb-insights-commercial-evaluation-license-download/ and fill in the attached form following the instructions provided.
2. On receipt of an email from KX providing access to your license download this file and save to a secure location on your computer.
3. Set an environment variable on your computer pointing to the folder containing the license file (instructions for setting environment variables on PyKX supported operating systems can be found [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/). 
	* Variable Name: `QLIC`
	* Variable Value: `/user/path/to/folder`

!!! Note

	PyKX will not operate with a vanilla or legacy kdb+ license which does not have access to specific feature flags embedded within the license. In the absence of a license with appropriate feature flags PyKX will fail to initialise with full feature functionality.

## Supported Environments

KX only officially supports versions of PyKX built by KX, i.e. versions of PyKX installed from wheel files. Support for user-built installations of PyKX (e.g. built from the source distribution) is only provided on a best-effort basis. Currently, PyKX provides wheels for the following environments:

- Linux (`manylinux_2_17_x86_64`) with CPython 3.8-3.11
- macOS (`macosx_10_10_x86_64`) with CPython 3.8-3.11
- Windows (`win_amd64`) with CPython 3.8-3.11

## Dependencies

### Python Dependencies

PyKX depends on the following third-party Python packages:

- `pandas~=1.2`
- `numpy~=1.22`

They are installed automatically by `pip` when PyKX is installed.

PyKX also has an optional Python dependency of `pyarrow>=3.0.0`, which can be included by installing the `pyarrow` extra, e.g. `pip install pykx[pyarrow]`

!!! Warning

    Trying to use the `pa` conversion methods of `pykx.K` objects or the `pykx.toq.from_arrow` method when PyArrow is not installed (or could not be imported without error) will raise a `pykx.PyArrowUnavailable` exception.  `pyarrow` is supported Python 3.8-3.10 but remains in Beta for Python 3.11.

### Optional Non-Python Dependencies

- `libssl` for TLS on [IPC connections](../api/ipc.md).

### Windows Dependencies

To run q or PyKX on Windows, `msvcr100.dll` must be installed. It is included in the [Microsoft Visual C++ 2010 Redistributable](https://www.microsoft.com/en-ca/download/details.aspx?id=26999).

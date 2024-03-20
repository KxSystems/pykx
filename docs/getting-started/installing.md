# Installing

Installation of PyKX is available in using three methods

1. Installing PyKX from PyPI
2. Installation from source
3. Installation using Anaconda

??? Warning "Anaconda OS support"

	PyKX on Anaconda is only supported for Linux x86 and arm based architectures at this time

!!! Note Python Support

	PyKX is only officially supported on Python versions 3.8-3.12, Python 3.7 has reached end of life and is no longer actively supported, please consider upgrading

=== "Installing PyKX from PyPI"
	Ensure you have a recent version of `pip`:
	
	```
	pip install --upgrade pip
	```
	
	Then install the latest version of PyKX with the following command:

	```
	pip install pykx
	```

=== "Installing PyKX from source"
	Installing PyKX from source requires you to have access to a [github](https://github.com) account, once you have access to github you can clone the PyKX repository as follows

	```
	git clone https://github.com/kxsystems/pykx
	```

	Once cloned you can move into the cloned directory and install PyKX using `pip`

	```
	cd pykx
	pip install .
	```

=== "Installing PyKX from Anaconda"
	If you use `conda` you can install PyKX from the `kx` channel on Anaconda as follows typing `y` when prompted to accept the installation

	```
	conda install -c kx pykx
	```

!!! Warning

    Python packages should typically be installed in a virtual environment. [This can be done with the venv package from the standard library](https://docs.python.org/3/library/venv.html).

## PyKX License access and enablement

Installation of PyKX following the instructions above provides users with access to the library with limited functional scope, full details of these limitations can be found [here](../user-guide/advanced/modes.md). To access the full functionality of PyKX you must first download and install a KX license, this can be achieved either through use of a personal evaluation license or receipt of a commercial license.

!!! Warning "Legacy kdb+/q licenses do not support PyKX by default"

        PyKX will not operate with a vanilla or legacy kdb+ license which does not have access to specific feature flags embedded within the license. In the absence of a license with appropriate feature flags PyKX will fail to initialise with full feature functionality.

### License installation from a Python session

The following steps outline the process by which a user can gain access to and install a kdb Insights personal evaluation license for PyKX from a Python session.

??? Note "Commercial evaluation installation workflow"

	The same workflow used for the personal evaluations defined below can be used for commercial evaluations, the only difference being the link used when signing up for your evaluation license. In the case of commercial evaluation this should be https://kx.com/kdb-insights-commercial-evaluation-license-download/

1. Start your Python session

	```bash
	$ python
	```

2. Import the PyKX library which will prompt for user input accept this message using `Y` or hitting enter

	```python
	>>> import pykx as kx

	Thank you for installing PyKX!

	We have been unable to locate your license for PyKX. Running PyKX in unlicensed mode has reduced functionality.
	Would you like to continue with license installation? [Y/n]: 
	```

3. You will then be prompted asking if you would like to redirect to the kdb Insights personal license installation website

	```bash
	To apply for a PyKX license, please visit https://kx.com/kdb-insights-personal-edition-license-download.
	Once the license application has completed, you will receive a welcome email containing your license information.
	Would you like to open this page? [Y/n]:
	```

4. Ensure that you have completed the form for accessing a kdb Insights personal evaluation license and have received your welcome email.
5. Your will be prompted asking if you wish to install your license based on downloaded license file or using the base64 encoded string provided in your email as follows. Enter `1`, `2` or `3` as appropriate.

	```bash
	Please select the method you wish to use to activate your license:
	  [1] Download the license file provided in your welcome email and input the file path (Default)
	  [2] Input the activation key (base64 encoded string) provided in your welcome email
	  [3] Proceed with unlicensed mode:
	Enter your choice here [1/2/3]: 
	```

6. Once you have decided on decided on your option please finish your installation following the appropriate final step below

	=== "1"

		```bash
		Please provide the download location of your license (E.g., ~/path/to/kc.lic) : 
		```

	=== "2"

		```bash
		Please provide your activation key (base64 encoded string) provided with your welcome email : 
		```

7. Validate that your license has been installed correctly

	```python
	>>> kx.q.til(10)
	pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
	```

!!! Note "Troubleshooting and Support"

	If once you have completed these installation steps you are still seeing issues please visit our [troubleshooting](../troubleshooting.md) guide and [support](../support.md) pages.

### License installation using environment variables

To provide environment specific flexibility there are two methods by which users can install a license using environment variables. In both cases this method is flexible to the installation of both `kc.lic` and `k4.lic` versions of a license.

#### Using a supplied license file directly

1. Visit [here](https://kx.com/kdb-insights-personal-edition-license-download/) for a personal edition or [here](https://kx.com/kdb-insights-commercial-evaluation-license-download/) for a commercial evaluation license and fill in the attached form following the instructions provided.
2. On receipt of an email from KX providing access to your license download the license file and save to a secure location on your computer.
3. Set an environment variable on your computer pointing to the folder containing the license file (instructions for setting environment variables on PyKX supported operating systems can be found [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/).
       * Variable Name: `QLIC`
       * Variable Value: `/user/path/to/folder`

#### Using the base64 encoded license content

1. Visit [here](https://kx.com/kdb-insights-personal-edition-license-download/) for a personal edition or [here](https://kx.com/kdb-insights-commercial-evaluation-license-download/) for a commercial evaluation license and fill in the attached form following the instructions provided.
2. On receipt of an email from KX providing access to your license copy the base64 encoded contents of your license provided in plain-text within the email
3. Set an environment variable `KDB_LICENSE_B64` on your computer pointing with the value copied in step 2 (instructions for setting environment variables on PyKX supported operating systems can be found [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/).
       * Variable Name: `KDB_LICENSE_B64`
       * Variable Value: `<copied contents from email>`

If looking to make use of a `k4.lic` you can do so by setting the base64 encoded content of your file as the environment variable `KDB_K4LICENSE_B64`.

## Supported Environments

KX only officially supports versions of PyKX built by KX, i.e. versions of PyKX installed from wheel files. Support for user-built installations of PyKX (e.g. built from the source distribution) is only provided on a best-effort basis. Currently, PyKX provides wheels for the following environments:

- Linux (`manylinux_2_17_x86_64`, `linux-arm64`) with CPython 3.8-3.12
- macOS (`macosx_10_10_x86_64`, `macosx_10_10_arm`) with CPython 3.8-3.12
- Windows (`win_amd64`) with CPython 3.8-3.12

## Dependencies

### Python Dependencies

#### Required Python dependencies

PyKX depends on the following third-party Python packages:

- `numpy~=1.20, <2.0; python_version=='3.7'`
- `numpy~=1.22, <2.0; python_version<'3.11', python_version>'3.7'`
- `numpy~=1.23, <2.0; python_version=='3.11'`
- `numpy~=1.26, <2.0; python_version=='3.12'`
- `pandas>=1.2, < 2.2.0`
- `pytz>=2022.1`
- `toml~=0.10.2`

They are installed automatically by `pip` when PyKX is installed.

The following provides a breakdown of how these libraries are used within PyKX

- [Numpy](https://pypi.org/project/numpy) is used by PyKX when converting data from PyKX objects to numpy equivalent array/recarray style objects, additionally low level integration allowing direct calls to numpy functions such as `numpy.max` with PyKX objects relies on the numpy Python API.
- [Pandas](https://pypi.org/project/pandas) is used by PyKX when converting PyKX data to Pandas Series/DataFrame equivalent objects, additionally when converting data to PyArrow data formats as supported by the optional dependencies below Pandas is used as an intermendiary data format.
- [pytz](https://pypi.org/project/pytz/) is used by PyKX when converting data with timezone information to PyKX objects in order to ensure that the timezone offsets are accurately applied.
- [toml](https://pypi.org/project/toml/) is used by PyKX for configuration parsing, in particular when users make use of `.pykx-config` files for configuration management as outlined [here](../user-guide/configuration.md).


#### Optional Python Dependencies

- `pyarrow>=3.0.0`, which can be included by installing the `pyarrow` extra, e.g. `pip install pykx[pyarrow]`.
- `find-libpython~=0.2`, which can be included by installing the `debug` extra, e.g. `pip install pykx[debug]`.
- `ast2json~=0.3`, which is required for KX Dashboards Direct integration and can be installed with the `dashboards` extra, e.g. `pip install pykx[dashboards]`
- `dill>=0.2`, which is required for the Beta feature `Remote Functions` can be installed via pip with the `beta` extra, e.g. `pip install pykx[beta]`

!!! Warning

    Trying to use the `pa` conversion methods of `pykx.K` objects or the `pykx.toq.from_arrow` method when PyArrow is not installed (or could not be imported without error) will raise a `pykx.PyArrowUnavailable` exception.  `pyarrow` is supported Python 3.8-3.10 but remains in Beta for Python 3.11-3.12.

The following provides a breakdown of how these libraries are used within PyKX

- [PyArrow](https://pypi.org/project/pyarrow) is used by PyKX for the conversion of PyKX object to and from their PyArrow equivalent table/array objects. 
- [find-libpython](https://pypi.org/project/find-libpython) can be used by developers using PyKX to source the `libpython.{so|dll|dylib}` file required by [PyKX under q](../pykx-under-q/intro.md).

### Optional Non-Python Dependencies

- `libssl` for TLS on [IPC connections](../api/ipc.md).
- `libpthread` on Linux/MacOS when using the `PYKX_THREADING` environment variable.

### Windows Dependencies

To run q or PyKX on Windows, `msvcr100.dll` must be installed. It is included in the [Microsoft Visual C++ 2010 Redistributable](https://www.microsoft.com/en-ca/download/details.aspx?id=26999).

Alternatively installation of all required Windows dependencies can be completed through execution of the `w64_install.ps1` supplied at the root of the PyKX github [here](https://github.com/KxSystems/pykx) as follows using PowerShell:

```PowerShell
git clone https://github.com/kxsystems/pykx
cd pykx
.\w64_install.ps1
```

## Next steps

- [Quickstart guide](quickstart.md)
- [User guide introduction](../user-guide/index.md)

---
title:  PyKX installation guide
description: Getting started with PyKX
date: April 2024
author: KX Systems, Inc.,
tags: PyKX, setup, install,
---
# PyKX installation guide

_This section explains how to install PyKX on your machine._

## Pre-requisites

Before you start, make sure you have:

- **Python** (versions 3.8-3.12)
- **pip**

Recommended: a virtual environment with packages such as [venv](https://docs.python.org/3/library/venv.html) from the standard library.

## Supported environments

KX only supports versions of PyKX built by KX (installed from wheel files) for:

- **Linux** (`manylinux_2_17_x86_64`, `linux-arm64`) with CPython 3.8-3.12
- **macOS** (`macosx_10_10_x86_64`, `macosx_10_10_arm`) with CPython 3.8-3.12
- **Windows** (`win_amd64`) with CPython 3.8-3.12

??? Note "Special instructions for Windows users."

	To run q or PyKX on Windows, you have two options:

	- **Install** `#!bash msvcr100.dll`, included in the [Microsoft Visual C++ 2010 Redistributable](https://www.microsoft.com/en-ca/download/details.aspx?id=26999).

	- **Or Execute** `#!bash w64_install.ps1` supplied at the root of the PyKX GitHub [here](https://github.com/KxSystems/pykx) as follows, using PowerShell:

	```PowerShell
	git clone https://github.com/kxsystems/pykx
	cd pykx
	.\w64_install.ps1
	```
We provide assistance to user-built installations of PyKX only on a best-effort basis.

## 1. Install PyKX

You can install PyKX from three sources:

!!! Note ""

	=== "Install PyKX from PyPI"

		Ensure you have a recent version of `#!bash pip`:

		```
		pip install --upgrade pip
		```
		Then install the latest version of PyKX with the following command:

		```
		pip install pykx
		```

	=== "Install PyKX from Anaconda"
			
		For Linux x86 and arm-based architectures, you can install PyKX from the `#!bash kx` channel on Anaconda as follows:

		```
		conda install -c kx pykx
		```
		Type `#!bash y` when prompted to accept the installation.


	=== "Install PyKX from GitHub"
			
		Clone the PyKX repository:

		```
		git clone https://github.com/kxsystems/pykx
		```

		Enter the cloned repository and install PyKX using `#!bash pip`:

		```
		cd pykx
		pip install .
		```

At this point you have [partial access to PyKX](../user-guide/advanced/modes.md#operating-in-the-absence-of-a-kx-license). To gain access to all PyKX features, follow the steps in the next section, otherwise go straight to [3. Verify PyKX Installation](#3-verify-pykx-installation).  

## 2. Install a KDB Insights license

To use all PyKX functionalities, you need to download and install a KDB Insights license. 

!!! Warning "Legacy kdb+/q licenses do not support all PyKX features."

There are two types of KDB Insights licenses for PyKX: personal and commercial. For either of them, you have two installation options:

  - a) from Python
  - b) using environment variables

### 2.a Install license in Python

Follow the steps below to install a KDB Insights license for PyKX from Python:

1. Start your Python session:

	```bash
	$ python
	```

2. Import the PyKX library. When prompted to accept the installation, type `Y` or press `Enter`:

	```python
	>>> import pykx as kx

	Thank you for installing PyKX!

	We have been unable to locate your license for PyKX. Running PyKX in unlicensed mode has reduced functionality.
	Would you like to continue with license installation? [Y/n]: 
	```

3. Choose whether you wish to install a personal or commercial license, type `Y` or press `Enter` to choose a personal license

	```python
	Is the intended use of this software for:
	    [1] Personal use (Default)
	    [2] Commercial use
	Enter your choice here [1/2]:
	```

4. When asked if you would like to apply for a license, type `Y` or press `Enter`:

	=== "Personal license"

		```bash
		To apply for a PyKX license, navigate to https://kx.com/kdb-insights-personal-edition-license-download
		Shortly after you submit your license application, you will receive a welcome email containing your license information.
		Would you like to open this page? [Y/n]:
		```

	=== "Commercial license"

		```bash
		To apply for your PyKX license, contact your KX sales representative or sales@kx.com.
        Alternately apply through https://kx.com/book-demo.  
		Would you like to open this page? [Y/n]:
		```

5. For personal use, complete the form to receive your welcome email. For commercial use, the license will be provided over email after the commercial evaluation process has been followed with the support of your sales representative.

6. Choose the desired method to activate your license by typing `1`, `2`, or `3` as appropriate:

	```bash
	Select the method you wish to use to activate your license:
		[1] Download the license file provided in your welcome email and input the file path (Default)
		[2] Input the activation key (base64 encoded string) provided in your welcome email
		[3] Proceed with unlicensed mode
	Enter your choice here [1/2/3]: 
	```

7. Depending on your choice (`1`, `2`, or `3`), complete the installation by following the final step as below:

	=== "1"

		=== "Personal license"

			```bash
			Provide the download location of your license (for example,  ~/path/to/kc.lic): 
			```

		=== "Commercial license"

			```bash
			Provide the download location of your license (for example, ~/path/to/k4.lic):
			```

	=== "2"

		```bash
		Provide your activation key (base64 encoded string) provided with your welcome email: 
		```
	=== "3"

		```bash
		No further actions needed.
		```

8. Validate the correct installation of your license:

	```python
	>>> kx.q.til(10)
	pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
	```

### 2.b Install license with environment variables

For environment-specific flexibility, there are two ways to install your license: by using a file or by copying text. Both are sourced in your welcome email. Click on the tabs below, read the instructions, and choose the method you wish to follow:

!!! Note ""

	=== "Using a file"

		1. For personal usage, navigate to the [personal license](https://kx.com/kdb-insights-personal-edition-license-download/) and complete the form. For commercial usage, contact your KX sales representative or sales@kx.com or apply through https://kx.com/book-demo.

		2. On receipt of an email from KX, download and save the license file to a secure location on your computer.

		3. Set an environment variable pointing to the folder with the license file. (Learn how to set environment variables from [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/)).
       		* **Variable Name**: `#!bash QLIC`
      	    * **Variable Value**: `#!bash /user/path/to/folder`

	=== "Using text"

		1. For personal usage, navigate to the [personal license](https://kx.com/kdb-insights-personal-edition-license-download/) and complete the form. For commercial usage, contact your KX sales representative or sales@kx.com or apply through https://kx.com/book-demo.

		2. On receipt of an email from KX, copy the `#!bash base64` encoded contents of your license provided in plain-text within the email.

		3. On your computer, set an environment variable `#!bash KDB_LICENSE_B64` when using a personal license or `KDB_K4LICENSE_B64` for a commercial license, pointing with the value copied in step 2. (Learn how to set environment variables from [here](https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/)).
       		* **Variable Name**: `KDB_LICENSE_B64`  / `KDB_K4LICENSE_B64`
      	    * **Variable Value**: `<copied contents from email>`

To validate if you successfully installed your license with environment variables, start Python and import PyKX as follows:

```bash
$ python
>>> import pykx as kx
>>> kx.q.til(5)
pykx.LongVector(pykx.q('0 1 2 3 4'))
```

As you approach the expiry date for your license you can have PyKX automatically update your license by updating the environment variable `KDB_LICENSE_B64` or `KDB_K4LICENSE_B64` with your new license information. Once PyKX is initialised with your expired license it will attempt to overwrite your license with the newly supplied value. This is outlined as follows:

```python
$python
>>> import pykx as kx
Initialisation failed with error: exp
Your license has been updated using the following information:
  Environment variable: 'KDB_K4LICENSE_B64'
  License write location: /user/path/to/license/k4.lic
```

## 3. Verify PyKX installation

To verify if you successfully installed PyKX on your system, run:

```bash
python -c"import pykx;print(pykx.__version__)"
```

This command should display the installed version of PyKX.

## Dependencies

??? Info "Expand for Required and Optional PyKX dependencies"

	=== "Required"

		PyKX depends on the following third-party Python packages:

		- `numpy~=1.20, <2.0; python_version=='3.7'`
		- `numpy~=1.22, <2.0; python_version<'3.11', python_version>'3.7'`
		- `numpy~=1.23, <2.0; python_version=='3.11'`
		- `numpy~=1.26, <2.0; python_version=='3.12'`
		- `pandas>=1.2, < 2.2.0`
		- `pytz>=2022.1`
		- `toml~=0.10.2`

		**Note**: All are installed automatically by `#!bash pip` when you install PyKX.

		Here's a breakdown of how PyKX uses these libraries:

		- [NumPy](https://pypi.org/project/numpy): converts data from PyKX objects to NumPy equivalent Array/Recarray style objects; direct calls to NumPy functions such as `numpy.max` with PyKX objects relies on the NumPy Python API.
		- [Pandas](https://pypi.org/project/pandas): converts PyKX data to Pandas Series/DataFrame equivalent objects or to PyArrow data formats. Pandas is used as an intermendiary data format.
		- [pytz](https://pypi.org/project/pytz/): converts data with timezone information to PyKX objects to ensure that the offsets are accurately applied.
		- [toml](https://pypi.org/project/toml/): for configuration parsing and management, with `.pykx-config` as outlined [here](../user-guide/configuration.md).


	=== "Optional"

		**Optional Python dependencies:**

		- **`pyarrow >=3.0.0`**: install `pyarrow` extra, for example `pip install pykx[pyarrow]`.
		- **`find-libpython ~=0.2`**: install `debug` extra, for example `pip install pykx[debug]`.
		- **`ast2json ~=0.3`**: install with `dashboards` extra, for example `pip install pykx[dashboards]`
		- **`dill >=0.2`**: install via pip, with`beta` extra, for example `pip install pykx[beta]`

        Here's a breakdown of how PyKX uses these libraries: 

		- [PyArrow](https://pypi.org/project/pyarrow): converts PyKX objects to and from their PyArrow equivalent table/array objects. 
		- [find-libpython](https://pypi.org/project/find-libpython): provides the `libpython.{so|dll|dylib}` file required by [PyKX under q](../pykx-under-q/intro.md).
		- [ast2json](https://pypi.org/project/ast2json/): required for KX Dashboards Direct integration.
		- [dill](https://pypi.org/project/dill/): required for the Beta feature `Remote Functions`.

	    **Optional non-Python dependencies:**

		- `libssl` for TLS on [IPC connections](../api/ipc.md).
		- `libpthread` on Linux/MacOS when using the `PYKX_THREADING` environment variable.

!!! Note "Troubleshooting and Support"

	If you encounter any issues during the installation process, refer to the following sources for assistance:
	
	   - Visit our [troubleshooting](../troubleshooting.md) guide.
	   - Ask a question on the KX community at [learninghub.kx.com](https://learninghub.kx.com/forums/forum/pykx/).
       - Use Stack Overflow and tag [`pykx`](https://stackoverflow.com/questions/tagged/pykx) or [`kdb`](https://stackoverflow.com/questions/tagged/kdb) depending on the subject.
	   - Go to [support](../support.md).

## Next steps

That's it! You can now start using PyKX in your Python projects:

- [Quickstart guide](quickstart.md)
- [User guide introduction](../user-guide/index.md)

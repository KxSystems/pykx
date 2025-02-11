---
title:  PyKX installation guide
description: Getting started with PyKX
date: April 2024
author: KX Systems, Inc.,
tags: PyKX, setup, install,
---
# PyKX installation guide

_This page explains how to install PyKX on your machine._

!!! License

	PyKX is released under a dual license covering the files within the [PyKX repository](https://github.com/kxsystems/pykx) as outlined [here](../license.md).

	**Acceptance of license terms:**
	
	By downloading, installing, or using PyKX, you acknowledge and agree that you have read, understood, and accept the license [link](../license.md) and will adhere to its terms. 

## Pre-requisites

Before you start, make sure you have:

- [**Python**](https://www.python.org/downloads/) (versions 3.8-3.13)
- [**pip**](https://pypi.org/project/pip/)

Recommended: a virtual environment with packages such as [venv](https://docs.python.org/3/library/venv.html) from the standard library.

## Supported environments

KX only supports versions of PyKX built by KX (installed from wheel files) for:

- **Linux** (`manylinux_2_17_x86_64`, `linux-arm64`) with CPython 3.8-3.13
- **macOS** (`macosx_10_10_x86_64`, `macosx_10_10_arm`) with CPython 3.8-3.13
- **Windows** (`win_amd64`) with CPython 3.8-3.13

We provide assistance to user-built installations of PyKX only on a best-effort basis.

## 1. Install PyKX

You can install PyKX from three sources:

!!! Note "Installing in air-capped environments"

        If you are installing in a location without internet connection you may find [this section](#installing-in-an-air-gapped-environment) useful.

=== "Install PyKX from PyPI"

	Ensure you have a recent version of `#!bash pip`:

	```sh
	pip install --upgrade pip

	```
	Then install the latest version of PyKX with the following command:

	```sh
	pip install pykx

	```

=== "Install PyKX from Anaconda"

	For Linux x86 and arm-based architectures, you can install PyKX from the `#!bash kx` channel on Anaconda as follows:

	```sh
	conda install -c kx pykx

	```
	Type `#!bash y` when prompted to accept the installation.


=== "Install PyKX from GitHub"

	Clone the PyKX repository:

	```sh
	git clone https://github.com/kxsystems/pykx

	```

	Enter the cloned repository and install PyKX using `#!bash pip`:

	```sh
	cd pykx
	pip install .

	```

At this point you have [partial access to PyKX](../user-guide/advanced/modes.md#operating-in-the-absence-of-a-kx-license). To gain access to all PyKX features, follow the steps in the next section, otherwise go straight to [3. Verify PyKX Installation](#3-verify-pykx-installation).

## 2. Install a kdb Insights license

To use all PyKX functionalities, you need to download and install a kdb Insights license.

!!! Warning "Legacy kdb+/q licenses do not support all PyKX features."

There are two types of kdb Insights licenses for PyKX: personal and commercial. For either of them, you have two installation options:

  - a) from Python
  - b) using environment variables

### 2.a Install license in Python

Follow the steps below to install a kdb Insights license for PyKX from Python:

1. Start your Python session:

	```bash
	$ python
	```

2. Import the PyKX library. When prompted to accept the installation, type `#!python Y` or press `#!python Enter`:

	```python
	>>> import pykx as kx

	Thank you for installing PyKX!

	We have been unable to locate your license for PyKX. Running PyKX in unlicensed mode has reduced functionality.
	Would you like to install a license? [Y/n]:
	```

3. Indicate whether you have access to an existing PyKX enabled license or not, type `#!python N` or press `#!python Enter` to continue with accessing a new license:

	```python
	Do you have access to an existing license for PyKX that you would like to use? [N/y]:
	```

4. Choose whether you wish to install a personal or commercial license, type `#!python Y` or press `#!python Enter` to choose a personal license

	```python
	Is the intended use of this software for:
	    [1] Personal use (Default)
	    [2] Commercial use
	Enter your choice here [1/2]:
	```

5. When asked if you would like to apply for a license, type `#!python Y` or press `#!python Enter`:

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

6. For personal use, complete the form to receive your welcome email. For commercial use, the license will be provided over email after the commercial evaluation process has been followed with the support of your sales representative.

7. Choose the desired method to activate your license by typing `#!python 1`, `#!python 2`, or `#!python 3` as appropriate:

	```bash
	Select the method you wish to use to activate your license:
		[1] Download the license file provided in your welcome email and input the file path (Default)
		[2] Input the activation key (base64 encoded string) provided in your welcome email
		[3] Proceed with unlicensed mode
	Enter your choice here [1/2/3]:
	```

8. Depending on your choice (`#!python 1`, `#!python 2`, or `#!python 3`), complete the installation by following the final step as below:

	=== "1"

		=== "Personal license"

			```bash
			Provide the download location of your license (for example, ~/path/to/kc.lic): 
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

9. Validate the correct installation of your license:

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

!!! Tip "Tip: automatic license renewal setup"

    When your license nears its expiry date, you can set PyKX to automatically renew it. To do this, modify the environment variable `#!bash KDB_LICENSE_B64` or `#!bash KDB_K4LICENSE_B64` with your new license information. When PyKX initializes with the expired license, it will attempt to overwrite it with the new value:

	```shell
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

      - `pandas>=1.2, <2.0; python_version=='3.8'`
      - `pandas>=1.2, <=2.2.3; python_version>'3.8'`
      - `numpy~=1.22; python_version<'3.11'`
      - `numpy~=1.23; python_version=='3.11'`
      - `numpy~=1.26; python_version>='3.12'`
      - `pytz>=2022.1`
      - `toml~=0.10.2`
      - `dill>=0.2.0`
      - `requests>=2.25.0`

		**Note**: All are installed automatically by `#!bash pip` when you install PyKX.

		Here's a breakdown of how PyKX uses these libraries:

		- [NumPy](https://pypi.org/project/numpy): converts data from PyKX objects to NumPy equivalent Array/Recarray style objects; direct calls to NumPy functions such as `numpy.max` with PyKX objects relies on the NumPy Python API.
		- [Pandas](https://pypi.org/project/pandas): converts PyKX data to Pandas Series/DataFrame equivalent objects or to PyArrow data formats. Pandas is used as an intermediary data format.
		- [pytz](https://pypi.org/project/pytz/): converts data with timezone information to PyKX objects to ensure that the offsets are accurately applied.
		- [toml](https://pypi.org/project/toml/): for configuration parsing and management, with `.pykx-config` as outlined [here](../user-guide/configuration.md).
		- [dill](https://pypi.org/project/dill): use in the serialization and deserialization of Python objects when interfacing between kdb+ and Python processes using [remote functions](../user-guide/advanced/remote-functions.md) or [real-time capture](../user-guide/advanced/streaming/index.md) functionality.


	=== "Optional"

		**Optional Python dependencies:**

		- **`pyarrow >=3.0.0, <19.0.0`**: install `pyarrow` extra, for example `pip install pykx[pyarrow]`.
		- **`find-libpython ~=0.2`**: install `debug` extra, for example `pip install pykx[debug]`.
		- **`ast2json ~=0.3`**: install with `dashboards` extra, for example `pip install pykx[dashboards]`
		- **`dill >=0.2`**: install via pip, with `remote` extra, for example `pip install pykx[remote]`
		- **`beautifulsoup4 >=4.10.0`**: install with `help` extra, for example `pip install pykx[help]`
		- **`markdown2 >=2.5.0`**: install with `help` extra, for example `pip install pykx[help]`
		- **`psutil >=5.0.0`**: install via pip, with `streaming` extra, for example `pip install pykx[streaming]`
		- **`torch >2.1`**: install via pip, with `torch` extra, for example `pip install pykx[torch]`

        Here's a breakdown of how PyKX uses these libraries:

		- [PyArrow](https://pypi.org/project/pyarrow): converts PyKX objects to and from their PyArrow equivalent table/array objects.
		- [find-libpython](https://pypi.org/project/find-libpython): provides the `libpython.{so|dll|dylib}` file required by [PyKX under q](../pykx-under-q/intro.md).
		- [ast2json](https://pypi.org/project/ast2json/): required for KX Dashboards Direct integration.
		- [psutil](https://pypi.org/project/psutil/): facilitates the stopping and killing of a q process on a specified port allowing for orphaned q processes to be stopped, functionality defined [here](../api/util.md#pykxutilkill_q_process).
		- [torch](https://pytorch.org/docs/stable/): required for conversions between `#!python torch.Tensor` objects and their PyKX equivalents.

	    **Optional non-Python dependencies:**

		- `libssl` for TLS on [IPC connections](../api/ipc.md).
		- `libpthread` on Linux/MacOS when using the `PYKX_THREADING` environment variable.

!!! Note "Troubleshooting and Support"

	If you encounter any issues during the installation process, refer to the following sources for assistance:
	
	   - Visit our [troubleshooting](../help/troubleshooting.md) guide.
	   - Ask a question on the KX community at [learninghub.kx.com](https://learninghub.kx.com/forums/forum/pykx/).
       - Use Stack Overflow and tag [`pykx`](https://stackoverflow.com/questions/tagged/pykx) or [`kdb`](https://stackoverflow.com/questions/tagged/kdb) depending on the subject.
	   - Go to [support](../help/support.md).

## Optional: Installing a q executable

The following section is optional and primarily required if you are looking to make use of the [Real-Time Capture](../user-guide/advanced/streaming/index.md) functionality provided by PyKX.

### Do I need a q executable?

For the majority of functionality provided by PyKX you do not explicitly need access to a q executable. Users within a Python process who do not have a q executable will be able to complete tasks such as the following:

- Convert data to/from Python types
- Run analytics on in-memory and on-disk databases
- Create databases
- Query remote q/kdb+ processes via IPC
- Execute numpy functions with PyKX data

If however you need to make use of the [Real-Time Capture](../user-guide/advanced/streaming/index.md) functionality you will need access to a q executable. Fundamentally the capture and persistence of real-time data and the application of analytics on this streaming data is supported via deployment of code on q processes.

### Configuring PyKX to use an existing executable

By default when attempting to start a q process for use within the Real-Time Capture workflows PyKX will attempt to call `q` directly, this method however is not fully reliable when using the Python `subprocess` module. As such the following setup can be completed to point more explicitly at your executable.

If you already have a q executable, PyKX can use this when initializing the Real-Time Capture APIs through the setting of the following in your [configuration file](../user-guide/configuration.md#configuration-file) or as [environment variables](../user-guide/configuration.md#environment-variables):

| **Variable**        | **Explanation**                                                                                                      |
| :------------------ | :--------------------------------------------------------------------------------------------------------------- |
| `PYKX_Q_EXECUTABLE` | Specifies the location of the q executable which should be called. Typically this will be `QHOME/[lmw]64/q[.exe]`|
| `QHOME`             | The directory to which q was installed                                                                           |

### Installing an executable

#### Installing using PyKX

For users who do not have access to a q executable, PyKX provides a utility function `kx.util.install_q` to allow users access q locally.

The following default information is used when installing the q executable:

| **Parameter**    | **Default**         | **Explanation**                                                                                                        |
| :--------------- | :------------------ | :--------------------------------------------------------------------------------------------------------------------- |
| location         | `'~/q'` or `'C:\q'` | The location to which q will be installed if not otherwise specified.                                                  |
| date             | `'2024.07.08'`      | The dated version of kdb+ 4.0 which is to be installed.                                                                |

The following provide a number of examples of using the installation functionality under various conditions.

- Installing to default location

	```python
	>>> kx.util.install_q()
	```

- Installing to a specified location

	```python
	>>> kx.util.install_q('~/custom')
	```

Installation of q via this method will update the configuration file `.pykx-config` at either `~/q` or `C:\q` to include the location of `QHOME` and `PYKX_Q_EXECUTABLE` to be used.

#### Installing without PyKX

The installed q executable is not required to be installed via PyKX. If you wish to install q following the traditional approach you can follow the install instructions outlined [here](https://code.kx.com/q/learn/install/) or through signing up for a free-trial [here](https://kx.com/download-kdb/).

### Installing in an air-gapped environment

Installing Python libraries in air-gapped environments requires users to first download the [Python wheel](https://realpython.com/python-wheels/) files for the libraries you need to install.

!!! Note "Build using the same environment as you're installing"

	When downloading the `.whl` files and dependencies make sure you are using the same OS and Python version as you will be when installing in your isolated environment 

In the case of PyKX users can in a internet enabled environment either

1. Download the `.whl` file for the OS, library version and Python version you are intending to use on the air-gapped environment. These files can be sourced from [here](https://pypi.org/project/pykx/#files).
2. Generate the `.whl` file from a git clone of the [PyKX repository](https://github.com/kxsystems/pykx). An example of this is as follows:

	```bash
	$ git clone https://github.com/kxsystems/pykx
	$ cd pykx
	$ pip install build
	# The below will install the `.whl` to a `dist/` folder
	$ python -m build .
	```
Once locally downloaded the dependencies of the `*.whl` file can be downloaded as follows:

	```bash
	$ pip download dist/*.whl
	```

Copy the content of your `dist/` folder to an external storage device (USB-key etc.) and upload the `.whl` files to your air-gapped device.

Install the wheels which for simplicity are stored at a location `/opt/airgap/wheels`

```bash
pip install --no-cache /opt/airgap/wheels/*
```

### Verify PyKX can use the executable

Verifying that PyKX has access to the executable can be done through execution of the function `#!python kx.util.start_q_subprocess` and requires either your configuration file or environment variables to include `PYKX_Q_EXECUTABLE`. This is outlined [here](#configuring-pykx-to-use-an-existing-executable).

```python
>>> import pykx as kx
>>> server = kx.util.start_q_subprocess(5052)
>>> conn = kx.SyncQConnection(port=5052)     # Connect to subprocess
>>> conn('1+1')
pykx.LongAtom(pykx.q('2'))
>>> server.kill()
```

## Next steps

That's it! You can now start using PyKX in your Python projects:

- [Quickstart guide](quickstart.md)
- [Updating/Upgrading your license](../user-guide/advanced/license.md)

---
title: Install KDB-X Python
description: Install KDB-X Python, configure a KDB-X license, verify the installation, and prepare air-gapped environments
last_updated: July 2026
author: KX Systems, Inc.,
keywords:
  - KDB-X Python
  - pykx
  - install
  - pip
  - license
  - air-gapped environments
  - Windows
---

# Install KDB-X Python

_Install KDB-X Python, configure a license, and verify the installation._

**Estimated time:** 5–10 minutes for a standard `pip` installation, license setup, and verification.

!!! warning "Software license terms"

	KX releases KDB-X Python under a dual license that covers the files in the [KDB-X Python repository](https://github.com/kxsystems/pykx). Review the [KDB-X Python license terms](../license.md) before installation.

	**Acceptance of license terms:**
	
	By downloading, installing, or using KDB-X Python, you acknowledge and agree that you have read, understood, and accept the [KDB-X Python license terms](../license.md).

## Prerequisites

Before you start, make sure you have:

- [**Python**](https://www.python.org/downloads/) 3.9-3.14
- [**pip**](https://pypi.org/project/pip/)

Create and activate a virtual environment to isolate KDB-X Python from other projects:

=== "macOS and Linux"

	```sh
	python3 -m venv .venv
	source .venv/bin/activate
	```

=== "Windows PowerShell"

	```powershell
	py -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```

	??? tip "If PowerShell blocks virtual environment activation"

		PowerShell may require permission to run the activation script. Enable locally created scripts for the current PowerShell session, then activate the environment:

		```powershell
		Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
		.\.venv\Scripts\Activate.ps1
		```

		When prompted, confirm the policy change. The setting applies only to the current PowerShell session. PowerShell discards it when you close the window.

		If an organizational policy prevents this change, run subsequent commands with `.\.venv\Scripts\python.exe` instead of `python`.

## Supported environments

KX supports KDB-X Python wheels for CPython 3.9-3.14 on Linux, macOS, and Windows.

??? info "Supported wheel platform tags"

	The wheel platform tag depends on the operating system and, for Linux, the CPython version:

	| **Platform** | **CPython versions** | **Wheel platform tags** |
	| --- | --- | --- |
	| Linux | 3.9-3.11 | `manylinux2014_x86_64`, `manylinux2014_aarch64` |
	| Linux | 3.12-3.14 | `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64` |
	| macOS | 3.9-3.14 | `macosx_10_15_x86_64`, `macosx_10_15_arm64` |
	| Windows | 3.9-3.14 | `win_amd64` |

## Install from PyPI

PyPI is the recommended installation method for most users. If you manage packages with Conda, uv, or pipx, or need to build from source, expand **Other installation methods**. For an offline system, follow [Air-gapped installation](#air-gapped-installation).

!!! info "Why the package is named `pykx`"

	KDB-X Python is the product name, but the Python distribution and import package remain `pykx` for compatibility with applications written for PyKX 3.x. Use `pykx` in package-manager commands and `import pykx` in Python code.

	If you are upgrading from PyKX 3.x, review the [migration guide](../upgrades/3040.md).

Upgrade `pip`:

```sh
python -m pip install --upgrade pip
```

Install the latest KDB-X Python release from PyPI:

```sh
python -m pip install --upgrade pykx
```

??? info "Other installation methods"

	=== "Conda"

		On Linux x86 and ARM architectures, create an environment and install KDB-X Python from the `kx` channel:

		```sh
		conda create --name kdbx-python python
		conda activate kdbx-python
		conda install -c kx pykx
		```

	=== "uv"

		Create a virtual environment and install KDB-X Python:

		```sh
		uv venv
		uv pip install pykx
		```

	=== "pipx"

		Install KDB-X Python and its dependencies in an isolated virtual environment:

		```sh
		pipx install pykx --include-deps
		```

		Activate the environment that pipx creates before you import KDB-X Python.

	=== "Build from source"

		Install KDB-X Python directly from the repository:

		```sh
		git clone https://github.com/kxsystems/pykx
		cd pykx
		python -m pip install .
		```

		Building from source requires Git and may require platform-specific build tools. KX provides best-effort support for user-built installations.

Without a KDB-X license, KDB-X Python runs with reduced functionality. To continue without a license, skip to [Verify the installation](#verify-the-installation).

## Install a KDB-X license

Before you start, make sure you have one of these KDB-X licenses ready:

- A `kc.lic` license key from the [KX Developer Center](https://developer.kx.com/products/kdb-x/install).
- A `k4.lic` license provided separately by KX.

[Review the KDB-X license requirements](https://code.kx.com/kdb-x/get_started/kdb-x-install.html#license-requirements) for more information.

=== "Python prompt"

	Start an interactive Python session and import KDB-X Python:

	```sh
	$ python
	```

	```python
	>>> import pykx as kx
	```

	When prompted, confirm that you want to install a license. Choose a license file or base64-encoded license key, then provide its location or value.

=== "Environment variables"

	Configure one license source before starting Python:

	| **License source** | **Environment variable** | **Value** |
	| --- | --- | --- |
	| `kc.lic` or `k4.lic` file | `QLIC` | Directory that contains the license file |
	| Base64-encoded `kc.lic` key | `KDB_LICENSE_B64` | License key supplied by KX |
	| Base64-encoded `k4.lic` key | `KDB_K4LICENSE_B64` | License key supplied by KX |

	KDB-X Python reads the configured license when the Python process starts.

## Verify the installation

Print the installed version and active license mode:

```sh
python -c "import pykx as kx; print(kx.__version__); print(f'Licensed: {kx.licensed}')"
```

The command prints the KDB-X Python version and one of these results:

- `Licensed: True` confirms that KDB-X Python initialized embedded q with the installed license.
- `Licensed: False` confirms that KDB-X Python runs in unlicensed mode.

If the result does not match your configuration, refer to [Troubleshooting](../help/troubleshooting.md).

## Air-gapped installation

Use a connected system that matches the operating system, architecture, and Python version of the air-gapped system.

Prepare a wheelhouse using one of these sources:

=== "PyPI"

	Download KDB-X Python and its dependencies:

	```sh
	$ python -m pip download --destination-directory wheelhouse pykx
	```

=== "Build from source"

	Clone the repository and build KDB-X Python and its dependencies:

	```sh
	$ git clone https://github.com/kxsystems/pykx
	$ cd pykx
	$ python -m pip wheel --wheel-dir wheelhouse .
	```

Copy the `wheelhouse` directory to the air-gapped system, then install from it. This example uses `/opt/airgap/wheels`:

```sh
python -m pip install --no-index --find-links=/opt/airgap/wheels pykx
```

After installation, [install a KDB-X license](#install-a-kdb-x-license) and [verify the installation](#verify-the-installation).

## Related installation topics

- [Configure KDB-X Python](../user-guide/configuration.md)
- [Review dependencies and bundled assets](../user-guide/configuration.md#dependencies-and-bundled-assets)
- [Manage or renew a license](../user-guide/advanced/license.md)
- [Get installation help](../help/support.md)

## Next steps

- [Run the quickstart](quickstart.md)

# FAQs

## Known Issues

* [PyKX known issues](extras/known_issues.md)
* [PyKX under q known issues](pykx-under-q/known_issues.md)

## How to work around the `'cores` licensing error?

```
>>> import pykx as kx
<frozen importlib._bootstrap>:228: PyKXWarning: Failed to initialize embedded q; falling back to unlicensed mode, which has limited functionality. Refer to https://code.kx.com/pykx/user-guide/advanced/modes.html for more information. Captured output from initialization attempt:
    '2022.09.15T10:32:13.419 license error: cores
```

This error indicates your license is limited to a given number of cores but PyKX tried to use more cores than the license allows.

- On Linux you can use `taskset` to limit the number of cores used by the python process and likewise PyKX and EmbeddedQ:

	```bash
	# Example to limit python to the 4 first cores on a 8 cores CPU
	$ taskset -c 0-3 python
	```

- You can also do this in python before importing PyKX (Linux only):

	```bash
	>>> import os
	>>> os.sched_setaffinity(0, [0, 1, 2, 3])
	>>> import pykx as kx
	>>> kx.q('til 10')
	pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
	```

- On Windows you can use the `start` command with its `/affinity` argument (see: `> help start`):

	```bat
	> start /affinity f python
	```

	(above, `0xf = 00001111b`, so the python process will only use the four cores for which the mask bits are equal to 1)

## How does PyKX determine the license that is used?

The following outlines the paths searched for when loading PyKX

1. Search for `kx.lic`, `kc.lic` and `k4.lic` license files in this order within the following locations
	1. Current working directory
	1. Location defined by environment variable `QLIC` if set
	1. Location defined by environment variable `QHOME` if set
2. If a license is not found use the following environment variables (if they are set) to install and make use of a license
	1. `KDB_LICENSE_B64` pointing to a base64 encoded version of a `kc.lic` license
	1. `KDB_K4LICENSE_B64` pointing to a base64 encoded version of a `k4.lic` license
3. If a license has not been located according to the above search you will be guided to install a license following a prompt based license installation walkthrough.

## Can I use PyKX in a subprocess?

Yes, however doing so requires some considerations. To ensure that PyKX is initialized in a clean environment it is suggested that the creation of subprocesses reliant on PyKX should be done within a code block making use of the `kx.PyKXReimport` functionality as follows:

```python
import pykx as kx
import subprocess
with kx.PyKXReimport():
    subprocess.Popen(['python', 'file.py']) # Run Python with a file that imports PyKX
```

Failure to use this functionality can result in segmentation faults as noted in the troubleshooting guide [here](troubleshooting.md). For more information on the `PyKXReimport` functionality see its API documentation [here](api/reimporting.md).

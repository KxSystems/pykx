---
title: FAQ
description: Frequently questions for PyKX
maintained by: KX Systems, Inc.
date: Aug 2024
tags: PyKX, FAQ
---
# FAQ

## How do I prevent the `#!python 'cores` licensing error when I run `#!python import pykx`?

```python
>>> import pykx as kx
<frozen importlib._bootstrap>:228: PyKXWarning: Failed to initialize embedded q; falling back to unlicensed mode, which has limited functionality. Refer to https://code.kx.com/pykx/user-guide/advanced/modes.html for more information. Captured output from initialization attempt:
    '2022.09.15T10:32:13.419 license error: cores
```

This error indicates PyKX tried to use more cores than your license allows. You can fix this by limiting the number of cores used by the python process.

- On Linux you can use `#!bash taskset` to limit the number of cores used by a process:

```bash
# Example to limit python to the 4 first cores on a 8 cores CPU
$ taskset -c 0-3 python
```

- You can also do this in python before importing PyKX (Linux only):

```python
>>> import os
>>> os.sched_setaffinity(0, [0, 1, 2, 3])
>>> import pykx as kx
>>> kx.q('til 10')
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
```

- On Windows you can use the `#!bat start` command with its `#!bat /affinity` argument (see: `#!bat > help start`):

```bat
> start /affinity f python
```

(above, `#!bat 0xf = 00001111b`, so the python process will only use the four cores for which the mask bits are equal to 1)

## How does PyKX determine the license that is used?

The following steps are run by PyKX to find the license when you execute `#!python import pykx`:

1. Search for **kx.lic**, **kc.lic** and **k4.lic** license files in this order within the following locations:
	1. Current working directory
	1. Location defined by environment variable `#!bash QLIC` if set
	1. Location defined by environment variable `#!bash QHOME` if set
2. If a license is not found PyKX will use the following environment variables (if they are set) to install and make use of a license:
	1. `#!bash KDB_LICENSE_B64` containing a base64 encoded version of a **kc.lic** license
	1. `#!bash KDB_K4LICENSE_B64` containing a base64 encoded version of a **k4.lic** license
3. If a license has not been located you will be guided to install a license following a prompt based license installation.

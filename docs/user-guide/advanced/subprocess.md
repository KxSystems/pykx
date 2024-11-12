---
title: PyKX in Subprocesses
description: Outline using pykx in python subprocesses
date: August 2024
author: KX Systems, Inc.
tags: PyKX
---

# Using PyKX in python subprocesses

_This page outlines using PyKX in a Python subprocess._

To use PyKX in a python subprocess you should spawn the process using the `#!python kx.PyKXReimport` function as follows:

```python
import pykx as kx
import subprocess
with kx.PyKXReimport():
    subprocess.Popen(['python', 'file.py']) #_Run a python subprocess that loads a python script containing a PyKX import
```

Failing to reimport the PyKX package running in the parent process can cause the subprocess to crash with a segmentation fault. The `#!python PyKXReimport` function and possible causes of segmentation faults is covered in more detail [here](../../api/reimporting.md).

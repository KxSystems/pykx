---
title: KDB-X Python in Subprocesses
description: Outline using pykx in python subprocesses
date: August 2024
author: KX Systems, Inc.
tags: KDB-X Python
---

# Using KDB-X Python in python subprocesses

_This page outlines using KDB-X Python in a Python subprocess._

To use KDB-X Python in a python subprocess you should spawn the process using the `#!python kx.PyKXReimport` function as follows:

```python
import pykx as kx
import subprocess
with kx.PyKXReimport():
    subprocess.Popen(['python', 'file.py']) #_Run a python subprocess that loads a python script containing a import pykx
```

Failing to reimport the KDB-X Python package running in the parent process can cause the subprocess to crash with a segmentation fault. The `#!python PyKXReimport` function and possible causes of segmentation faults is covered in more detail [here](../../api/reimporting.md).

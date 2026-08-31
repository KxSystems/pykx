---
title: Reimporting module
description: API reference page for reimporting the KDB-X Python module in a Python session
author: KX Systems
date: September 2024
tags: reimport, pykx, import
---
# Reimporting

!!! note "Usually not required"

    KDB-X Python now automatically manages the environment variables needed to reimport it, so in
    most cases you can start a subprocess that imports KDB-X Python (or, for a KDB-X 5.0 `q`
    process, loads `pykx.q`) directly, without `PyKXReimport`.

    `PyKXReimport` is **still required** when starting a **kdb+ 4.0/4.1** `q` subprocess from a
    Python process that has imported KDB-X Python. Alternatively, set `PYKX_IGNORE_QHOME=true`
    so `QHOME` is not redirected in the first place. 
    Refer to [QHOME, symlinking and subprocesses](../user-guide/configuration.md#qhome-symlinking-and-subprocesses) for details.

::: pykx.reimporter

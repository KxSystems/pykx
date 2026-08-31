---
title: KDB-X Python documentation
description: Find KDB-X Python installation and migration guidance, Quickstart, learning resources, how-to guides, integrations, examples, and API reference
last_updated: July 2026
author: KX Systems, Inc.,
keywords:
  - KDB-X Python
  - pykx
  - Python
  - q
  - KDB-X
  - documentation
---

# KDB-X Python

_Documentation for installing KDB-X Python and using q data types, queries, Python conversions, and IPC._

[KDB-X Python](./getting-started/what-is-kdb-x-python.md) is the official Python interface to KDB-X. Python code can create and query q objects, call q functions in the embedded runtime, and connect to separate q processes over IPC.

!!! info "Migration from PyKX"

    **KDB-X Python** (`pykx>=4.0`) is the evolution of **PyKX** (`pykx<4.0`). If you are upgrading, [read the migration guide](upgrades/3040.md) to understand what changed and how to update your code and configuration.

## Start here

New users should complete the **Get Started** steps in order. Otherwise, choose the relevant topic area.

<div class="large-tile-grid home-offerings">
    <div>
        <h3>Get Started</h3>
        <ul>
            <li><a href="getting-started/what-is-kdb-x-python.html">About</a></li>
            <li><a href="getting-started/installing.html">Install</a></li>
            <li><a href="getting-started/quickstart.html">Quickstart</a></li>
        </ul>
    </div>
    <div>
        <h3>Learn</h3>
        <ul>
            <li><a href="examples/interface-overview.html">Fundamentals</a></li>
            <li><a href="learn/objects.html">Objects and Attributes</a></li>
            <li><a href="https://academy.kx.com/courses/pykx-developer-level-1/">KX Academy Course</a></li>
        </ul>
    </div>
    <div>
        <h3>How-to Guides</h3>
        <ul>
            <li><a href="user-guide/configuration.html">Configure KDB-X Python</a></li>
            <li><a href="user-guide/fundamentals/creating.html">Interact with Data</a></li>
            <li><a href="user-guide/advanced/database/db_gen.html">Create Databases</a></li>
        </ul>
    </div>
    <div>
        <h3>Reference</h3>
        <ul>
            <li><a href="api/pykx-execution/q.html">API Reference</a></li>
            <li><a href="api/pykx-q-data/toq.html">Data Types and Conversions</a></li>
            <li><a href="user-guide/advanced/Pandas_API.html">pandas API</a></li>
        </ul>
    </div>
    <div>
        <h3>Integrations</h3>
        <ul>
            <li><a href="user-guide/advanced/numpy.html">NumPy</a></li>
            <li><a href="user-guide/advanced/streamlit.html">Streamlit</a></li>
            <li><a href="examples/charting.html">Python Charting Libraries</a></li>
        </ul>
    </div>
    <div>
        <h3>Examples</h3>
        <ul>
            <li><a href="examples/subscriber/readme.html">Subscribe to a q Process</a></li>
            <li><a href="examples/compress_and_encrypt/readme.html">Compress and Encrypt</a></li>
            <li><a href="examples/server/server.html">KDB-X Python as a Server</a></li>
        </ul>
    </div>
</div>

## Stay informed and get help

- Request a feature in the [KX Community Ideas Forum](https://forum.kx.com/c/ideas-feature-requests/10).
- Review the latest [KDB-X Python Release Notes](release-notes/changelog.md).
- Ask for help in the [KX Community Slack](https://kx-community.slack.com/).

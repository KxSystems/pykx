---
title: Jupyter magic commands
description: API reference for the q magic commands and q first mode in Jupyter notebooks
author: KX Systems
date: October 2026
tags: jupyter, notebook, magic, q
---
# Jupyter magic commands

_This page documents the magic commands and options available when running `pykx` in a Jupyter notebook. The [Jupyter notebooks example](../examples/jupyter-integration.ipynb) shows them in use._

`pykx` registers its magic commands automatically when it is imported in a Jupyter notebook or IPython session.

## `%%q`

```
%%q [options]
<q code>
```

Run the code in the cell as q. The cell runs on the embedded q process unless `--port` is given, in which case it runs on the q process listening on that port. The result of each expression in the cell is printed.

A `\d` in a cell changes the namespace for every cell that runs after it. End the cell with `\d .` to return to the root namespace.

**Options:**

Options which take a value must be followed by that value. The others are flags.

| Option | Short | Type | Description | Default |
| :----- | :---: | :--: | :---------- | :-----: |
| `--debug` | | | Print the q backtrace before raising a `QError`. | |
| `--display` | | | Call `display` on the result rather than `print`. | |
| `--save` | | `str` | Save the code in the cell to this file. The path must end in `.q`, or `.q_` to save a locked file. | |
| `--execute` | | `True` or `False` | Run the code in the cell. With `--save`, `False` saves the code without running it. | `True` |
| `--host` | `-h` | `str` | The host of the q process to connect to. | `localhost` |
| `--port` | `-p` | `int` | The port to connect on. Setting this runs the cell on that q process. | |
| `--user` | `-u` | `str` | The username to connect with. | |
| `--pass` | | `str` | The password to connect with. | |
| `--timeout` | `-t` | `float` | The time in seconds before a query times out. | no timeout |
| `--nolarge` | | | Disable sending and receiving messages over 2 GB. | |
| `--tls` | | | Connect using TLS. | |
| `--unix` | | | Connect using a Unix domain socket. | |
| `--noctx` | | | Disable the context interface for the connection. | |
| `--reconnection_attempts` | | `int` | The number of times to try to reconnect. | `-1` |

With `--save` on an IPC connection the file is written on the q process, not locally.

**Examples:**

```python
%%q
til 10
```

```python
%%q --display
([] a: 1 2 3)
```

```python
%%q --host localhost --port 5000 --user user --pass password --noctx
til 10
```

```python
%%q --save new_file.q
vals:til 10
vals * 3
```

## q first mode

In q first mode every cell runs as q without `%%q`. Cells which run Python must begin with `%%py`.

Enable the mode by setting the environment variable `PYKX_JUPYTERQ` to `true` before importing `pykx`, or at runtime with `pykx.util.jupyter_qfirst_enable()`. Return to Python first execution with `pykx.util.jupyter_qfirst_disable()`. From q, `.pykx.enableJupyter()` enables the mode.

```python
>>> import pykx as kx
>>> kx.util.jupyter_qfirst_enable()
```

```python
%%py
for fruit in ['apple', 'orange', 'banana']:
    print(fruit)
```

```python
%%py
kx.util.jupyter_qfirst_disable()
```

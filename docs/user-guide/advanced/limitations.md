# Library limitations

When q is run embedded within a Python process (as opposed to over IPC), it is restricted in how it can operate. This is a result of the fact that when running embedded it does not have the main loop or timers that one would expect from a typical q process. The following are a number of examples showing these limitations in action

## IPC Interface 

As a result of the lack of a main loop PyKX cannot be used to respond to q IPC requests as a server. Callback functions such as [`.z.pg`](https://code.kx.com/q/ref/dotz/#zpg-get) defined within a Python process will not operate as expected.

In a Python process, start a q IPC server:

```python
>>> import pykx as kx
>>> kx.q('\\p 5001')
pykx.Identity(pykx.q('::'))
>>>
```

Then in a Python or q process, attempt to connect to it:

```python
>>> import pykx as kx
>>> q = kx.QConnection(port=5001) # Attempt to create a q connection to a pykx embedded q instance
# Will hang indefinitely since the embedded q process cannot respond to IPC requests
```

```q
// Attempting to create an IPC connection to a PyKX embedded q instance
// will hang indefinitely since the embedded q process cannot respond to IPC requests
q)h: hopen `::5001
```

!!! danger "Do not use PyKX as an IPC server"

    Attempting to connect to a Python process running PyKX over IPC from another process will hang indefinitely.

## Timers

Timers in q rely on the use of the q main loop, as such these do not work within PyKX. For example:

```python
>>> import pykx as kx

>>> kx.q('.z.ts:{0N!x}') # Set callback function which should be run on a timer
>>> kx.q('\t 1000') # Set timer to tick every 1000ms
pykx.Identity(pykx.q('::')) # No output follows because the timer doesn't actually tick when within
                            # a Python process
```

Attempting to use the timer callback function directly using PyKX will raise an `AttributeError` as follows

```python
>>> kx.q.z.ts
AttributeError: ts: .z.ts is not exposed through the context interface because the main loop is inactive in PyKX.
```

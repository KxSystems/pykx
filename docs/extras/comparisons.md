## How does PyKX compare to other q interfaces for Python?

There are three historical interfaces which allow interoperability between Python and q/kdb+.

1. [Embedpy](https://code.kx.com/q/ml/embedpy)
2. [PyQ](https://github.com/KxSystems/pyq)
3. [qPython](https://github.com/KxSystems/pyq)

An understanding of the functionality and shortcomings of each of these interfaces provides users of PyKX with the ability to contextualise aspects of this libraries design.

!!! Warning "Interface support"

	Of the interfaces described below both embedPy and PyQ are maintained by KX and are supported on a best efforts basis under the [Fusion](https://code.kx.com/q/interfaces) initiative. qPython is in maintenance mode and not supported by KX. It is suggested that users migrate from using these historical interfaces to using PyKX to pick up the latest updates from KX.

### EmbedPy

EmbedPy provides an approach for using Python from q, but it does not provide a way to interface with q from Python. The EmbedPy interface was designed specifically for q developers who wish to leverage functionality in Python which is not immediately/easily available to q developers. This includes but is not limited to Machine Learning functionality, statistical methods, and plotting.

### PyQ

PyQ brings the Python and q interpreters into the same process so that code written in either of the languages operates on the same data. Unfortunately to use PyQ one must execute the PyQ binary, or start PyQ from q. This makes PyQ unsuitable for most Python use-cases which require the use of a Python binary. It is not possible to start a Python process, and then import PyQ.

Because of this, it is impossible to develop Python software that depends on PyQ, unless you are willing to run it in a different process. This barrier reasonably makes Python developers hesitant to use PyQ, as it locks them into using the PyQ binary to execute their program.

PyKX provide a more Pythonic approach to interfacing between Python and q than is offered by PyQ. For one PyKX can be run explicitly from a Python session unlike PyQ which relies on execution of a special binary or initialization from q. In addition to this PyKX provides a class-based hierarchical type system built atop q's type management system. This allows for sub-classes to be used. PyKX also provides a [context interface](../api/pykx-execution/ctx.md) which can be used to load q scripts and interact with q namespaces in a Pythonic manner. Finally the query functionality provided by PyKX allows for more flexibility in the objects used in tabular updates through use of the q functional select, exec, update and delete functions rather than generating a qSQL statement.

### qPython

Like PyKX, qPython takes a Python-first approach, but unlike PyKX it works entirely over IPC. Python objects being sent to q and q objects being returned are serialized, sent over a socket, and then deserialized. While this is a common use case for many users, it is a very expensive process both in terms of processing time and memory usage. For many users wishing to use q data within Python for analysis this overhead can be limiting.

At a fundamental level the IPC interface provided by PyKX is different to that provided by qPython. Firstly qPython reads and converts data directly from the socket using Python, in comparison PyKX leverages the q memory space embedded within Python to store the data for later conversion from a referenced location in that memory space.

This provides two distinct advantages:

1. There is increased flexibility in the supported conversion types, within PyKX data can be converted to Python, Numpy, Pandas and PyArrow data types from their underlying q representation. This is in contrast to qPython which automatically converts to Numpy/Pandas based on underlying wire type.
2. By converting from q rather than the socket representation we can make greater use of the underlying C representation of the data which improves performance in data decoding. This has the effect of boosting performance up to 8x that of qPython when managing large complex datasets

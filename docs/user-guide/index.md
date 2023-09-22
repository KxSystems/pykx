# User Guide

The user guide provided here covers all the core elements of interacting with and creating PyKX objects using the PyKX library. Each of the subsections of this user guide explains the library based on applications of the software to various scenarios rather than through a function by function execution guide. For example [Creating PyKX objects](fundamentals/creating.md) describes the many ways you can generate PyKX objects for later use.

This user guide is broken into two sections:

1. `Configuration` - This details all the options of configuration available to PyKX using a configuration file and/or environment variables.
2. `Fundamentals` - This defines the basic concepts necessary to interact with PyKX objects, clarifies elements of the libraries usage and some technical considerations which should be made by new users when trying to make the most out of PyKX.
3. `Advanced usage and performance considerations` - A user should only make use of this section once they are familiar with the fundamentals section of this documentation. This section outlines the usage of advanced features of the library such as running under q and IPC interactions. Additionally it outlines performance enhancements that can be enabled by a user and limitations imposed by embedding q/kdb+ within a Python environment.

The following outlines the various topics covered within the above sections:

## Fundamentals

| Section                                                               | Description |
|-----------------------------------------------------------------------|-------------|
| [Interacting with PyKX objects](fundamentals/creating.md)             | How can you create and interact with PyKX objects in various ways. |
| [Evaluating q code with PyKX](fundamentals/evaluating.md)             | How do you evaluate q code under various conditions. |
| [Querying PyKX data](fundamentals/querying.md)                        | How do you query tables locally and remotely using PyKX with the qSQL query API and SQL|
| [Indexing PyKX objects](fundamentals/indexing.md)                     | What considerations need to be made when indexing and accessing elements within PyKX objects Pythonically. |
| [Handling nulls and infinities](fundamentals/nulls_and_infinities.md) | How are null and infinite values handled within PyKX.|

## Advanced usage and performance considerations

| Section                                                              | Description |
|----------------------------------------------------------------------|-------------|
| [Communicating via IPC](advanced/ipc.md)                             | How can you interact synchronously and asynchronously with a kdb+/q server. |
| [Using q functions in a Pythonic way](advanced/context_interface.md) | Evaluating and injecting q code within a Python session using a Pythonic context interface which exposes q objects as first class Python objects. |
| [Numpy integration](advanced/numpy.md)                               | Description of the various low-level integrations between PyKX and numpy. Principally describing NEP-49 optimisations and the evaluation of numpy functions using PyKX vectors directly. |
| [Modes of operation](advanced/modes.md)                              | A brief description of the modes of operation of PyKX outlining it's usage in the presence and absence of a license and the limitations that this imposes.
| [Performance considerations](advanced/performance.md)                | Guidance on how to treat management and interactions with PyKX objects to achieve the best performance possible. |
| [Library limitations](advanced/limitations.md)                       | For users familiar with q/kdb+ and previous Python interfaces what limitations does PyKX impose. |


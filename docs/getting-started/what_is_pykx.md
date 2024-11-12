---
title:   What is PyKX?
description: Overview of Pykx
date: June 2024
author: KX Systems, Inc.,
tags: about PyKX, q/kdb+, use cases,
---

# What is PyKX?

_This page briefly describes PyKX, its use cases, and its connection with q/kdb+._

## Introduction

**PyKX** is a Python-first interface to **kdb+** (the world's fastest time-series database) and **q** (kdb+'s underlying vector programming language). PyKX is the result of 10+ years of integrations between two languages: Python and q. Its aim is to help users query and analyze huge amounts of in-memory and on-disk time-series data, significantly faster than other libraries.

## Use cases

PyKX supports three main use cases, allowing Python data engineers and data scientists to:

1. Store, query, manipulate and use q objects within a Python process.
2. Query external q processes via an [Inter-Process Communication (IPC)](..//user-guide/advanced/ipc.md) interface.
3. Embed Python functionality within a native q session using its [under q](../pykx-under-q/intro.md) functionality.

??? Note "Expand to learn more about q/kdb+"

    Used throughout the financial sector for 25+ years, q and kdb+ have been a cornerstone of modern financial markets. This technology provides a storage mechanism for historical market data and performant tooling to analyze this vast streaming, real-time and historical data.

    - **Kdb+** is a high-performance column-oriented database designed to process and store large amounts of data. Commonly accessed data is available in RAM which makes it faster to access than disk stored data. Operating with temporal data types as a first class entity the use of q and it's query language qsql against this database creates a highly performant time-series analysis tool available.

    - **q** is the vector programming language which is used for all interactions with kdb+ databases, known both for its speed and expressiveness. PyKX exposes q as a domain-specific language (DSL) embedded within Python. The assumption is that q is mainly used for data processing and database management. 
    
    This approach benefits users in two ways:
    
    - Helps users familiar with q to make the most of its advanced analytics and database management.
    - Empowers kdb+/q users who lack q expertise to get up and running quickly.

    For more information on using q/kdb+ and getting started with see the following links:

    - [An introduction to q/kdb+](https://code.kx.com/q/learn/tour/)
    - [Tutorial videos introducing kdb+/q](https://code.kx.com/q/learn/q-for-all/)

## PyKX vs. Python/q interfaces

There are three historical interfaces which allow interoperability between Python and q/kdb+:

1. [Embedpy](https://code.kx.com/q/ml/embedpy)
2. [PyQ](https://github.com/KxSystems/pyq)
3. [qPython](https://github.com/KxSystems/pyq)

How does PyKX compare to other q interfaces for Python? Here’s a TL;DR comparison table highlighting the key differences between EmbedPy, PyQ, qPython, and PyKX:

| **Feature**      | **EmbedPy**                                      | **PyQ**                               | **qPython**                                                             | **PyKX**                                                                         |
| ---------------- | ------------------------------------------------ | ------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Interoperability | Python from q                                    | Python & q in same process            | Python-first, over IPC                                                  | Python-first, in-process & IPC                                                   |
| Execution        | Designed for q developers, run in q session      | Requires PyQ binary or start from q   | Processing completed on q session via IPC, deserialization using Python | Run from Python session, class-based type system                                 |
| Use Case         | Leverage Python functionality not available in q | Operate on same data across languages | Common use case, expensive in processing & memory                       | Store, query, manipulate q objects within Python, via IPC or Python in q session |
| Flexibility      | \-                                               | Locked into using PyQ binary          | \-                                                                      | Pythonic interface, context interface for q scripts, q first mode, IPC available |
| Data Conversion  | q to/from Numpy/Python only                      | q to/from Numpy/Pandas only           | Data converted directly from socket                                     | Leverages q memory space embedded within Python, supports NumPy, Pandas, PyArrow |

??? Note "Expand to learn more about EmbedPy, PyQ, qPython, and PyKX"

    To give you a clear understanding of how each interface operates and their suitability for different use cases, here are some additional details:															
                                                                
    - **EmbedPy** allows using Python from q but does not interface with q from Python. It’s mainly for q developers to access Python functionalities like machine learning, statistical methods, and plotting.															
    - **PyQ** integrates Python and q interpreters in the same process, but it requires executing a special PyQ binary or starting from q, which is not ideal for Python use cases that require a standard Python binary.															
    - **qPython** takes a Python-first approach but works entirely over IPC (Inter-Process Communication), meaning Python objects sent to q and q objects returned are serialized, sent over a socket, and then deserialized, which can be resource-intensive.															
    - **PyKX** supports storing, querying, manipulating, and using q objects within a Python process and querying external q processes via IPC. PyKX provides a more Pythonic approach with a class-based hierarchical type system and a context interface for interacting with q scripts in a Pythonic manner.															

!!! tip "EmbedPy, PyQ, qPython: Interface support"

	KX maintains both embedPy and PyQ on a best-efforts basis under the [Fusion](https://code.kx.com/q/interfaces) initiative. qPython is in maintenance mode, not supported by KX. If you're using EmbedPy, PyQ, or qPython, we recommend switching to PyKX to pick up the latest updates from KX.

## Next steps

- Follow the [Installation guide](installing.md)
- Get up and running with the PyKX [Quickstart guide](quickstart.md)
- Get to know more functionalities with the [PyKX Introduction Notebook](../examples/interface-overview.ipynb)

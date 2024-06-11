# What is PyKX?

## Introduction

PyKX is a Python first interface to the world's fastest time-series database kdb+ and its underlying vector programming language, q. PyKX takes a Python first approach to integrating q/kdb+ with Python following 10+ years of integrations between these two languages. Fundamentally it provides users with the ability to efficiently query and analyze huge amounts of in-memory and on-disk time-series data.

This interface exposes q as a domain-specific language (DSL) embedded within Python, taking the approach that q should principally be used for data processing and management of databases. This approach does not diminish the ability for users familiar with q, or those wishing to learn more about it, from making the most of its advanced analytics and database management functionality. Rather it empowers those who want to make use of the power of kdb+/q who lack this expertise to get up and running quickly.

PyKX supports three principal use cases:

1. It allows users to store, query, manipulate and use q objects within a Python process.
2. It allows users to query external q processes via an IPC interface.
3. It allows users to embed Python functionality within a native q session using it's [under q](../pykx-under-q/intro.md) functionality.

Users wishing to install the library can do so following the instructions [here](installing.md).

Once you have the library installed you can get up and running with PyKX following the quickstart guide [here](quickstart.md).

## What is q/kdb+?

Mentioned throughout the documentation q and kdb+ are respectively a highly efficient vector programming language and highly optimised time-series database used to analyse streaming, real-time and historical data. Used throughout the financial sector for 25+ years this technology has been a cornerstone of modern financial markets providing a storage mechanism for historical market data and tooling to make the analysis of this vast data performant.

Kdb+ is a high-performance column-oriented database designed to process and store large amounts of data. Commonly accessed data is available in RAM which makes it faster to access than disk stored data. Operating with temporal data types as a first class entity the use of q and it's query language qsql against this database creates a highly performant time-series analysis tool available.

q is the vector programming language which is used for all interactions with kdb+ databases and which is known both for its speed and expressiveness.

For more information on using q/kdb+ and getting started with see the following links:

- [An introduction to q/kdb+](https://code.kx.com/q/learn/tour/)
- [Tutorial videos introducing kdb+/q](https://code.kx.com/q/learn/q-for-all/)

## Next steps

- [Installation guide](installing.md)
- [Quickstart guide](quickstart.md)
- [User guide introduction](../user-guide/index.md)

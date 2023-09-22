# PyKX Documentation

## About

PyKX is an interface between the vector programming language q, it's associated time-series columnar database kdb+, underlying data types and Python. This provides users with the ability to efficiently interact with the worlds fastest time-series database and to apply analytics against vast amounts of data in-memory or on-disk.

PyKX takes a Python-first approach to this integration. This is to say it takes the stance that q should be used primarily as a data processing engine and database. Additionally q should be used primarily as a domain-specific language (DSL) embedded within Python, which is a more general-purpose language.

This is not to say that the interface is limiting to developers familiar to q. Expert users of q using this interface are capable of running the same analyses they would normally run within a q process using PyKX. If a user so chooses, they can simply run q code through PyKX. However, through PyKX, Python developers who have no experience with q can access and leverage q through a Pythonic interface.

The documentation presented here should provide a user with the stepping stones needed to query vast amounts of historical data, apply vector analytics to q data types and see the value of operating with the q data formats for the application of analytic functions.


## Documentation Breakdown

### [Getting Started](getting-started/what_is_pykx.md)

This provides documentation for users that are new to q/kdb+ and PyKX. Included in this section are installation instructions and quickstart guides which should allow a user to get up and running.

### [User Guide](user-guide/index.md)

Our user guide provides useful information which allows a user to get an understanding of the key concepts behind PyKX, how the library is intended to be used and includes examples of the library functionality.

### [API](api/pykx-execution/q.md)

The API reference guide contains detailed descriptions of the functions, modules and objects managed by PyKX. It describes how functions can be called, data types manipulated and data queried in addition to much broader usage of the library. Use of the API reference assumes you have a strong understanding of how the library is intended to be used through the getting started and user guide sections.

### Extras

The `Extras` section contains additional information users may find interesting, this includes the following:

- [Known interface issues](extras/known_issues.md)
- [Comparisons against other q-Python interfaces](extras/comparisons.md)

## [Getting Help](support.md)

If you have any issues or questions relating to PyKX the support page provides users with links to helpful community locations and support contact information for PyKX.

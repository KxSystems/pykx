# PyKX Roadmap

This page outlines areas of development focus for the PyKX team to provide you with an understanding of the development direction of the library. This is not an exhaustive list of all features/areas of focus but should give you a view on what to expect from the team over the coming months. Additionally this list is subject to change based on the complexity of the features and any customer feature requests raised following the publishing of this list.

If you need a feature that's not included in this list please let us know by raising a [Github issue](https://github.com/KxSystems/pykx/issues)!

## Nov 2023 - Jan 2024

- Support Python 3.12
- Tighter integration with [Streamlit](https://streamlit.io/) allowing streamlit applications to interact with kdb+ servers and on-disk databases
- User defined Python functions to be supported when operating with local qsql.select functionality
- [JupyterQ](https://github.com/KxSystems/jupyterq) and [ML-Toolkit](https://github.com/KxSystems/ml) updates to allow optional PyKX backend replacing embedPy
- Pythonic data sorting for PyKX Tables

## Feb - Apr 2024

- Database management functionality allowing for Pythonic persistence and management of on-disk kdb+ Databases (Beta)
- Improvements to multi-threaded PyKX efficiency, reducing per-call overhead for running PyKX on separate threads
- Configurable initialisation logic in the absense of a license. Thus allowing users who have their own workflows for license access to modify the instructions for their users.
- Addition of `cast` keyword when inserting/upserting data into a table reducing mismatch issues

## Future

- Tighter integration between PyKX/q objects and PyArrow arrays/Tables
- Expansion of supported datatypes for translation to/from PyKX
- Continued additions of Pandas-like functionality on PyKX Table objects
- Performance improvements through enhanced usage of Cython
- Real-time/Streaming functionality utilities
- Data pre-processing and statitics modules for operation on PyKX tables and vector objects

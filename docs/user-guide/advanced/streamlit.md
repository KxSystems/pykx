---
title: Streamlit Integration
description: Integrate PyKX Connections into you Streamlit application
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, streamlit, visualisation, query, web, application
---

# Streamlit Integration

!!! Warning

	Streamlit makes use of a caching mechanism which makes use of multiple threads, to make use of PyKX under these conditions it is suggested that users set `PYKX_THREADING` as `True`, for more information on the threading feature see [here](threading.md), for information on setting configuration see [here](../configuration.md).

[Streamlit](https://streamlit.io) provides an open source framework allowing users to turn Python scripts into sharable web applications. Functionally, Streamlit provides access to external data-sources using the concept of `connections` which allow users to develop conforming APIs which will integrate directly with streamlit applications as an extension connection types.

The integration outlined below makes use of this by generating a new `pykx.streamlit.PyKXConnection` connection type which provides the ability to create synchronous connections to existing q/kdb+ sessions.

A full breakdown of the API documentation of this class can be found [here](../../api/streamlit.md).

## Requirements and limitations

To run this functionality, users must have `streamlit>=1.28` installed local to their Python session.

This can be installed using the following command when installing PyKX:

```bash
pip install pykx[streamlit]
```

## Using PyKX with Streamlit

The PyKX Streamlit integration provides users with the ability to do the following:

1. Establish a Streamlit compliant connection to a q/kdb+ process
1. Check health of a connection and restart connection as necessary
1. Query the remote process using `q`, `SQL` and `qSQL`

As mentioned above PyKX provides a streamlit connection type `pykx.streamlit.PyKXConnection` which can be used with the streamlit [`st.connection`](https://docs.streamlit.io/develop/api-reference/connections/st.connection) functionality to integrate your streamlit application with PyKX.

In the below section we will discuss how these connections are established, maintained and used for query.

### Connecting to kdb+

In the below example we connect to a variety of kdb+ processes on port 5050 with a streamlit connection. In each case we name the connection `'pykx'` but this name is arbitrary and is open to a user to modify

=== "Basic Connection generation"

	Connect to a process which does not require username/password

	```python
        import streamlit as st
        import pykx as kx
        connection = st.connection('pykx',
                                   type=kx.streamlit.PyKXConnection,
                                   host='localhost',
                                   port=5050)
	```

=== "User - Password protected connection"

	Connect to a process requiring a username/password to be provided

	```python
	import streamlit as st
	import pykx as kx
	connection = st.connection('pykx',
                                   type=kx.streamlit.PyKXConnection,
                                   host='localhost',
                                   port=5050,
                                   username='user',
                                   password='password')
	```

=== "Connection to automatically reconnect if dropped"

	Attempt to reconnect to the process if connection is lost 5 times on an exponential backoff

	```python
	import streamlit as st
	import pykx as kx
	connection = st.connection('pykx',
                                   type=kx.streamlit.PyKXConnection,
                                   host='localhost',
                                   port=5050,
                                   reconnection_attempts=5)
	```

### Checking and restoring the health of your connections

In streamlit, your application may be running for a significant period of time. In such situations it is not uncommon for your original connection to a server to drop.

To help with such cases there are a number of methods provided by PyKX to recover your environment:

- The addition of an `is_healthy` method to facilitate checking if the remote server can be interacted with.
- The availability of a `reset` method to allow a connection which is deemed not to be healthy to be re-established.

The following provides an example code block showing use of these methods

```python
import streamlit as st
import pykx as kx
connection = st.connection('pykx',
                           type=kx.streamlit.PyKXConnection,
                           host='localhost',
                           port=5050)

if not connection.is_healthy():
    connection.reset()
```

### Querying using a connection

Process query is available in three formats

1. SQL
1. Pythonic qSQL
1. q

The following code blocks show use of each of these query types.

In each case we assume that a healthy connection has been established and the user is attempting to retrieve the maximum value of data in column 'price' by symbol ('sym') from a table named 'trade'

=== "Pythonic qSQL"

	```python
	>>> conn.query('trade',
        ...     columns=kx.Column('price').max(),
        ...     by=kx.Column('sym'),
        ...     format='qsql')
	pykx.KeyedTable(pykx.q('
	sym | price    
	----| ---------
	AAPL| 0.9877844
	GOOG| 0.9598964
	IBM | 0.9785   
	'))
	```

=== "SQL"

	SQL querying requires that your server have access to the [SQL interface to kdb+](https://code.kx.com/insights/core/sql.html) to be loaded on the server

	```python
	>>> conn.query('select sym, max(price) from trade GROUP BY sym', format='sql')
	pykx.Table(pykx.q('
	sym  price    
	--------------
	AAPL 0.9877844
	GOOG 0.9598964
	IBM  0.9785   
	'))
	```

=== "q"

	```python
	>>> conn.query('select max price by sym from trade', format='q')
	pykx.KeyedTable(pykx.q('
	sym | price    
	----| ---------
	AAPL| 0.9877844
	GOOG| 0.9598964
	IBM | 0.9785   
	'))
	```

## Example

Now that you have seen some of the functions in action you can generate a streamlit script to read data from a table and generate a graph.

### Pre-requisites

You must have available to you a q session running on port 5050 and which has available the following table

```q
\p 5050
N:1000
tab:([]sym:N?`AAPL`MSFT`GOOG`FDP;price:100+N?100f;size:10+N?100)
```

### Script

The following script generates a simple streamlit application which

1. Sets environment variables and imports required libraries
1. Defines a function to run for generation of the streamlit application completing the following
  1. Name the streamlit application
  1. Create a connection to the q process initialised on port 5050
  1. Query the q process retrieving a small tabular subset of data using the Pythonic Query API
  1. Generates a Matplotlib graph directly using the PyKX table
  1. Displays both the table and graph

The script which follows can be downloaded [here](examples/streamlit.py)

??? Note "Expand here to view the script text"

	```python
	# Set environment variables needed to run Steamlit integration
	import os

	# This is optional but suggested as without it's usage caching
	# is not supported within streamlit
	os.environ['PYKX_THREADING'] = 'true'

	import streamlit as st
	import pykx as kx
	import matplotlib.pyplot as plt

	def main():
	    st.header('PyKX Demonstration')
	    connection = st.connection('pykx',
	                               type=kx.streamlit.PyKXConnection,
	                               port=5050)
	    if connection.is_healthy():
	        tab = connection.query(
	            'tab',
	            where = kx.Column('size') < 11
	            )
	    else:
	        try:
	            connection.reset()
	        except BaseException:
	            raise kx.QError('Connection object was not deemed to be healthy')
	    fig, x = plt.subplots()
	    x.scatter(tab['size'], tab['price'])

	    st.write('Queried kdb+ remote table')
	    st.write(tab)

	    st.write('Generated plot')
	    st.pyplot(fig)

	if __name__ == "__main__":
	   try:
	       main()
	   finally:
	       kx.shutdown_thread()
	```

## Next Steps

- Learn more about querying your data [here](../fundamentals/query/index.md)
- Learn more about Interprocess Communication [here](ipc.md)

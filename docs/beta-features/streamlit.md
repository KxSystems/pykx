# Streamlit Integration

!!! Warning

	This module is a Beta Feature and is subject to change. To enable this functionality for testing please follow the configuration instructions [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`

	This functionality is presently not supported on Windows, for full utilisation of this functionality `PYKX_THREADING='true'` nust be set in configuration.

## Introduction

[Streamlit](https://streamlit.io) provides an open source framework allowing users to turn Python scripts into sharable web applications. Functionally, Streamlit provides access to external data-sources using the concept of `connections` which allow users to develop conforming APIs which will integrate directly with streamlit applications as extension connection types.

The integration outlined below makes use of this by generating a new `pykx.streamlit.PyKXConnection` connection type which provides the ability to create synchronous connections to existing q/kdb+ sessions.

A full breakdown of the API documentation of this class can be found [here](../api/streamlit.md).

## Requirements and limitations

To run this functionality, users must have `streamlit>=1.28` installed local to their Python session.

This can be installed using the following command:

```bash
pip install pykx[streamlit]
```


## Functional walkthrough

This walkthrough will demonstrate the following steps:

1. Initialize a q/kdb+ server on a specified port and populating some data.
1. Generate a `streamlit.py` script which queries the q server and creates a basic streamlit application.
1. Run the streamlit application and view locally

### Initializing a q/kdb+ server

This step ensures you have a q process running and a kdb+ table available to query. If you have this already, proceed to the next step.

Ensure that you have q installed. If you do not have this installed please follow the guide provided [here](https://code.kx.com/q/learn/install/), retrieving your license following the instructions provided [here](https://kx.com/kdb-insights-personal-edition-license-download).

```bash
q -p 5050
```

Create a table which you will use within your Python analytics defined below.

```q
q)N:1000
q)tab:([]sym:N?`AAPL`MSFT`GOOG`FDP;price:100+N?100f;size:10+N?100)
```

Set a requirement for users to provide a username/password if you wish to add security to your q process.

```q
.z.pw:{[u;p]$[(u~`user)&p~`password;1b;0b]}
```

### Generate a streamlit script/application

The following script generates a simple streamlit application which:

1. Set environment variables and import required libraries
1. Define a function to run for generation of the streamlit application
    1. Name the streamlit application.
    1. Create a connection to the q process initialized on port 5050 above.
    1. Query the q process retrieving a small tabular subset of data using a qsql statement.
    1. Generate a Matplotlib graph directly using the PyKX table.
    1. Display both the table and graph

This script can additionally be downloaded [here](examples/streamlit.py).

```python
# Set environment variables needed to run Steamlit integration
import os
os.environ['PYKX_BETA_FEATURES'] = 'true'

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
                               port=5050,
                               username='user',
                               password='password')
    if connection.is_healthy():
        tab = connection.query('select from tab where size<11')
    else:
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

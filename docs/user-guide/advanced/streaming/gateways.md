---
title: Managing query routing
description: How to manage what and how users can query data
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, query, routing, analytics
---

# Manage query routing

_This page outlines how to provide a central, password-protected query location for users._

!!! Warning "Disclaimer"

         The functionality outlined below provides the necessary tools for users to build complex streaming infrastructures. The generation and management of such workflows rest solely with the users. KX supports only individual elements used to create these workflows, not the end-to-end applications.

When providing users with access to data within your system, you must consider the following priorities:

1. How can you provide a central point for users to query your data?
2. How do you regulate the users who can query your system?
3. How do you route queries to multiple processes containing different datasets and aggregate this data?

A `#!python Gateway` process can handle each of these. The gateway is responsible for defining the processes that can be queried within your system and regulates via user-configured logic what is required for a user to establish a connection to the gateway.

PyKX provides a simplistic gateway which allows connections to multiple processes and supports synchronous queries against your processes. Although it doesn't scale to large numbers of high traffic use-cases, it provides a starting infrastructure suitable for small teams of developers.

!!! note "Have your say"

	The above usage patterns provide a basic gateway design but does not cover all cases/usage patterns, if there is functionality that you would like to see let us know by opening an issue [here](https://github.com/KxSystems/pykx/issues).

## Create a gateway

In the following sections we will generate a Gateway process to help you to complete the following:

1. Limit users permitted to query your APIs to those with a username and password provided in a supplied text file.
2. Generate a custom function which queries the APIs generated on your `#!python RTP` and `#!python HDB` processes [here](custom_apis.md#add-an-api-to-an-existing-rtp-and-hdb) aggregating the results.

### Configure your gateway

Before adding custom gateway APIs and secure login to the process, configure the gateway to operate on port 5015 with established connections against two processes:

1. `#!python 'rtp'`: The Real-Time Processor established [here](rta.md) on port 5014
2. `#!python 'hdb'`: The Historical Database established [here](basic.md) on port 5012

```python
gateway = kx.tick.GATEWAY(port=5015, connections={'rtp': 'localhost:5014', 'hdb': 'localhost:5012'})
```

If you need to add additional connections once you initialized the `#!python GATEWAY`, use the `#!python add_connections` function as shown below:

```python
gateway.add_connections({'rtp': 'localhost:5014'})
```

??? "API documentation"
    The following bullet-points provide links to the various functions used within the above section

    - [`kx.tick.GATEWAY`](../../../api/tick.md#pykx.tick.GATEWAY)
    - [`gateway.add_connections`](../../../api/tick.md#pykx.tick.GATEWAY.add_connection)

### Add a custom username/password check

Once you have an initialized Gateway process, define a custom username/password check which any user connecting to the gateway will be validated against. In the example below, the validation function checks that a user is named `#!python test_user` and has a password matching the regex `#!python password.*`

```python
def validation_function(username, password):
    if username == 'test_user':
        pattern = re.compile("password.*")
        if bool(pattern.match(password)):
            return True
    return False
```

Now that you have specified the validation function, set this function on the `#!python Gateway` process. For this to operate, you need to ensure the library `#!python re` is available:

```python
gateway.libraries({'re': 're'})
gateway.connection_validation(validation_function)
```

Users attempting to interact with this gateway will now need to adhere to the above conditions providing the username `#!python test_user` and a password `#!python password.*`.

??? "API documentation"
    The following bullet-points provide links to the various functions used within the above section

    - [`gateway.libraries`](../../../api/tick.md#pykx.tick.STREAMING.libraries)
    - [`gateway.connection_validation`](../../../api/tick.md#pykx.tick.GATEWAY.connection_validation)

### Define a custom API for users to call

After establishing the gateway and defining a validation function for connecting processes, add a Custom Gateway API.

Within the Gateway process, there is a Python class defined `#!python gateway` which contains a function `#!python call_port`. This function takes the name given to a port when establishing remote connections [here](#configure-your-gateway) and the parameters required to call this function.

When we developed our custom query APIs [here](custom_apis.md#add-an-api-to-an-existing-rtp-and-hdb) we registered an API `#!python symbol_count` on both the `#!python rtp` and `#!python hdb` processes, the following function definition makes use of the `#!python call_port` function to invoke these functions for a specified table and symbol combination.

```python
def gateway_function(table, symbol):
    rtp = gateway.call_port('rtp', table, symbol)
    try:
        hdb = gateway.call_port('hdb', table, symbol)
    except BaseException:
        print('Failed to retrieve data from HDB')
        hdb = 0
    return rtp + hdb

gateway.register_api('sum_of_symbols', gateway_function)
```

Now that your gateway function has been registered, start the gateway:

```python
gateway.start()
```

Users should now be in a position to query the `#!python sum_of_symbols` API on the Gateway process as follows:

```python
with kx.SyncQConnection(port=5015, username='test_user', password='password123') as q:
    ret = q('sum_of_symbols', 'trade', 'AAPL')
ret
```

??? "API documentation"
    The following bullet-points provide links to the various functions used within the above section

    - [`gateway.register_api`](../../../api/tick.md#pykx.tick.STREAMING.register_api)

### Run all setup at once

To help with restart and to simplify the configuration of your system, you can complete each of the sections above at configuration time for your initialized class. The following code block contains all the code used to configure the gateway:

```python
def validation_function(username, password):
    if username == 'test_user':
        pattern = re.compile("password.*")
        if bool(pattern.match(password)):
            return True
    return False

def gateway_function(table, symbol):
    rtp = gateway.call_port('rtp', table, symbol)
    try:
        hdb = gateway.call_port('hdb', table, symbol)
    except BaseException:
        print('Failed to retrieve data from HDB')
        hdb = 0
    return rtp + hdb

gateway = kx.tick.GATEWAY(
    port=5015,
    connections={'rtp': 'localhost:5014', 'hdb': 'localhost:5012'},
    libraries={'re':'re'},
    apis={'sum_of_symbols': gateway_function},
    connection_validator=validation_function
    )
gateway.start()
```

The advantage of this approach is that it allows process/workflow restart, for example, in case you lose connection to a downstream process. As all definitions are cached in configuration, you can easily restart them.

```python
gateway.restart()
```

??? "API documentation"
    The following bullet-points provide links to the various functions used within the above section

    - [`kx.tick.GATEWAY`](../../../api/tick.md#pykx.tick.GATEWAY)
    - [`gateway.start`](../../../api/tick.md#pykx.tick.GATEWAY.start)
    - [`gateway.restart`](../../../api/tick.md#pykx.tick.GATEWAY.restart)

## Next steps

For some further reading, here are some related topics you may find interesting:

- Learn more about Interprocess Communication (IPC) [here](../ipc.md).
- Create a Historical Database from static datasets [here](../database/index.md)

---
title: Custom Query API Development
description: How to generate a custom query API
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, streaming, publishing
---

# Custom query API development

_This page outlines how you can augment your streaming process with accessible named query APIs._

!!! Warning "Disclaimer"

         The functionality outlined below provides the necessary tools for users to build complex streaming infrastructures. The generation and management of such workflows rest solely with the users. KX supports only individual elements used to create these workflows, not the end-to-end applications.

The addition and use of custom query APIs is often crucial for making your data accessible to users. Users connected to your process via IPC or by a querying Gateway process call these APIs. You can place custom query APIs on any process type discussed in the [basic](basic.md), [analysis](rta.md) and [subscription](subscribe.md) sections.

In each case, you can add a query API by calling the `#!python register_api` method on each of the process types or during the configuration of an [`#!python RTP`](../../../api/tick.md#pykx.tick.RTP) , [`#!python HDB`](../../../api/tick.md#pykx.tick.HDB) or [`#!python GATEWAY`](../../../api/tick.md#pykx.tick.GATEWAY) process in your system. A breakdown of gateway processes follows this section [here](gateways.md). In the examples below we add query APIs to the historical database created when configuring the [basic](basic.md) infrastructure and the RTP processing the aggregate dataset.

## Configure an API for your Real-Time Processor

You can add APIs to your process at configuration time or while the process is in operation, to allow an iterative development. The following sections show how both approaches can be achieved to create a Python function which takes multiple parameters:

1. The `#!python table` which is being queried
2. The `#!python symbol` which a user is interested in

And returns the number of instances of that symbol:

```python
def custom_api(table, symbol):
    return kx.q.sql(f'select count(*) from {table} where sym like $1', symbol)['xcol'][0]
```

### Add an API to an existing RTP and HDB

Now that you have the function definition, use the `#!python register_api` function to augment the `#!python rtp` class created [here](rta.md#start-your-rtp). 

```python
rtp.register_api('symbol_count', custom_api)
```

Similarly, you can add the equivalent API to your `#!python HDB` process generated [here](basic.md) by accessing the `#!python hdb` class as follows:

```python
basic.hdb.register_api('symbol_count', custom_api)
```

??? "API documentation"
    The following bullet-points provide links to the various functions used within the above section

    - [`rtp.register_api`](../../../api/tick.md#pykx.tick.STREAMING.register_api)

### Add an API when configuring your system

In the previous section you added custom APIs to a running system. To make APIs available on restart, you can add them at the configuration time for the processes. For instance, let's modify the example [here](rta.md#run-all-setup-at-once) to include an API.

If we're adding an API at configuration, it's supplied as a dictionary mapping the name of the API to the API code:

```python
def preprocessor(table, data):
    if table == 'trade':
        return data
    else:
        return None

def postprocessor(table, data):
    agg = kx.q[table].select(
        columns = {'min_px':'min price',
                   'max_px': 'max price',
                   'spread_px': 'max[price] - min price'},
         by = {'symbol': 'symbol'})
    kx.q['agg'] = agg # Make the table accessible from q
    with kx.SyncQConnection(port=5010, wait=False, no_ctx=True) as q:
        q('.u.upd', 'aggregate', agg._values)
    return None

def custom_api(table, symbol):
    return kx.q.sql(f'select count(*) from {table} where sym like $1', symbol)['xcol'][0]

rtp = kx.tick.RTP(port=5014,
                  subscriptions = ['trade'],
                  libraries={'kx': 'pykx'},
                  pre_processor=preprocessor,
                  post_processor=postprocessor,
                  apis={'symbol_count': custom_api},
                  vanilla=False)
rtp.start({'tickerplant': 'localhost:5013'})
```

Currently we don't support the addition of APIs to the components of the [basic infrastructure](basic.md) at startup. To configure a historical database at startup with more fine-grained control, configure it manually as outlined [here](complex.md).

??? "API documentation"
    The following bullet-points provide links to the various functions used within the above section

    - [`kx.tick.RTP`](../../../api/tick.md#pykx.tick.RTP)
    - [`rtp.start`](../../../api/tick.md#pykx.tick.RTP.start)

### Test an API
In the above we are defining that users calling this function will do so by making use of the named function `#!python symbol_count`. You can directly test this once registered, as it follows:

```python
rtp('symbol_count', 'trade', 'AAPL')
```

Alternatively, you can test this using IPC:

```python
with kx.SyncQConnection(port=5014, no_ctx=True) as q:
    q('symbol_count', 'trade', 'AAPL')
```

## Next steps

Now that you have data being published to your system you may be interested in the following:

- Generate a query routing gateway to allow queries across multiple processes [here](gateways.md).
- Manually configuring the [basic infrastructure](basic.md) as outlined [here](complex.md).

For some further reading, here are some related topics:

- Learn more about Interprocess Communication(IPC) [here](../ipc.md).
- Learn more about how you can query your data [here](../../fundamentals/query/index.md)

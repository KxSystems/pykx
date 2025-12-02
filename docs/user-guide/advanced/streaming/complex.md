---
title: Complex Streaming Control
description: How to edit/manage your streaming workflows with PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, streaming, simple
---

# Complex streaming control

_This page outlines some of the more complex and fine-grained controls that are possible for your streaming workflows._

!!! Warning "Disclaimer"

	 The functionality outlined below provides the necessary tools for users to build complex streaming infrastructures. The generation and management of such workflows rest solely with the users. KX supports only individual elements used to create these workflows, not the end-to-end applications.

[Basic infrastructure](basic.md), [Analysing streaming data](rta.md) and [Custom query API development](custom_apis.md) sections deal with the simplest interactions supported by PykX. Let's explore additional keyword arguments/functionalities that can provide significant value in building your infrastructure.

The sections below discuss in detail why it's important and how to update the examples used throughout the other sections of the Real-Time Data Capture documentation. The following highlights the topics covered:

| Topic                            | Description |
|:---------------------------------|:------------|
| Fine-grained ingest control    | Instead of relying on the packaged [basic](basic.md) logic to generate your tickerplant, RDB and HDB, control these processes more explicitly and learn why this is useful. |
| Process logs                   | Learn how to modify startup of your processes to save output to files or print to your process. |
| How to stop processes          | You already know how to start and restart processes. This section shows you how to stop them. |


## Fine-grained ingest control

In the [basic infrastructure](basic.md) section we made use of the function [`#!python kx.tick.BASIC`](../../../api/tick.md#pykx.tick.BASIC) to start the component parts of a PyKX streaming workflow namely:

- [Tickerplant](basic.md#tickerplant): The ingestion point which logs incoming messages and publishes messages to down-stream subscribers.
- [Real-Time Database(RDB)](basic.md#real-time-database): A process which contains the current day's data in-memory and writes the data to disk at end-of-day.
- [Historical Database(HDB)](basic.md#historical-databases): A process on which data for days prior to the current day has been loaded as a memory-mapped on-disk dataset.

While the single-call basic infrastructure is useful, you might want to load these process types on separate virtual/physical machines. For example, you might consider loading your RDB on a process with significantly higher RAM requirements to your HDB, where user queries are limited in expected RAM by well-controlled APIs.

A full breakdown of the APIs for each of these process types is provided in the dropdown for the API documentation below.

To manually generate a [basic infrastructure](basic.md) using the individual APIs, follow the steps below:

1. Start the Tickerplant process by defining the `#!python trade` and `#!python aggregate` tables:

    ```python
    import pykx as kx
    trade = kx.schema.builder({
        'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        'price': kx.FloatAtom, 'volume': kx.LongAtom})
    agg = kx.schema.builder({
        'time': kx.TimespanAtom, 'sym': kx.SymbolAtom,
        'max_price': kx.FloatAtom , 'median_volume': kx.FloatAtom})

    tick = kx.tick.TICK(
        port=5010,
        tables = {'trade': trade, 'aggregate': agg},
        log_directory = 'log'
        )
    tick.start()
    ```

2. Next, generate the Historical Database process on port 5012 by loading historical data (if it exists) from a database at `#!python /tmp/db`. The RDB will connect to this process on initialization and trigger end-of-day operations:

    ```python
    hdb = kx.tick.HDB(port=5012)
    hdb.start(database='db')
    ```

3. Now that you have initialized the tickerplant and HDB, start the RDB process on port 5011. Connect to the tickerplant on port 5010 as follows:

    ```python
    rdb = kx.tick.RTP(port=5011)
    rdb.start({
        'tickerplant': 'localhost:5010',
        'hdb': 'localhost:5012',
        'database': 'db'})
    ```

This workflow is equivalent to the [basic infrastructure](basic.md) walkthrough.

??? "API documentation"
    Links to the functions used in the above section:

    - [`kx.tick.TICK`](../../../api/tick.md#pykx.tick.TICK)
    - [`tick.start`](../../../api/tick.md#pykx.tick.TICK.start)
    - [`kx.tick.RTP`](../../../api/tick.md#pykx.tick.RTP)
    - [`rtp.start`](../../../api/tick.md#pykx.tick.RTP.start)
    - [`kx.tick.HDB`](../../../api/tick.md#pykx.tick.HDB)
    - [`hdb.start`](../../../api/tick.md#pykx.tick.HDB.start)

## Process logs

Each of the process types covered within the documentation for Real-Time Data Capture is a sub-process which runs a separate executable to the Python process which initialized it. The benefit is in allowing to build complex workflows from a single Python process. However, it can make lifecycle management and tracking of these processes difficult.

By default, the initialization of `#!python TICK`, `#!python RTP`, `#!python HDB` and `#!python GATEWAY` processes prints information from `#!python stdout` and `#!python stderr` to the parent process which started the sub-processes. While this is useful in providing a user with up-to-date information about these processes, it makes separating logs from different processes difficult.

Each process type supports a keyword argument `#!python process_logs` which can have the following input types:

| **Input type** | **Description**                                                                |
|:-----------|:---------------------------------------------------------------------------|
| `#!python True`     | Logs should be printed to `#!python stdout`/`#!python stderr` of the parent Python process |
| `#!python False`    | Logs from the child process are suppressed and redirected to `#!python /dev/null`   |
| string     | Logs are redirected to the file location specified by the `#!python str`        |

1. Here's an example of redirecting logs to a file:

    - Define a query API which prints timing information relating to the query execution.
    - Register this query API to an `#!python RTP` process which logs data to a file `#!python process_logs.txt`.
    - Call the query API with a function which sleeps for 5 seconds and read the content of `#!python process_logs.txt`.
    - Define the query API, using [`#!python datetime`](https://docs.python.org/3/library/datetime.html) to time the query.

    ```python
    def time_api(query, *parameters):
        init_time = datetime.datetime.now()
        result = kx.q(query, *parameters)
        print(f'query time: {datetime.datetime.now() - init_time}')
        return result
    ```

2. Create your RTP process logging output to `#!python process_logs.txt` ensuring access to:

    ```python
    rtp = kx.tick.RTP(
        port=5011,
        libraries={'datetime': 'datetime', 'kx': 'pykx'},
        process_logs='process_logs.txt',
        apis={'time_api': time_api}
        )
    ```

3. Call the query API and read the content of `#!python process_logs.txt`. Note that to call this API you do not need to `#!python start` the process as we are not attempting to connect to the Tickerplant/HDB processes:

    ```python
    rtp('time_api', b'{system"sleep 5";x+10}', 10)
    with open('process_logs.txt') as f:
        print(f.read())
    ```

## How to stop processes

While we hope that we will always generate the perfect code, there can be times when being able to stop processing of our system is a requirement. As the streaming infrastructure for PyKX operates by starting sub-processes from Python, the control of these processes is more complex than it would be, should the parent process be in full control.

For each of the `#!python BASIC`, `#!python TICK`, `#!python RTP`, `#!python HDB` and `#!python GATEWAY` classes, the initialized class objects have an associated `#!python stop` function. Call this function if you want to gracefully shut down processing and kill the underlying process. You can invoke it using the `#!python rtp` process started in the previous section as an example:

```python
rtp.stop()
```

While graceful process closure is always advised, it may not always be possible. In case your parent process has been shut down and you no longer have access to the `#!python <process_name>.stop()` functionality, use `#!python kx.util.kill_q_process`. This takes the port number that your sub-process was started on and kills it. Caution should be taken when invoking this function.

```python
kx.util.kill_q_process(5010)
```

??? "API documentation"
    Links to the functions used in this section:

    - [`kx.tick.BASIC`](../../../api/tick.md#pykx.tick.BASIC)
    - [`kx.tick.TICK`](../../../api/tick.md#pykx.tick.TICK)
    - [`kx.tick.RTP`](../../../api/tick.md#pykx.tick.RTP)
    - [`kx.tick.HDB`](../../../api/tick.md#pykx.tick.HDB)
    - [`kx.tick.GATEWAY`](../../../api/tick.md#pykx.tick.GATEWAY)
    - [`kx.util.kill_q_process`](../../../api/util.md#pykxutildebug_environment)

## Next steps

Now that you have your basic infrastructure up and running you might be interested in some of the following:

- Learn how to publish data to your streaming infrastructure [here](publish.md).
- Learn how to subscribe to data from your streaming infrastructure [here](subscribe.md).

For some further reading, here are some related topics:

- Learn how to generate a Historical Database [here](../database/index.md).

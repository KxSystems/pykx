---
title: Example: Real-Time Streaming
description: The development of a basic streaming workflow using PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, streaming, basic
---

# Example: Real-Time Streaming using PyKX

_This page outlines the steps taken and functionality shown in demonstrating your first PyKX streaming application_

To run this example please download the [zip](./real-time-pykx.zip) file containing the notebook or visit our github repository [here](https://github.com/KxSystems/pykx/tree/main/docs/examples/streaming) to view the code directly.

In this example we will generate a real-time and historical analysis system which completes the following actions:

1. Allows ingestion of high-volume trade and quote financial data
2. Persists this data at end of day to a historical database.
3. Develop a real-time analytic which combines data from two independent real-time tables
4. Develop a number of query analytics on the historical database and real-time database which provide the count of the number of trades/quotes for a specified ticker symbol.
5. Generate a username/password protected gateway process which a user can query to combine the results from the real-time and historical data view into one value.

Each of the analytics provided in steps 3, 4 and 5 are Python analytics operating on data in kdb+/PyKX format.

## Want more information?

The documentation surrounding real-time streaming with PyKX is extensively outlined [here](../../user-guide/advanced/streaming/index.md). For information on specific parts of the infrastructures that can be generated you might find the following links useful:

| Title                                                                    | Description                                                                                                                         |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Start basic ingest](../../user-guide/advanced/streaming/basic.md)       | Learn how to start the core components of a basic streaming ingest framework.                                                       |
| [Publish data](../../user-guide/advanced/streaming/publish.md)           | Learn how to publish data to your real-time capture system using Python, q and C.                                                   |
| [Subscribe to data](../../user-guide/advanced/streaming/subscribe.md)    | How do you subscribe to new updates being received in your system?                                                                  |
| [Real-Time Analytics](../../user-guide/advanced/streaming/rta.md)        | Generate insights into your real-time data and account for common problems.                                                         |
| [Custom query APIs](../../user-guide/advanced/streaming/custom_apis.md)   | Learn how to querying historical and real-time data using custom Python APIs.                                                       |
| [Query access gateways](../../user-guide/advanced/streaming/gateways.md) | Learn how to create a query API which traverses multiple processes and can limit user access to only information they need to know. |

If you want to read through the API documentation for this functionality it is contained in it's entirety [here](../../api/tick.md).
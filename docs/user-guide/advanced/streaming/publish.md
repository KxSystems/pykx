---
title: Publish Data
description: How to publish data to your streaming infrastructure
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, streaming, publishing
---

# Publish Data

_This page outlines how you can publish new data to your streaming infrastructure._

!!! Warning "Disclaimer"

         The functionality outlined below provides the necessary tools for users to build complex streaming infrastructures. The generation and management of such workflows rest solely with the users. KX supports only individual elements used to create these workflows, not the end-to-end applications.

Publishing data to a PyKX streaming workflow is completed by publishing messages to a [tickerplant process](basic.md#tickerplant) using [Interprocess Communication (IPC)](../ipc.md). The sections below show how to achieve this with Python and q in a [basic streaming infrastructure](basic.md). Commonly in KX literature and whitepapers, processes which publish data to a tickerplant are described as Feedhandlers.

Any messages that is published to a tickerplant is a triplet list with the following structure `#!python [Function;Table;Data]`, where:

- Function: The name of the function to be called on all downstream subscribers. In this case, it's `#!python .u.upd`. This function takes two arguments: table and data.
- Table: The name of the table to be passed as the first argument to the Function above.
- Data: The data which is to be passed as the second argument to the Function above.

## Basic examples

The below sections provide examples in Python and q showing the publishing of 10 messages to the `#!python trade` table defined in the basic infrastructure [here](basic.md#get-started). Data will be randomly generated in each case.

!!! Note

	In each example, supply of the `timespan` object is optional. If omitted, data will be tagged with arrival time and persisted by the database using this time information.

In a later section of this page, we provide a more complex data feed which you can use to emulate a data feed from Python which can be used in the remaining pages relating to streaming data.

### Python

The following Python code allows you to publish 10 messages to the streaming infrastructure created [here](basic.md):

```python
import pykx as kx
import numpy as np

ticker_list = ['AAPL', 'GOOG', 'IBM', 'BRK']

for i in range(1, 10):
    with kx.SyncQConnection(port=5010, wait=False) as q:
        msg = [kx.TimespanAtom('now'),
               np.random.choice(ticker_list),
               np.random.random(10) * 10 * i,
               np.random.randint(100) * i]
        q('.u.upd', 'trade', msg)
```

In the above code we create a Synchronous Connection against the Tickerplant process on port 5010, sending messages with no expectation of a response denoted through setting `#!python wait=False`. We create a message (`#!python msg`) containing 4 elements:

1. The current time as a `#!python kx.TimespanAtom` type object 
1. Name of the trade symbol (`#!python ticker`) randomly generated from a pre-determined list
1. The price of the stock randomly generated
1. The volume of the stock that was traded.

Finally, this message is sent to the tickerplant alongside the name of the table `#!python trade` and the function which is to be called `#!python .u.upd`.

### q

The following q code allows you to publish 10 messages to the streaming infrastructure created [here](basic.md):

```q
h:hopen 5010

// Function for sending updates to trade table
upd_trades:{neg[x](".u.upd";y;z)}[h;`trade]

// Function for generating a sample message
msg:{
  (.z.N;
   rand `AAPL`GOOG`IBM`BRK;
   x*rand 10.0;
   x*rand 100)
  }

// Send 10 messages using the values 1-10 to update the price/volume values
(upd_trades msg@)each 1+til 10
```

In the above code we open a connection to the Tickerplant process on port 5010. Sending 10 messages created using the function `msg` and `upd_trades`. The message generated contains 4 elements:

1. The current time generated using `#!python .z.N`
1. Name of the trade symbol (`#!python ticker`) randomly generated from a pre-determined list
1. The price of the stock randomly generated
1. The volume of the stock that was traded.

### Other languages

It's possible to publish data to PyKX streaming infrastructures using other languages, such as C and Java:

- [Publishing to kdb+ using Java](https://www.timestored.com/kdb-guides/kdb-java-api#feedhandling)
- [Publishing to kdb+ tickerplant using C](https://code.kx.com/q/wp/capi/#publishing-to-a-kdb-tickerplant)

## Continuous streaming example

In the below section we generate a script which completes the following:

1. Takes a parameter at startup which indicates how many messages should be published per update.
1. Generates a random trade message using `#!python kx.random.random`.
1. Publishes this message to the [basic infrastructure](basic.md) tickerplant on port 5010.
1. Repeats until a user stops the processing data feed.

You can view this script below or [download](scripts/feed.py) and run it following the instructions outlined below.

```python
import pykx as kx

import sys

try:
    args = sys.argv[1]
except BaseException:
    args=''
n = 1 if args=='' else int(args)

print('Starting Data Feed ...')
init = False

def main():
    global init
    symlist = ['AAPL', 'JPM', 'GOOG', 'BRK', 'WPO', 'IBM']
    while True:
        trade = [kx.random.random(n, symlist),
                 10 * kx.random.random(n, 10.0),
                 10 * kx.random.random(n, 100)
                 ]
        with kx.SyncQConnection(port=5010, wait=False, no_ctx=True) as q:
            q('.u.upd', 'trade', trade)
        if not init:
            print('First message(s) sent, data-feed publishing ...')
            init=True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Data feed stopped')
```

Before you start, ensure you have the basic infrastructure running with default values. To use the above `#!python feed.py` script, run it as follows:

- Publish one message per update

	```bash
	python feed.py
	```

- Publish ten messages per update

	```bash
	python feed.py 10
	```

## Next steps

Now that you have data being published to your system you may be interested in the following:

- Subscribe to real-time updates following the instructions [here](subscribe.md).
- Query your real-time and historical data using custom APIs [here](custom_apis.md).
- Perform complex analysis on your real-time data following the instructions [here](rta.md).

For some further reading, here are some related topics:

- Learn more about Interprocess Communication (IPC) [here](../ipc.md).

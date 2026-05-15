---
title:  KX for Databricks cluster setups
description: How to use KX for Databricks
date: September 2024
author: KX Systems, Inc.,
tags: PyKX, Databricks, single node, multi node, workflows
---

# Cluster setups

_The purpose of this page is to help you set up your single node or multi node clusters._

You can run KX for Databricks on both single node and multi-node computes. Choose the compute and RAM requirements depending on your data/pipeline requirements.

For the best experience, read the [KX for Databricks Overview](introduction.md) and the [Get started](getStarted.md) guide first.

Now let's explore examples for two types of cluster setups:

1. [Single node workflow](#1-single-node-workflow)
2. [Multi node workflow](#2-multi-node-workflow)

!!! info "Pre-requisites"

    The following pre-requisites must be satisfied in order to run the below notebooks:

    - Configuration of single node or multi node compute
    - Latest version of PyKX must be installed
    - QLIC env variable must point to a valid kc.lic file
    - Load Trade, Quote & Execution data from a Databricks Unity Catalogue.
    
    For more information refer to the [Get Started documentation](getStarted.md).

## 1. Single node workflow

A single node compute runs spark locally and the driver works as both master and worker, with no worker nodes.
It is intended for jobs that use small amounts of data or non-distributed workloads such as single-node machine learning libraries.

Import the required Python libraries:

```python
import pykx as kx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
```

### Hourly OHLC - Pythonic vs. q magic

!!! note "OHLC is the open, high, low, and close price for a given period."

The example below calculates the hourly OHLC in two ways: using regular Python syntax and then the q magic.

!!! example ""

	=== "Pythonic"

        Calculate the OHLC using regular Python syntax:

        ```python
        kx.q("{select Open:first price,High:max price,Low:min price,Close:last price by bucket:0D01 xbar time,sym from x}", trade).head()
        ```

        This should produce the following output:

        ```
        bucket                          sym     Open    High    Low     Close
        2024.07.04D04:00:00.000000000   AEF     617f    617f    617f    617f
                                        BKR     93.21   93.21   93.21   93.21
                                        CLMT    2327.99 2327.99 2327.99 2327.99
                                        CMG     780f    780f    777.6   780f
                                        F.PRA   333.99  336.79  332.6   336.69
        ```

    === "q magic"

        Calculate the OHLC using q magic:

        ```python
        %%q
        select Open:first price,High:max price,Low:min price,Close:last price by bucket:0D01 xbar time,sym from trade
        ```

        This should produce the following output:

        ```python
        bucket                        sym   | Open    High    Low     Close  
        ------------------------------------| -------------------------------
        2024.07.04D04:00:00.000000000 AEF   | 617     617     617     617    
        2024.07.04D04:00:00.000000000 BKR   | 93.21   93.21   93.21   93.21  
        2024.07.04D04:00:00.000000000 CLMT  | 2327.99 2327.99 2327.99 2327.99
        2024.07.04D04:00:00.000000000 CMG   | 780     780     777.6   780    
        2024.07.04D04:00:00.000000000 F.PRA | 333.99  336.79  332.6   336.69 
        ```

Notice how both examples produce the same result.

### VWAP

!!! note "VWAP - The average price a security has traded at throughout the day, based on both volume and price."

The example below calculates the volume-weighted average price (VWAP) using q magic:

```python
%%q 
select vwap:size wavg price by sym from trade
```

??? example "Output"

    ```
    sym   | vwap    
    ------| --------
    AEF   | 625.727 
    AGND  | 228.8684
    AUY   | 251.6101
    AXJL  | 494.2899
    AXX   | 326.6949
    ```

### Volatility

!!! note "Volatility - A statistical measure of the dispersion of returns for a given security or market index."

The example below calculates volatility for the stock `AEF`.

Define the functions used to calculate the volatility:

```python
get_volatility_pykx = kx.q("{[bid;ask] {r:0^log[x]-log next x;sqrt ema[0.001] r*r } 0.5*ask+bid}")

def get_volatility(quote_tbl, sym) -> dict:
    subset_quotes = quote_tbl.loc[
        (quote_tbl["sym"] == sym)
        & (quote_tbl["time"] >= timedelta(hours=9, minutes=25))
        & (quote_tbl["time"] < timedelta(hours=16, minutes=0))
    ]
    return {
        "datetime": subset_quotes["time"],
        "volatility": get_volatility_pykx(subset_quotes["bid"], subset_quotes["ask"])
    }
```

Pass the `quote` table and the `AEF` symbol to the `get_volatility` function:

```python
get_volatility(quote, 'AEF')
```

??? example "Output"

    ```python
    {'datetime': pykx.TimestampVector(pykx.q('2024.07.04D09:25:03.491000000 2024.07.04D09:25:46.166000000 2024.07.04D09:27:..')),
    'volatility': pykx.FloatVector(pykx.q('0.003473635 0.003496871 0.003534634 0.003532866 0.003612577 0.003686721 0.003..'))}
    ```

### Spread

!!! note "Spread - The difference or gap that exists between two prices, rates, or yields."

The example below calculates the moving average over the spread for the stock `AEF`.

Define the functions used to calculate the moving average over the spread:

```python
def get_mavg_of_spread(bid, ask, mavg_size: int):
    spreads = ask - bid
    return np.convolve(spreads, np.ones(mavg_size), mode="same") / mavg_size

kx.q["mavg_of_spread"] = get_mavg_of_spread

def get_spread(quotes: kx.Table, sym: str) -> kx.Table:
    return kx.q.qsql.select(
        quotes,
        columns={"datetime": "time", "spread": "mavg_of_spread[bid; ask; 1000]", "bid":"bid", "ask":"ask"},
        where=[f'sym=`$"{sym}"', "bid>0", "ask>0", "time>=09:25", "time <16:00"]
    )
```

Pass the `quote` table and the `AEF` symbol to the `get_spread` function:

```python
get_spread(quote,'AEF').head()
```

??? example "Output"

    ```python
        datetime                        spread      bid     ask
    0   2024.07.04D09:25:03.491000000   15.07707    580.67  653.61
    1   2024.07.04D09:25:46.166000000   15.07735    600f    630f
    2   2024.07.04D09:27:01.015000000   15.07766    588.88  625f
    3   2024.07.04D09:27:10.556000000   15.07844    580.67  653.61
    4   2024.07.04D09:27:10.557000000   15.07873    580.67  653.61
    ```

### Slippage

!!! note "Slippage - The difference between the expected price of a trade and the price at which the trade is executed."

The example below calculates the Slippage based on the midpoint between the prevailing quoted bid and ask prices.

Create the `merged` table by performing an as-of-join on the `execs` and `quote` tables:

```python
merged = kx.q.aj(kx.SymbolVector(["sym", "time"]), execs, quote)
```

Calculate the Mid,Delta and Slippage columns:

```python
merged["mid"] = 0.5 * (merged["bid"] + merged["ask"])
merged["diff"] = np.where(
    merged["side"] == "BUY", merged["mid"] - merged["price"], merged["price"] - merged["mid"]
)
merged["slippage"] = (merged["diff"] / merged["mid"]) * 10000

merged.head()
```

??? example "Output"

    ```python
        date        sym     time                            price       size    side    venue   bid     ask     bsize   asize   mode    ex  mid     diff    slippage
    0   2022.03.31  GE.PRA  2024.07.04D09:30:02.676000000   397.6131    12      BUY     venue1  396.51  397.96  1       1       R       N   397.235 -0.3781 -9.518295
    1   2022.03.31  MQC     2024.07.04D09:30:02.851000000   254.8359    258     SELL    venue3  254.32  254.99  2       1       R       V   254.655 0.1809  7.103729
    2   2022.03.31  BKR     2024.07.04D09:30:03.426000000   93.64192    154     BUY     venue2  93.64   93.72   1       1       R       N   93.68   0.03808 4.064902
    3   2022.03.31  GE.PRA  2024.07.04D09:30:04.489000000   397.5469    12      SELL    venue2  397.21  398.8   1       2       R       T   398.005 -0.4581 -11.50991
    4   2022.03.31  DXGE    2024.07.04D09:30:06.677000000   570.1041    60      SELL    venue3  569.08  571.34  2       3       R       P   570.21  -0.1059 -1.857211
    ```

## 2. Multi node workflow

A multi node (distributed) compute consists of one driver node and one or more worker nodes.

The driver node maintains state information of all notebooks attached to the compute, maintains the SparkContext, interprets all the commands you run from a notebook or a library on the compute, and runs the Apache Spark master that coordinates with the Spark executors.
Worker nodes run the Spark executors and other services required for proper functioning compute.

Use multi-node compute for larger jobs with distributed workloads.

Before you start, make sure you meet the pre-requisites at the top of this page.

### Import required libraries

Import a PySpark and Python libraries by running:

```python
import pykx as kx
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, DateType, IntegerType
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)
import copy as cp
```

### Prepare the data

!!! note "The below examples will make use of the full dataset while using PySpark and a subset of the data while using the Python Driver."

Create `trade_sdf_all_syms` and `quote_sdf_all_syms` Spark Dataframes containing trade and quote data respectively using Spark SQL:

```python
trade_sdf_all_syms = spark.sql('SELECT TTime as time, Symbol as sym, cast(Trade_Volume AS INT) as size, Trade_Price as price FROM default.nyse_trades')
quote_sdf_all_syms = spark.sql('SELECT TTime as time, Symbol as sym, Bid_Size as bsize, Bid_Price as bid, Offer_Size as asize, Offer_Price as ask FROM default.nyse_quotes')
```

Filter the data to the a subset of the tickers:

```python
SYMBOLS = ['GOOG', 'AAPL', 'FB', 'AMZN', 'MSFT']  
trade_sdf_few_syms = trade_sdf_all_syms.where(F.col("Symbol").isin(SYMBOLS)).orderBy(["Symbol","TTime"])
quote_sdf_few_syms = quote_sdf_all_syms.where(F.col("Symbol").isin(SYMBOLS)).orderBy(["Symbol","TTime"])
```

Convert the Spark Dataframes to Pandas Dataframes:

```python
trade_pdf = trade_sdf_few_syms.toPandas()
quote_pdf = quote_sdf_few_syms.toPandas()
```

### Slippage markouts (Python Driver/Single Node)

!!! note "Slippage - The difference between the expected price of a trade and the price at which the trade is executed."

In the below example, we will use PyKX to calculate the execution slippage markouts for a subset of the data (5 tickers) using just the Python Driver (Single node).

??? example "Define the required functions used to calculate the slippage markouts"

    ```python
    # adjust the time column in the past/future by offset e.g 10 20 30 seconds, 1 5 10 minutes
    def time_offset(trade, offset):
        offset_trade = cp.copy(trade)
        offset_trade['time'] = (offset_trade['time'] + offset).sorted()
        return offset_trade[['sym', 'time']]

    # compute N markouts and calculate slippage using mid price
    def trade_to_mid_markouts(trade, quote):
        import os
        os.environ['QLIC'] = '/dbfs/tmp/'
        import pykx as kx

        trade = kx.toq(trade)
        quote = kx.toq(quote)

        trade = trade.sort_values(['time']).grouped('sym')
        quote = quote.sort_values(['time']).grouped('sym')
        quote['mid'] = (quote['ask'] + quote['bid']) / 2

        # markouts we are interested in
        seconds = [1, 10, 30]
        minutes = [1, 5, 10, 30]

        qseconds = kx.toq(seconds).cast(kx.SecondVector)
        qminutes = kx.toq(minutes).cast(kx.MinuteVector)

        for s, q in zip(seconds, qseconds):
            trade['tp'+str(s)+'s'] = trade['price'] - time_offset(trade, q).merge_asof(quote[['sym', 'time', 'mid']], on=['sym', 'time'])['mid']
            trade['tm'+str(s)+'s'] = trade['price'] - time_offset(trade, -q).merge_asof(quote[['sym', 'time', 'mid']], on=['sym', 'time'])['mid']

        for s, q in zip(minutes, qminutes):
            trade['tp'+str(s)+'m'] = trade['price'] - time_offset(trade, q).merge_asof(quote[['sym', 'time', 'mid']], on=['sym', 'time'])['mid']
            trade['tm'+str(s)+'m'] = trade['price'] - time_offset(trade, q).merge_asof(quote[['sym', 'time', 'mid']], on=['sym', 'time'])['mid']

        return trade.pd()
    ```

Pass the `trade_pdf` and `quote_pdf` Pandas Dataframes to the `trade_to_mid_markouts` function in order to calculate the slippage:

```python
result = trade_to_mid_markouts(trade_pdf, quote_pdf)

print('Executions input row count - ', f'{trade_pdf.shape[0]:,}')
print('Market quotes input row count - ', f'{quote_pdf.shape[0]:,}')
print('TCA report result row count - ', f'{kx.q.count(result).np():,}')

result.tail()
```

!!! note "Since we have supplied a Pandas dataframe, the workload will be performed by only the Python driver."

??? example "Output"

    ```python
    Executions input row count -  1,851,406
    Market quotes input row count -  27,424,147
    TCA report result row count -  1,851,406

            time                        sym	    size    price   tp1s    tm1s    tp10s   tm10s   tp30s   tm30s   tp1m    tm1m    tp5m    tm5m    tp10m   tm10m   tp30m   tm30m
    1851401	2022-10-03 19:59:55.559408  AAPL    1       142.935	0.000   0.000   -0.025  0.010   -0.025  0.045   -0.025  -0.025  -0.025  -0.025  -0.025  -0.025  -0.025  -0.025
    1851402	2022-10-03 19:59:58.444119  MSFT    25      241.700	0.185   0.185   0.185   0.185   0.185   3.200   0.185   0.185   0.185   0.185   0.185   0.185   0.185   0.185
    1851403	2022-10-03 19:59:59.146947  AAPL    1       142.935	-0.025  0.000   -0.025  0.010   -0.025  0.055   -0.025  -0.025  -0.025  -0.025  -0.025  -0.025  -0.025  -0.025
    1851404	2022-10-03 19:59:59.551589  MSFT    1       241.700	0.185   0.185   0.185   0.185   0.185   3.200   0.185   0.185   0.185   0.185   0.185   0.185   0.185   0.185
    1851405	2022-10-03 19:59:59.668143  MSFT    150     241.700	0.185   0.185   0.185   0.185   0.185   3.200   0.185   0.185   0.185   0.185   0.185   0.185   0.185   0.1815
    ```

### Slippage markouts (PySpark/Multi node)

In the below example, we will use PyKX to calculate the execution slippage markouts for all of the data (11686 tickers) using PySpark (Multi node).

Prepare the schema for the results dataframe:

```python
full_schema = "time timestamp, sym string,  size int, price double"

seconds = [1, 10, 30]
minutes = [1, 5, 10, 30]

for s in seconds:
        full_schema += ', tp'+str(s)+'s float'
        full_schema += ', tm'+str(s)+'s float'
for m in minutes:
        full_schema += ', tp'+str(m)+'m float'
        full_schema += ', tm'+str(m)+'m float'
```

Pass the `trade_sdf_all_syms` and `quote_sdf_all_syms` Spark Dataframes to the `trade_to_mid_markouts` function in order to calculate the slippage:

```python
result_pyspark = trade_sdf_all_syms.groupby('sym').cogroup(quote_sdf_all_syms.groupby('sym')).applyInPandas(trade_to_mid_markouts, schema=full_schema)

print('Executions input row count - ', f'{trade_sdf_all_syms.count():,}')
print('Market quotes input row count - ', f'{quote_sdf_all_syms.count():,}')
print('TCA report result row count - ', f'{result_pyspark.count():,}')

result_pyspark.show(5)
```

!!! note "Since we have supplied a Spark dataframe, the workload will be farmed out to the configured worker nodes for processing before returning the result to the Python driver."

??? example "Output"

    ```python
    Executions input row count -  77,026,358
    Market quotes input row count -  1,541,781,424
    TCA report result row count -  77,026,358

    +--------------------+------+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+-----+-----+
    |                time|   sym|size|price| tp1s| tm1s|tp10s|tm10s|tp30s|tm30s| tp1m| tm1m| tp5m| tm5m| tp10m| tm10m|tp30m|tm30m|
    +--------------------+------+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+-----+-----+
    |2022-10-03 09:16:...|AKIC U|  85| 9.95| 0.21| 0.21| 0.21| 0.21| 0.21| 0.21| 0.21| 0.21| 0.17| 0.17|-0.265|-0.265|-0.03|-0.03|
    |2022-10-03 11:20:...|AKIC U|   9| 9.96| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0|  -1.0|  -1.0| -1.0| -1.0|
    |2022-10-03 11:56:...|AKIC U|  21| 9.96| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -1.0| -0.02| -0.02|-0.02|-0.02|
    |2022-10-03 13:36:...|AKIC U|  21| 9.95|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01| -0.01| -0.01|-0.01|-0.01|
    |2022-10-03 13:36:...|AKIC U|  21| 9.95|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01|-0.01| -0.01| -0.01|-0.01|-0.01|
    +--------------------+------+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+-----+-----+
    ```
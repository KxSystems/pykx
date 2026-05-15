---
title:  Get started with KX for Databricks 
description: How to install and use KX for Databricks
date: September 2024
author: KX Systems, Inc.,
tags: PyKX, setup, install,
---

# Get started with KX for Databricks 

_This page explains how to deploy and get started with KX for Databricks._

1. [Install PyKX](#1-install-pykx)
2. [Licensing](#2-licensing)
3. [Load data using Spark Dataframes](#3-load-data-using-spark-dataframes)
4. [PyKX pythonic vs. q magic (%%)](#4-pykx-pythonic-vs-q-magic-)


## 1. Install PyKX

Install PyKX on the cluster using one of the following methods:

!!! Note ""

	=== "Install PyKX from PyPI"

        Create a new notebook and install using `pip`:

        ```python
        pip install pykx
        ```

    === "Install PyKX from Anaconda"

        Create a new notebook and install using `conda`:

        ```python
        conda install -c kx pykx
        ```

    === "Install PyKX from the Python Wheel"

        Download the Python Wheel file from the [Download Portal](https://portal.dl.kx.com/assets/pypi/pykx) and store it on the cluster:

        ```shell
        > ls /path/to/folder
        pykx-2.5.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        ```

        Create a new notebook and install from the wheel file:

        ```python
        pip install /path/to/folder/pykx-2.5.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        ```
!!! tip "Tip: After installing the packages, you may need to restart the kernel by running the following command:"
    
    
    ```python
    dbutils.library.restartPython()
    ```

## 2. Licensing

To use all PyKX functionalities, you need to download and install a KDB Insights license. 

1. Choose a license type:

    !!! Note ""

        === "Personal License"

            For personal usage, navigate to [Personal License Download](https://kx.com/kdb-insights-personal-edition-license-download/) and complete the form.

        === "Commercial License"

            For commercial usage, contact your KX sales representative, email [sales@kx.com](mailto:sales@kx.com), or apply through [Book demo](https://kx.com/book-demo).


    !!! info "More on the [license installation process](https://code.kx.com/pykx/2.5/getting-started/installing.html#2-install-a-kdb-insights-license)."

2. On receipt of an email from KX, download and save the `kc.lic` license file to a secure location on your cluster:

    ```shell
    > ls /path/to/folder
    kc.lic
    ```

3. Set an environment variable `QLIC` pointing to the folder containing the license file:

    ```python
    import os
    os.environ['QLIC'] = '/path/to/folder'
    ```
    That's it! To verify if you successfully installed PyKX, run the following:

    ```python
    import pykx as kx
    print('PyKX Version: ', kx.__version__)
    ```
    You should have the following output:

    ```python
    PyKX Version:  2.5.1
    ```
    !!! note "Note: The license requires internet connectivity."

        Update the workspace security policies to whitelist the following:

        - The remote servers should treat kdb+ on-demand as a CDN-fronted service and employ DNS-based filtering. 
        - The current list of hostnames required for kdb+ on-demand operation is {h,g}.kdbod.{com,net,org}.

## 3. Load data using Spark Dataframes

**a. From the Databricks Unity Catalogue:**

```python
dfTrades = spark.table('<name of catalogue>.<name of schema>.<name of table>')
dfQuotes = spark.table('<name of catalogue>.<name of schema>.<name of table>')
```

**b. Directly from a CSV file:**

```python
dfTrades = spark.read.format("csv").option("header", "true").option("delimiter", ",").load("/path/to/file")
dfQuotes = spark.read.format("csv").option("header", "true").option("delimiter", ",").load("/path/to/file")
```

To convert the Spark Dataframe to a Pandas Dataframe, use the `toPandas()` function:

```python
pandas_trade = dfTrades.toPandas()
pandas_quote = dfQuotes.toPandas()
```

Convert the Pandas Dataframe to a q table with the `toq()` function:

```python
pykx_trades = kx.toq(pandas_trade)
pykx_quotes = kx.toq(pandas_quote)
```

## 4. PyKX pythonic vs. q magic (%%)

Reference q functions in PyKX with the standard python syntax:

```python
kx.q('til 10')
```

This produces a sequence of numbers from 0 to 9:

```python
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
```

Another way to achieve this is by using native q by leveraging q magic (`%%q`) :

```python
%%q
til 10
```

You will get the same output:

```python
0 1 2 3 4 5 6 7 8 9
```

??? example "Simple load and query example using PyKX"

    Consider a `samples` catalogue with a `finance` schema that has a `trades` table. Load data from the table into a Spark Dataframe `dfTrade`:

    ```python
    dfTrade = spark.table('samples.finance.trades')
    ```

    Next, convert the Spark Dataframe to a Pandas Dataframe `panda_trade` using the `toPandas()` function:

    ```python
    panda_trade = dfTrade.toPandas()
    ```

    Using the `toq()` function to convert the Pandas Dataframe to a q table `pykx_trade`:

    ```python
    pykx_trade = kx.toq(panda_trade)
    ```

    To perform a simple OHLC query on the data, use regular python syntax:

    ```python
    kx.q("{select Open:first price,High:max price,Low:min price,Close:last price by bucket:0D01 xbar time,sym from x}", pykx_trade).head()
    ```

    This should produce the following output:

    ```python
    bucket                          sym Open        High        Low         Close
    2017.08.31D00:00:00.000000000   A   347.8271    349.256     346.0935    348.2853
                                    AAN	347.4742    349.1463    346.2195    348.3558
                                    ACA	347.4005    348.4313    346.5526    347.2568
                                    ACP	346.8897    350.9313    345.8342    347.8243
    ```

    You can achieve the same result with q magic by storing the table:

    ```python
    kx.q['pykx_trade'] = pykx_trade
    ```

    and performing the same query:

    ```python
    %%q
    select Open:first price,High:max price,Low:min price,Close:last price by bucket:0D01 xbar time,sym from pykx_trade
    ```

    In the end, you get the same output:

    ```python
    bucket                        sym | Open     High     Low      Close   
    ----------------------------------| -----------------------------------
    2017.08.31D00:00:00.000000000 A   | 347.8271 349.256  346.0935 348.2853
    2017.08.31D00:00:00.000000000 AAN | 347.4742 349.1463 346.2195 348.3558
    2017.08.31D00:00:00.000000000 ACA | 347.4005 348.4313 346.5526 347.2568
    2017.08.31D00:00:00.000000000 ACP | 346.8897 350.9313 345.8342 347.8243
    2017.08.31D00:00:00.000000000 ACRE| 349.8609 351.621  345.0562 345.9471
    ```
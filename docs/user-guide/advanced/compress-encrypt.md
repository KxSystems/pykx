---
title:  PyKX compress and encrypt
description: How to compress and encrypt data in PyKX
date: October 2024
author: KX Systems, Inc.,
tags: compression, encryption, PyKX
---

# Compress and encrypt data
_This page explains how to compress and encrypt data in PyKX._

With the volumes of sensitive data being produced within real-time applications today the ability to securely store this data and quickly access it can be challenging. PyKX provides several utilities, in the form of class objects, for the management of how data is compressed and encrypted when being persisted.

### Compress

PyKX supports the compression of data to disk, allowing you to reduce disk space required for your persisted historical data. PyKX gives you a variety of compression/decompression options through the following algorithms:

- [`#!python gzip`](https://en.wikipedia.org/wiki/Gzip)
- [`#!python snappy`](https://en.wikipedia.org/wiki/Snappy_(compression))
- [`#!python zstd`](https://en.wikipedia.org/wiki/Zstd)
- [`#!python LZ4HC`](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm))

In addition to this, you can compress data to KX's own qIPC format. For full information, go to [KX file compression within kdb+/q](https://code.kx.com/q/kb/file-compression/).

### Encrypt

PyKX supports Data At Rest Encryption (DARE) with an explicit requirement on at least OpenSSL v1.0.2. To find out which version of OpenSSL is available to you via PyKX, use the following:

```python
>>> import pykx as kx
>>> kx.ssl_info()
pykx.Dictionary(pykx.q('
SSLEAY_VERSION   | OpenSSL 1.1.1q  5 Jul 2022
SSL_CERT_FILE    | /usr/local/anaconda3/ssl/server-crt.pem
SSL_CA_CERT_FILE | /usr/local/anaconda3/ssl/cacert.pem
SSL_CA_CERT_PATH | /usr/local/anaconda3/ssl
SSL_KEY_FILE     | /usr/local/anaconda3/ssl/server-key.pem
SSL_CIPHER_LIST  | ECDBS-ECASD-CHACHA94-REAL305:ECDHE-RSM-CHACHA20-OOTH1305:..
SSL_VERIFY_CLIENT| NO
SSL_VERIFY_SERVER| YES
'))
```

The encryption provided by this functionality is Transparent Disk Encryption (TDE). TDE protects data at rest by encrypting database files on the hard drive and as a result on backup media. Encrypting your data with PyKX is fully transparent to queries requiring no change to the logic used when querying data but results in a time penalty.

To use this functionality, you must have a password-protected master key available, ideally with a unique password of high-entropy. For more information on the generation of a master key and a password, go to the [DARE configuration](https://code.kx.com/q/kb/dare/#configuration) section.

## Functional walkthrough

This walkthrough demonstrates the following steps:

- Create a compression object for global and per-partition data persistence.
- Persist a variety of database partitions setting various compression configurations.
- Set the Python session to have globally configured encryption and compression settings.

### Generate compression objects

With PyKX, you can create compression and encryption class objects to set global configurations or use in specific individual functions. These respectively are supported via the `#!python kx.Compress` and `#!python kx.Encrypt` classes. For this section we will deal only with compression.

The full list of algorithms is part of the `#!python kx.CompressionAlgorithm` enumeration: 

```python
>>> import pykx as kx
>>> list(kx.CompressionAlgorithm)
[<CompressionAlgorithm.none: 0>, <CompressionAlgorithm.ipc: 1>, <CompressionAlgorithm.gzip: 2>, <CompressionAlgorithm.snappy: 3>, <CompressionAlgorithm.lz4hc: 4>, <CompressionAlgorithm.zstd: 5>]
```

You can further details through the `help` command:

```python
>>> help(kx.CompressionAlgorithm)
```

Once you are familiar with the options available to you, it's time to initialize your first compression class. In this case generating a compression object which uses the `#!python gzip` algorithm at compression `#!python level 8`.

```python
>>> import pykx as kx
>>> compress = kx.Compress(algo=kx.CompressionAlgorithm.gzip, level=8)
```

We use this object in the remaining sections of the walkthrough, in a local (one-shot) and global context.

### Persist database partitions with various configurations

Not all data is created equally, in time-series applications such as algorithmic trading it is often the case that older data is less valuable than newer data. As a result, when backfilling historical data, you may more aggressively compress older datasets. The PyKX compression logic allows you to persist different partitions within a historical database to different levels.

1. Create a database with the most recent data uncompressed

	```python
	>>> import pykx as kx
	>>> from datetime import date
	>>> N = 10000
	>>> db = kx.DB(path='/tmp/db')
	>>> qtable = kx.Table(
	...   data={
	...         'x': kx.random.random(N, 1.0),
	...         'x1': 5 * kx.random.random(N, 1.0),
	...         'x2': kx.random.random(N, ['a', 'b', 'c'])
	...     }
	... )
	>>> db.create(qtable, 'tab', date(2020, 1, 1))
	```

2. Add a new partition using `#!python gzip` compression

	```python
	>>> gzip = kx.Compress(algo=kx.CompressionAlgorithm.gzip, level=4)
	>>> qtable = kx.Table(
	...   data={
	...         'x': kx.random.random(N, 1.0),
	...         'x1': 5 * kx.random.random(N, 1.0),
	...         'x2': kx.random.random(N, ['a', 'b', 'c'])
	...     }
	... )
	>>> db.create(qtable, 'tab', date(2020, 1, 2), compress=gzip)
	```

3. Add a final partition using `#!python lz4hc` compression

	```python
	>>> lz4hc = kx.Compress(algo=kx.CompressionAlgorithm.lz4hc, level=10)
	>>> qtable = kx.Table(
	...   data={
	...         'x': kx.random.random(N, 1.0),
	...         'x1': 5 * kx.random.random(N, 1.0),
	...         'x2': kx.random.random(N, ['a', 'b', 'c'])
	...     }
	... )
	>>> db.create(qtable, 'tab', date(2020, 1, 3), compress=lz4hc)
	```

Notice the information about the persistence characteristics of your data using `#!python kx.q('-21!')`, for example:

```python
>>> kx.q('-21!`:/tmp/db/2020.01.01/tab/x')
pykx.Dictionary(pykx.q(''))
>>> kx.q('-21!`:/tmp/db/2020.01.02/tab/x')
pykx.Dictionary(pykx.q('
compressedLength  | 5467
uncompressedLength| 8016
algorithm         | 2i
logicalBlockSize  | 17i
zipLevel          | 4i
'))
>>> kx.q('-21!`:/tmp/db/2020.01.03/tab/x')
pykx.Dictionary(pykx.q('
compressedLength  | 6374
uncompressedLength| 8016
algorithm         | 4i
logicalBlockSize  | 17i
zipLevel          | 10i
'))
```

### Initialize compression and encryption globally

Global initialization of compression and encryption allows all data that is persisted within, from a process, to be compressed. This can be useful when completing large batch operations on data where being specific about per partition/per file operations isn't necessary. In the below section we will deal with compression and encryption separately.

PyKX uses compression settings that are globally readable via `#!python kx.q.z.zd`. When unset, this value returns a PyKX Identity value as follows:

```python
>>> kx.q.z.zd
pykx.Identity(pykx.q('::'))
```

To set the `#!python gzip` globally, use the `#!python global_init` on the generated `#!python kx.Compress` object.

```python
>>> compress = kx.Compress(algo=kx.CompressionAlgorithm.gzip, level=9)
>>> compress.global_init()
>>> kx.q.z.z.d
pykx.LongVector(pykx.q('17 2 9'))
```

Complete the global encryption initialisation by loading of the users encryption key into the process as follows:

```python
>>> encrypt = kx.Encrypt(path='/path/to/my.key', password='PassWorD')
>>> encrypt.global_init()
```

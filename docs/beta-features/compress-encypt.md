# Compression and Encryption

!!! Warning

	This module is a Beta Feature and is subject to change. To enable this functionality for testing please follow the configuration instructions [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`

## Introduction

With the volumes of sensitive data being produced within real-time applications today the ability to securely store this data and the ability to quickly access it can be challenging. PyKX provides users with a number of utilities, in the form of class objects, for the management of how data is compressed and encrypted when being persisted.

### Compression

The compression of data to disk is supported via PyKX allowing you to reduce disk space required for your persisted historical data. PyKX provides a variety of compression options allowing users to compress/decompress data using the following algorithms:

- [`gzip`](https://en.wikipedia.org/wiki/Gzip)
- [`snappy`](https://en.wikipedia.org/wiki/Snappy_(compression))
- [`zstd`](https://en.wikipedia.org/wiki/Zstd)
- [`LZ4HC`](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm))

In addition to this data can be compressed to KX's own qIPC format. For full information on KX file compression within kdb+/q see [here](https://code.kx.com/q/kb/file-compression/)

### Encryption

Data At Rest Encryption (DARE) is supported by PyKX with an explicit requirement on at least OpenSSL v1.0.2. To find out which version of OpenSSL you have available to you via PyKX you can find this using the following:

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

The encryption provided by this functionality specifically is Transparent Disk Encryption (TDE). TDE protects data at rest by encrypting database files on the hard drive and as a result on backup media. Encrypting your data with PyKX will be fully transparent to queries requiring no change to the logic used when querying data but will result in a time-penalty.

To use this functionality a user must have a password protected master key available, ideally with a unique password of high-entropy. For more information on the generation of a master key and a password more information is available [here](https://code.kx.com/q/kb/dare/#configuration).

## Functional walkthrough

This walkthrough will demonstrate the following steps:

- Create a compression objects to be used in global and per-partition data persistence.
- Persist a variety of Database partitions setting various compression configurations.
- Set the Python session to have globally configured encryption and compression settings.

### Generating compression objects

PyKX provides users with the ability to initialise compression and encryption class objects which can be used to set global configuration or by individual function operations. These respectively are supported via the `kx.Compress` and `kx.Encrypt` classes. For this section we will deal only with Compression.

As mentioned in the introduction compression within PyKX is supported using a variety of algorithms, the full list of algorithms that are available as part of the `kx.CompressionAlgorithm` enumeration.

```python
>>> import pykx as kx
>>> list(kx.CompressionAlgorithm)
[<CompressionAlgorithm.none: 0>, <CompressionAlgorithm.ipc: 1>, <CompressionAlgorithm.gzip: 2>, <CompressionAlgorithm.snappy: 3>, <CompressionAlgorithm.lz4hc: 4>, <CompressionAlgorithm.zstd: 5>]
```

Further details can be found through the `help` command:

```python
>>> help(kx.CompressionAlgorithm)
```

Once you are familiar with the options available to you it's time to initialize your first compression class. In this case generating a compression object which uses the `gzip` algorithm at compression level 8.

```python
>>> import pykx as kx
>>> compress = kx.Compress(algo=kx.CompressionAlgorithm.gzip, level=8)
```

This object will be used in the remaining sections of the walkthrough to use in a local (one-shot) and global context.

### Persisting Database partitions with various configurations

Not all data is created equally, in time-series applications such as algorithmic trading it is often the case that older data is less valuable than newer data. As a result of this it is often the case when backfilling historical data that you may more agressively compress older datasets. The compression logic provided by PyKX allows users to persist different partitions within a historical database to different levels.

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

2. Add a new partition using gzip compression

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

3. Add a final partition using `lz4hc` compression

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

Presently you can look at information about the persistence characteristics of your data using `kx.q('-21!')`, for example:

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

### Globally initialise compression and encryption

Global initialisation of compression and encryption allows all data that is persisted within from a process to be compressed. This can be useful when completing large batch operations on data where being specific about per partition/per file operations isn't necessary. In the below section we will deal with compression and encryption separately.

The compression settings that are used by PyKX are globally readable via `kx.q.z.zd`, when unset this value will return a PyKX Identity value as follows:

```python
>>> kx.q.z.zd
pykx.Identity(pykx.q('::'))
```

To set the process to use gzip globally this can be done using `global_init` on the generated `kx.Compress` object.

```python
>>> compress = kx.Compress(algo=kx.CompressionAlgorithm.gzip, level=9)
>>> compress.global_init()
>>> kx.q.z.z.d
pykx.LongVector(pykx.q('17 2 9'))
```

Globally initialising encryption is completed through the loading of the users encryption key into the process as follows

```python
>>> encrypt = kx.Encrypt(path='/path/to/my.key', password='PassWorD')
>>> encrypt.global_init()
```

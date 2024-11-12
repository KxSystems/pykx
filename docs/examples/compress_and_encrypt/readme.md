---
title:  PyKX Compress and encrypt
description: Compress and encrypt Examples
date: October 2024
author: KX Systems, Inc.,
tags: compression, encryption, PyKX
---

# Compress and encrypt example

_This example shows how to use various `#!python q` compression and encryption algorithms on a `#!python PyKX` table._

To follow along, download this <a href="./archive.zip" download>zip archive</a> that contains a copy of the python script and this writeup.

Here are the compression algorithms and their levels:

**Algorithm** | **Compression level**
--------- | -----------------
`ipc`     | `0`
`gzip`    | `0`-`9`
`snappy`  | `0`
`lz4hc`   | `1`-`12`

## Quickstart

To run this example, execute the `#!python compress_and_encrypt.py` file.

```py
$ python compress_and_encrypt.py
```

## Outcome

```py
Writing in-memory trades table with gzip: {
    "compressedLength": 12503352,
    "uncompressedLength": 36666552,
    "algorithm": 2,
    "logicalBlockSize": 17,
    "zipLevel": 9
}

Writing in-memory trades table with snappy: {
    "compressedLength": 18911879,
    "uncompressedLength": 36666552,
    "algorithm": 3,
    "logicalBlockSize": 17,
    "zipLevel": 0
}

Writing in-memory trades table with lz4hc: {
    "compressedLength": 15233016,
    "uncompressedLength": 36666552,
    "algorithm": 4,
    "logicalBlockSize": 17,
    "zipLevel": 12
}

Writing on-disk trades table with lz4hc: {
    "compressedLength": 15233016,
    "uncompressedLength": 36666552,
    "algorithm": 4,
    "logicalBlockSize": 17,
    "zipLevel": 12
}

Loading master key

Writing in-memory trades table with lz4hc and encryption: {
    "compressedLength": 15240112,
    "uncompressedLength": 36666552,
    "algorithm": 20,
    "logicalBlockSize": 17,
    "zipLevel": 12
}
```

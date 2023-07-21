# Compression and Encryption

This example shows how to use various `q` compression and encryption algorithms on a `PyKX` table.

To follow along with this example please feel free to download this <a href="./archive.zip" download>zip archive</a> that contains a copy of the python script and this writeup.

Here are the various compression algorithms used and the compression levels that they can use.

algorithm | compression level
--------- | -----------------
`ipc`     | `0`
`gzip`    | `0`-`9`
`snappy`  | `0`
`lz4hc`   | `1`-`12`

## Quickstart

This example can be ran by executing the `compress_and_encrypt.py` file.

```
$ python compress_and_encrypt.py
```

## Outcome

```
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

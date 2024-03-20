"""Functionality for the setting of compression and encryption configuration when
   handling on-disk data.

!!! Warning

        This functionality is provided in it's present form as a BETA
        Feature and is subject to change. To enable this functionality
        for testing please following configuration instructions
        [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`
"""

from . import beta_features
from .config import _check_beta

from enum import Enum
from math import log2
import os
from pathlib import Path

__all__ = [
    'CompressionAlgorithm',
    'Compress',
    'Encrypt',
]

beta_features.append('Compression and Encryption')


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


class CompressionAlgorithm(Enum):
    """
    The compression algorithm to be used when compressing a DB partition/column.

    Presently the supported algorithms are qipc, gzip, snappy, lz4hc and zstd.
    These algorithms support different compression levels denoting the agressivness
    of compression in each case.

    | algorithm | levels |
    |-----------|--------|
    | none      | 0      |
    | q IPC     | 0      |
    | gzip      | 0-9    |
    | snappy    | 0      |
    | lz4hc     | 0-16   |
    | zstd      | -7-22  |
    """
    none = 0
    ipc = 1
    gzip = 2
    snappy = 3
    lz4hc = 4
    zstd = 5


_compression_ranges = {
    CompressionAlgorithm.none:   range(0, 1),
    CompressionAlgorithm.ipc:    range(0, 1),
    CompressionAlgorithm.gzip:   range(0, 10),
    CompressionAlgorithm.snappy: range(0, 1),
    CompressionAlgorithm.zstd:   range(-7, 23),
    CompressionAlgorithm.lz4hc:  range(1, 17)}


class Encrypt():
    def __init__(self, path=None, password=None):
        """
        Initialize a class object which is used to control the use of encryption with PyKX.

        Parameters:
            path: Location of a users encryption key file as an 'str' object
            password: Password which had been set for encryption file

        Example:

            ```python
            >>> import pykx as kx
            >>> encrypt = kx.Encrypt('/path/to/mykey.key', 'mySuperSecretPassword')
            ```
        """
        _check_beta('Compression and Encryption')
        self.loaded = False
        path = Path(os.path.abspath(path))
        if not os.path.isfile(path):
            raise ValueError("Provided 'path' does not exist")
        self.path = path
        if password is None:
            raise ValueError('Password provided is None, please provide a str object')
        if not isinstance(password, str):
            raise TypeError('Password must be supplied as a string')
        self.password = password

    def load_key(self):
        """
        Load the encyption key within your process, note this will be a global load.

        Example:

            ```python
            >>> import pykx as kx
            >>> encrypt = kx.Encrypt('/path/to/mykey.key', 'mySuperSecretPassword')
            >>> encrypt.load_key()
            ```
        """
        q('{-36!(hsym x;y)}', self.path, bytes(self.password, 'UTF-8'))
        self.loaded = True


class Compress():
    def __init__(self,
                 algo=CompressionAlgorithm.none,
                 block_size=2**17,
                 level=None):
        """
        Initialize a class object which is used to control encryption within PyKX.

        Parameters:
            algo: Compression algorithm to be used when applying compression,
                this must be one of:

                - `kx.CompressionAlgorithm.none`
                - `kx.CompressionAlgorithm.ipc`
                - `kx.CompressionAlgorithm.gzip`
                - `kx.CompressionAlgorithm.snappy`
                - `kx.CompressionAlgorithm.lz4hc`

            block_size: Must be a port of 2 between 12 and 20 denoting the pageSize or
                allocation granularity to 1MB, see
                [here](https://code.kx.com/q/kb/file-compression/#compression-parameters)
                for more information.

            level: The degree to which compression will be applied, when non zero values
                are supported for a supported algorithm larger values will result in
                higher compression ratios.

        Example:

            ```python
            >>> import pykx as kx
            >>> comp = kx.Compress(kx.CompressionAlgorithm.gzip, level=5)
            ```
        """
        _check_beta('Compression and Encryption')
        self.algorithm = algo
        if block_size & (block_size - 1):
            raise ValueError(f'block_size must be a power of 2, not {block_size}')
        self.encrypt = False
        self.block_size = int(log2(block_size))
        if (algo == CompressionAlgorithm.zstd) & q('.z.K<4.1').py():
            raise ValueError("'CompressionAlgorithm.zstd' only supported on PyKX>=4.1")
        compression_range = _compression_ranges[algo]
        if level is None:
            level = compression_range.stop -1
        elif level not in compression_range:
            raise ValueError(
                f'Invalid level {level} for {algo} '
                f'algorithm. Valid range is {compression_range}')
        self.compression_level = level

    def global_init(self, encrypt=False):
        """
        Globally initialise compression settings, when completed any persistence
            operation making use of `kx.q.set` will be compressed based on the user
            specified compression settings

        Parameters:
            encrypt: A `kx.Encrypt` object denoting if and using what credentials
                encryption is to be applied.

        Example:

            ```python
            >>> import pykx as kx
            >>> comp = kx.Compress(kx.CompressionAlgorithm.gzip, level=2)
            >>> kx.q.z.zd
            pykx.Identity(pykx.q('::'))
            >>> comp.global_init()
            >>> kx.q.z.zd
            pykx.LongVector(pykx.q('17 2 2'))
            ```
        """
        if not self.encrypt:
            if isinstance(encrypt, Encrypt):
                if not encrypt.loaded:
                    encrypt.load_key()
                self.encrypt = True
            else:
                self.encrypt = False
        q.z.zd = [self.block_size,
                  self.algorithm.value + (16 if self.encrypt else 0),
                  self.compression_level]

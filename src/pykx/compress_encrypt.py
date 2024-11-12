from enum import Enum
from math import log2
import os
from pathlib import Path

__all__ = [
    'CompressionAlgorithm',
    'Compress',
    'Encrypt',
]


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


class CompressionAlgorithm(Enum):
    """
    The compression algorithm used when compressing a DB partition/column.

    Supported algorithms are qipc, gzip, snappy, lz4hc and zstd.
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
    def __init__(self, path: str = None, password: str = None) -> None:
        """
        A class for controlling the use of encryption with PyKX.

        Parameters:
            path: Location of a user's encryption key file
            password: Password for encryption file

        Returns:
            A `#!python None` object on successful invocation

        Example:

        ```python
        >>> import pykx as kx
        >>> encrypt = kx.Encrypt('/path/to/mykey.key', 'mySuperSecretPassword')
        ```
        """
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

    def load_key(self) -> None:
        """
        Load the encyption key from the file given during class initialization.
        This overwrites the master key in the embedded q process. See
        [here](https://code.kx.com/q/basics/internal/#-36-load-master-key) for details.

        Returns:
            A `#!python None` object on successful invocation

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
                 algo: CompressionAlgorithm = CompressionAlgorithm.none,
                 block_size: int = 2**17,
                 level: int = None
    ) -> None:
        """
        A class object for controlling q compression with PyKX.

        Parameters:
            algo: Compression algorithm to use. This must be one of:

                - `#!python kx.CompressionAlgorithm.none`
                - `#!python kx.CompressionAlgorithm.ipc`
                - `#!python kx.CompressionAlgorithm.gzip`
                - `#!python kx.CompressionAlgorithm.snappy`
                - `#!python kx.CompressionAlgorithm.lz4hc`

            block_size: Must be a power of 2 between 12 and 20 denoting the pageSize or
                allocation granularity to 1MB. Read [compression
                parameters](https://code.kx.com/q/kb/file-compression/#compression-parameters)
                for more information.

            level: Compression level for the `#!python algo` parameter. Algorithms that support
                non-zero values have higher compression ratios as the provided level increases.

        Returns:
            A `#!python None` object on successful invocation

        Example:

        ```python
        >>> import pykx as kx
        >>> comp = kx.Compress(kx.CompressionAlgorithm.gzip, level=5)
        ```
        """
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

    def global_init(self, encrypt: bool = False) -> None:
        """
        Globally initialise compression settings. Once run, using `#!python kx.q.set` to
            persist data to disk compresses the data based on specified compression settings.
            Refer to [compression by
            default](https://code.kx.com/q/kb/file-compression/#compression-by-default)
            for more details.

        Parameters:
            encrypt: A `#!python kx.Encrypt` object denoting if and using what credentials
                encryption is to be applied.

        Returns:
            A `#!python None` object on successful invocation

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

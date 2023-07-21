from enum import Enum
import json
from math import log2
from pathlib import Path
from textwrap import dedent
from typing import Dict, Optional, Union

import pykx
from pykx import q


class CompressionAlgorithm(Enum):
    """The compression algorithms available to q."""
    none = 0
    ipc = 1
    gzip = 2
    snappy = 3
    lz4hc = 4


def load_master_key(key_path: Path, password: Union[str, bytes]) -> None:
    """Loads the master key into the q process.

    Must be run prior to encrypting anything using the q process.
    """
    load_master_key = q('-36!')
    load_master_key([key_path, bytes(password, 'utf-8')])


def compress(source: Union[Path, pykx.K],
             target: Path,
             block_size: int = 2**17,
             algorithm: CompressionAlgorithm = CompressionAlgorithm.none,
             compression_level: Optional[int] = None,
             encrypt: bool = False
) -> Dict[str, int]:
    """Compresses (and/or encrypts) a K object written to disk or in-memory

    Parameters:
        source: What will be compressed/encrypted. Either a path to a q object on disk or a
            `pykx.K` object.
        target: The path the compressed/encrypted data will be written to.
        block_size: The size of the compressed blocks. Must be a power of `2` (i.e. `2 ** 17` for
            128 kB blocks). Minimum varies by platform, but generally a block size between
            `2 ** 12` and `2 ** 20` is advisable.
        algorithm: The compression algorithm to be used.
        compression_level: How compressed the data should be. Varies by selected algorithm. The
            valid values for each algorithm are shown below:

            algorithm | compression level
            --------- | -----------------
            `none`    | `0`
            `ipc`     | `0`
            `gzip`    | `0`-`9`
            `snappy`  | `0`
            `lz4hc`   | `1`-`12`

            Defaults to the maximum compression level for the selected algorithm.

        encrypt: Whether the data should be encrypted. The master key must be loaded.

    Returns:
        Info about the compressed data.
    """
    if isinstance(source, pykx.K):
        _compress = q('{y set x}')
    else:
        _compress = lambda x, y: q('-19!', (x, *y)) # noqa: E731

    compression_stats = q('-21!')

    if block_size & (block_size - 1):
        raise ValueError(f'block_size must be a power of 2, not {block_size}')

    compression_range = {
        CompressionAlgorithm.none:   range(0, 1),
        CompressionAlgorithm.ipc:    range(0, 1),
        CompressionAlgorithm.gzip:   range(0, 10),
        CompressionAlgorithm.snappy: range(0, 1),
        CompressionAlgorithm.lz4hc:  range(1, 13),
    }[algorithm]

    if compression_level is None:
        compression_level = compression_range.stop - 1
    elif compression_level not in compression_range:
        raise ValueError(
            f'Invalid compression level {compression_level} for {algorithm} '
            f'algorithm. Valid range is {compression_range}')

    return compression_stats(_compress(source, [
        target,
        int(log2(block_size)),
        algorithm.value + (16 if encrypt else 0),
        compression_level
    ])).py()


def setup():
    q(dedent('''
        mktrades:{[tickers; sz]
            dt:2015.01.01+sz?10;
            tm:sz?24:00:00.000;
            sym:sz?tickers;
            qty:10*1+sz?1000;
            px:90.0+(sz?2001)%100;
            t:([] dt; tm; sym; qty; px);
            t:`dt`tm xasc t;
            t:update px:6*px from t where sym=`goog;
            t:update px:2*px from t where sym=`ibm;
            t:update val:qty*px from t;
            t};

        trades:mktrades[`aapl`goog`ibm;1000000];
        '''))


def demo():
    print('Writing in-memory trades table with gzip:', end=' ')
    print(json.dumps(compress( # Using json for pretty printing
        q('trades'),
        Path('./trades_compressed_gzip'),
        algorithm=CompressionAlgorithm.gzip
    ), indent=4), end='\n\n')

    print('Writing in-memory trades table with snappy:', end=' ')
    print(json.dumps(compress( # Using json for pretty printing
        q('trades'),
        Path('./trades_compressed_snappy'),
        algorithm=CompressionAlgorithm.snappy
    ), indent=4), end='\n\n')

    print('Writing in-memory trades table with lz4hc:', end=' ')
    print(json.dumps(compress( # Using json for pretty printing
        q('trades'),
        Path('./trades_compressed_lz4hc'),
        algorithm=CompressionAlgorithm.lz4hc
    ), indent=4), end='\n\n')

    print('Writing on-disk trades table with lz4hc:', end=' ')
    source = Path(__file__).parent/'trades_uncompressed'
    q('set', source, q('trades'))
    print(json.dumps(compress( # Using json for pretty printing
        source,
        Path('./trades_ondisk_compressed_lz4hc'),
        algorithm=CompressionAlgorithm.lz4hc
    ), indent=4), end='\n\n')

    # WARNING: Do not use this key for anything in production! This is
    # purely for demonstration. See
    # https://code.kx.com/q/kb/dare/#configuration for information about
    # generating a key.
    print('Loading master key\n')
    load_master_key(Path(__file__).parent/'demokey.key', 'demokeypass')

    print('Writing in-memory trades table with lz4hc and encryption:', end=' ')
    print(json.dumps(compress( # Using json for pretty printing
        q('trades'),
        Path('./trades_encrypted_compressed_lz4hc'),
        algorithm=CompressionAlgorithm.lz4hc,
        encrypt=True
    ), indent=4), end='\n\n')


if __name__ == '__main__':
    setup()
    demo()

from enum import Enum
import json
from pathlib import Path
from textwrap import dedent
from typing import Dict, Union

import pykx as kx
from pykx import q


class CompressionAlgorithm(Enum):
    """The compression algorithms available to q."""
    none = 0
    ipc = 1
    gzip = 2
    snappy = 3
    lz4hc = 4


def compress(source: Union[Path, kx.K],
             target: Path,
) -> Dict[str, int]:
    """Compresses (and/or encrypts) a K object written to disk or in-memory

    Parameters:
        source: What will be compressed/encrypted. Either a path to a q object on disk or a
            `kx.K` object.
        target: The path the compressed/encrypted data will be written to.

    Returns:
        Info about the compressed data.
    """
    if isinstance(source, kx.K):
        _compress = q('{y set x}')
    else:
        _compress = lambda x, y: q("{y set x}", x, [y, *q.z.zd.py()]) # noqa: E731

    _compress(source, target)

    return q('-21!', target).py()


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
    kx.Compress(kx.CompressionAlgorithm.gzip).global_init()
    print(json.dumps(compress(
        q('trades'),
        Path('./trades_compressed_gzip')
    ), indent=4), end='\n\n')

    print('Writing in-memory trades table with snappy:', end=' ')
    kx.Compress(kx.CompressionAlgorithm.snappy).global_init()
    print(json.dumps(compress( # Using json for pretty printing
        q('trades'),
        Path('./trades_compressed_snappy'),
    ), indent=4), end='\n\n')

    print('Writing in-memory trades table with lz4hc:', end=' ')
    kx.Compress(algo=kx.CompressionAlgorithm.lz4hc, level=5).global_init()
    print(json.dumps(compress( # Using json for pretty printing
        q('trades'),
        Path('./trades_compressed_lz4hc')
    ), indent=4), end='\n\n')

    print('Writing on-disk trades table with lz4hc:', end=' ')
    source = Path(__file__).parent/'trades_uncompressed'
    q('set', source, q('trades'))
    print(json.dumps(compress( # Using json for pretty printing
        source,
        Path('./trades_ondisk_compressed_lz4hc'),
    ), indent=4), end='\n\n')

    # WARNING: Do not use this key for anything in production! This is
    # purely for demonstration. See
    # https://code.kx.com/q/kb/dare/#configuration for information about
    # generating a key.
    print('Loading master key\n')
    kx.Encrypt(Path(__file__).parent/'demokey.key', 'demokeypass').load_key()

    print('Writing in-memory trades table with lz4hc and encryption:', end=' ')
    print(json.dumps(compress( # Using json for pretty printing
        q('trades'),
        Path('./trades_encrypted_compressed_lz4hc')
    ), indent=4), end='\n\n')


if __name__ == '__main__':
    setup()
    demo()

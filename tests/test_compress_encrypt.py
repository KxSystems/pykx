import os
from pathlib import Path

# Do not import pykx here - use the `kx` fixture instead!
import pytest


def test_compress_encrypt_errors(kx):
    with pytest.raises(ValueError) as err:
        kx.Encrypt('/path')
    assert "Provided 'path' does not exist" in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.Encrypt('tests/test_files/encrypt.txt')
    assert "Password provided is None, please provide a str object" in str(err.value)

    with pytest.raises(TypeError) as err:
        kx.Encrypt('tests/test_files/encrypt.txt', 10)
    assert "Password must be supplied as a string" in str(err.value)

    with pytest.raises(kx.QError) as err:
        encrypt=kx.Encrypt('tests/test_files/testkek.key', 'passwrd')
        encrypt.load_key()
    assert "Invalid password for" in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.Compress(block_size=24)
    assert 'block_size must be a power of 2' in str(err.value)

    if os.getenv('PYKX_4_1_ENABLED') is None:
        with pytest.raises(ValueError) as err:
            kx.Compress(algo=kx.CompressionAlgorithm.zstd)
        assert "'CompressionAlgorithm.zstd' only supported on" in str(err.value)

    with pytest.raises(ValueError) as err:
        kx.Compress(algo=kx.CompressionAlgorithm.gzip, level=100)
    assert 'Invalid level 100 for CompressionAlgorithm.gzip' in str(err.value)


@pytest.mark.isolate
def test_compression():
    os.environ['PYKX_BETA_FEATURES'] = 'True'
    import pykx as kx
    compress = kx.Compress(kx.CompressionAlgorithm.ipc)
    compress.global_init()
    assert kx.q.z.zd.py() == [17, 1, 0]

    compress = kx.Compress(kx.CompressionAlgorithm.gzip, level=9)
    compress.global_init()
    assert kx.q.z.zd.py() == [17, 2, 9]

    compress = kx.Compress(kx.CompressionAlgorithm.lz4hc, level=10)
    compress.global_init()
    assert kx.q.z.zd.py() == [17, 4, 10]


@pytest.mark.isolate
def test_compression_4_1():
    os.environ['PYKX_4_1_ENABLED'] = 'True'
    os.environ['PYKX_BETA_FEATURES'] = 'True'
    import pykx as kx
    compress = kx.Compress(kx.CompressionAlgorithm.zstd, level=0)
    compress.global_init()
    assert kx.q.z.zd.py() == [17, 5, 0]


@pytest.mark.isolate
def test_encrypt():
    os.environ['PYKX_BETA_FEATURES'] = 'True'
    import pykx as kx
    encrypt = kx.Encrypt('tests/test_files/testkek.key', 'password')
    encrypt.load_key()
    # If this has run, the encryption key has been loaded appropriately
    # this can be tested more rigorously once kdb+ 4.0 2024.03.02
    assert kx.q('-36!(::)').py()


@pytest.mark.isolate
def test_encrypt_path():
    os.environ['PYKX_BETA_FEATURES'] = 'True'
    import pykx as kx
    encrypt = kx.Encrypt(Path('tests/test_files/testkek.key'), 'password')
    encrypt.load_key()
    # If this has run, the encryption key has been loaded appropriately
    # this can be tested more rigorously once kdb+ 4.0 2024.03.02
    assert kx.q('-36!(::)').py()

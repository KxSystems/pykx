"""Regression tests for KDB-X Python under q on Windows (KXI-46558).

`pykxq.dll` carried a static import on `q.dll`, so the Windows loader bound its q API calls to any
copy of the interpreter it found by name rather than the `q.exe` that was running.

Checked two ways: the import table is read out of the built DLLs, and `pykx.q` is loaded in a real
q process, once normally and once with a stray `q.dll` in the working directory.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from platform import system

import pytest


pytestmark = pytest.mark.skipif(
    system() != 'Windows',
    reason='the q.dll import and the loader search order are Windows only',
)

_QSCRIPT = Path(__file__).parent/'qscripts'/'underq_smoke.q'


def _pykx_dir():
    """Locate the package without importing it, which would start embedded q."""
    spec = importlib.util.find_spec('pykx')
    assert spec is not None and spec.origin is not None, 'pykx is not installed'
    return Path(spec.origin).parent


def _imported_dlls(path):
    """Return the DLL names in a PE image's import directory.

    Hand rolled to avoid a dumpbin or PE library dependency in the test extras.
    """
    data = Path(path).read_bytes()
    pe = int.from_bytes(data[0x3C:0x40], 'little')
    assert data[pe:pe + 4] == b'PE\0\0', f'{path} is not a PE image'
    coff = pe + 4
    n_sections = int.from_bytes(data[coff + 2:coff + 4], 'little')
    opt_size = int.from_bytes(data[coff + 16:coff + 18], 'little')
    opt = coff + 20
    # The data directory follows the optional header's fixed fields, which are 112 bytes for
    # PE32+ (magic 0x20b) and 96 for PE32. Entry 1 of it is the import table.
    magic = int.from_bytes(data[opt:opt + 2], 'little')
    dirs = opt + (112 if magic == 0x20b else 96)
    import_rva = int.from_bytes(data[dirs + 8:dirs + 12], 'little')
    if not import_rva:
        return []

    sections = []
    for i in range(n_sections):
        sh = opt + opt_size + i * 40
        virtual_size = int.from_bytes(data[sh + 8:sh + 12], 'little')
        raw_size = int.from_bytes(data[sh + 16:sh + 20], 'little')
        sections.append((
            int.from_bytes(data[sh + 12:sh + 16], 'little'),  # VirtualAddress
            max(virtual_size, raw_size),
            int.from_bytes(data[sh + 20:sh + 24], 'little'),  # PointerToRawData
        ))

    def offset(rva):
        for va, size, raw in sections:
            if va <= rva < va + size:
                return raw + rva - va
        raise AssertionError(f'RVA {hex(rva)} falls outside every section of {path}')

    names = []
    entry = offset(import_rva)
    while True:
        descriptor = data[entry:entry + 20]
        # The descriptor array is terminated by an all zero entry.
        if len(descriptor) < 20 or descriptor == bytes(20):
            return names
        name = offset(int.from_bytes(descriptor[12:16], 'little'))
        names.append(data[name:data.index(b'\0', name)].decode('ascii'))
        entry += 20


@pytest.mark.parametrize('lib', ['pykxq.dll', 'pykx.dll'])
def test_q_extension_does_not_statically_import_q_dll(lib):
    """Neither extension may name `q.dll` in its import table.

    One direct call to a q function is enough to put it back, so this guards the link line too.
    """
    path = _pykx_dir()/lib
    if not path.exists():
        pytest.skip(f'{lib} is not present in this install')
    imports = [x.lower() for x in _imported_dlls(path)]
    assert imports, f'no import directory found in {path}'
    assert 'q.dll' not in imports, (
        f'{lib} imports q.dll, so the Windows loader binds its q symbols to whichever copy of '
        f'the interpreter it finds rather than to the process that loaded it. Imports: {imports}'
    )


def _install_qhome():
    """Put pykx.q and pykx_init.q_ in QHOME for the q subprocess to load."""
    subprocess.run(
        (str(Path(sys.executable).as_posix()), '-c', 'import pykx;pykx.install_into_QHOME()'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
    )


def _run_smoke(cwd):
    return subprocess.run(
        ('q', str(_QSCRIPT), '-q'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180, cwd=str(cwd),
        env=dict(os.environ),
    )


def _assert_smoke_passed(out, context):
    assert 'fatal error' not in out.stdout, (
        f'q died while loading pykx.q {context}\n{out.stdout}'
    )
    assert 'underq_smoke,PASS' in out.stdout, f'pykx.q is not usable {context}\n{out.stdout}'
    assert out.returncode == 0, out.stdout


@pytest.mark.timeout(300)
def test_pykx_q_loads_under_q(tmp_path):
    _install_qhome()
    _assert_smoke_passed(_run_smoke(tmp_path), 'in a clean working directory')


@pytest.mark.timeout(300)
def test_pykx_q_loads_beside_a_stray_q_dll(tmp_path):
    """A `q.dll` in the working directory must not be picked up as the interpreter.

    This is the original report. The decoy is the embedded `q.dll`, a complete interpreter that
    loads cleanly, so the fault used to land on the first q object allocated through it.
    """
    _install_qhome()
    decoy = _pykx_dir()/'lib'/'w64'/'q.dll'
    if not decoy.exists():
        pytest.skip('no embedded q.dll available to use as a decoy')
    shutil.copy(decoy, tmp_path/'q.dll')
    _assert_smoke_passed(_run_smoke(tmp_path), 'with a q.dll in the working directory')


@pytest.mark.timeout(300)
def test_install_into_QHOME_plants_no_interpreter(tmp_path):
    """`install_into_QHOME` must not leave a `q.dll` where the loader would prefer it."""
    import pykx
    pykx.install_into_QHOME(to_local_folder=str(tmp_path))
    planted = [str(p) for p in tmp_path.rglob('q.dll')]
    assert not planted, f'install_into_QHOME installed a second q interpreter: {planted}'
    assert (tmp_path/'pykx.q').exists(), 'install_into_QHOME did not install pykx.q'

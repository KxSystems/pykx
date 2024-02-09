#!/usr/bin/env python

"""
Handles building/packaging PyKX. Generally pip should be used instead of
executing this file directly.
"""

# Setuptools may run into errors if distutils is imported before it is.
import setuptools # noqa: I100, F401

from contextlib import contextmanager # noqa: I100
from contextlib import redirect_stderr
from distutils.command.clean import clean as default_clean
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_python_inc
from glob import iglob
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Generator, List, Tuple

from Cython.Build import cythonize
import numpy as np
from setuptools import Extension
from setuptools.command.build_ext import build_ext as default_build_ext
from setuptools import setup
import tomli


script_dir = Path(__file__).parent
src_dir = script_dir/'src'/'pykx'

pypy = platform.python_implementation() == 'PyPy'

system = platform.system()

q_lib_dir_name = {
    'Darwin': 'm64',
    'Linux': 'l64',
    'Windows': 'w64',
}[system]

py_minor_version = sys.version_info[1]
debug = os.environ.get('PYKX_DEBUG', '').lower() in {'true', '1'}

windows_vcpkg_content = script_dir.resolve()/'vcpkg'/'installed'/'x64-windows-static-md'
windows_include_dirs = (str(windows_vcpkg_content/'include'),) if system == 'Windows' else ()
windows_library_dirs = () if system != 'Windows' else (str(Path(sys.exec_prefix)/'libs'),
                                                       str(Path(sys.base_exec_prefix)/'libs'),
                                                       str(windows_vcpkg_content/'lib'))
windows_libraries = () if system != 'Windows' else ('psapi', 'q')


def rmrf(path: str):
    """Delete the file tree with ``path`` as its root"""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        try:
            os.remove(path)
        except OSError:
            pass


@contextmanager
def cd(dest_dir: os.PathLike) -> Generator[None, None, None]:
    """Change the current working directory within the context."""
    src_dir = os.getcwd()
    os.chdir(dest_dir)
    try:
        yield
    finally:
        os.chdir(src_dir)


class build_ext(default_build_ext):
    def run(self):
        self.build_q_c_extensions()
        super().run()

    def build_q_c_extension(self, compiler, lib, lib_ext, library=None):
        libs = [
            'dl',
            *windows_libraries,
        ]
        if library is not None:
            libs.extend(library)
        return compiler.link_shared_object(
            objects=compiler.compile(
                sources=[str(src_dir/f'{lib}.c')],
                output_dir=self.build_temp,
                macros=[('KXVER', '3')],
                include_dirs=[
                    get_python_inc(),
                    str(src_dir/'include'),
                    *windows_include_dirs,
                ],
                debug=self.debug,
                extra_preargs=[
                    *(
                        ('-undefined dynamic_lookup',) if system == 'Darwin' else
                        ('/LD /Fepykxq.dll q.lib -I include',) if system == 'Windows' else ()
                    ),
                ],
            ),
            output_filename=str(Path(self.build_lib)/'pykx'/f'{lib}.{lib_ext}'),
            libraries=libs,
            library_dirs=[
                str(src_dir/'lib'/q_lib_dir_name),
                *windows_library_dirs,
            ],
            debug=self.debug,
            build_temp=self.build_temp,
        )

    def build_q_c_extensions(self):
        compiler = new_compiler(compiler=self.compiler,
                                verbose=self.verbose,
                                dry_run=self.dry_run,
                                force=self.force)
        customize_compiler(compiler)
        if hasattr(compiler, 'initialize'):
            compiler.initialize()
        lib_ext = 'dll' if system == 'Windows' else 'so'
        self.build_q_c_extension(compiler, 'pykx', lib_ext)
        self.build_q_c_extension(compiler, 'pykxq', lib_ext)
        if system != 'Windows':
            self.build_q_c_extension(compiler, '_tcore', lib_ext, library=['pthread'])


class clean(default_clean):
    """clean command that cleans some extra files and directories."""
    targets = (iglob(x) for x in (
        'build',
        'dist',
        '.eggs',
        'pykx.egg-info',
        str(src_dir/'*.so*'),
        str(src_dir/'build'),
    ))

    def run(self):
        with cd(script_dir):
            for z in (x for y in self.targets for x in y):
                rmrf(str(z))
        for f in src_dir.iterdir():
            if f.suffix == '.pyx':
                rmrf(str(f.parent/(f.stem + '.c')))
                rmrf(str(f.parent/(f.stem + '.html')))
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                # This command has some useless/annoying output that can make it appear like
                # there's an issue when there isn't, so we silence it.
                super().run()
                # If there's a real problem the error will propagate out, so we'll still see it


def cythonize_extensions(extensions: List[Extension]) -> List[Extension]:
    """Convert .pyx/.pxd Extensions into regular .c/.cpp Extensions"""
    if 'clean' in sys.argv:
        return []
    with cd(script_dir/'src'):
        cythonized = cythonize(
            extensions,
            language_level=3,
            nthreads=os.cpu_count(),
            annotate=debug,
            # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives # noqa: E501
            compiler_directives={
                'binding': True,
                'boundscheck': False,
                'wraparound': False,
                'profile': debug and not pypy,
                'linetrace': debug and not pypy,
                'always_allow_keywords': True,
                'embedsignature': False,
                'emit_code_comments': True,
                'initializedcheck': False,
                'nonecheck': False,
                'optimize.use_switch': True,
                # Warns about any variables that are implicitly declared
                # without a cdef declaration
                'warn.undeclared': False,
                'warn.unreachable': True,
                'warn.maybe_uninitialized': False,
                'warn.unused': True,
                'warn.unused_arg': False,
                'warn.unused_result': False,
                'warn.multiple_declarators': True,
            },
        )
    for cy in cythonized:
        cy.sources[0] = str(Path(f'src/{cy.sources[0]}'))
    return cythonized


def ext(name: str,
        libraries: Tuple[str] = (),
        extra_compile_args: Tuple[str] = (),
        extra_link_args: Tuple[str] = (),
        numpy: bool = False,
        cython: bool = True,
) -> Extension:
    nix_extra_compile_args = (
        '-O3',
        '-Wall',
        '-Wextra',
        # It'd be nice if we could leave -Wunused-variable enabled, but when Cython's binding
        # option is True (which it needs to be to generate signatures for its callables) tons of
        # unused variables are created. This clutters the compiler output, which could hide
        # important warnings.
        '-Wno-unused-variable',
    )
    return Extension(
        name=f'pykx.{name}',
        sources=[str(Path(f'pykx/{name}.pyx' if cython else f'src/pykx/{name}.c'))],
        include_dirs=[
            str(src_dir),
            str(src_dir/'include'),
            *windows_include_dirs,
            *([np.get_include()] if numpy else ()),
        ],
        library_dirs=[
            str(src_dir/'lib'/q_lib_dir_name),
            *windows_library_dirs,
        ],
        libraries=list(libraries),
        extra_compile_args=[
            *{
                'Darwin': nix_extra_compile_args,
                'Linux': nix_extra_compile_args,
            }.get(system, ()),
            *extra_compile_args
        ],
        extra_link_args=[
            *{
                'Linux': ('-Wl,-rpath,$ORIGIN/lib/Linux',),
            }.get(system, ()),
            *extra_link_args
        ],
        define_macros=[
            # Prevents the use of deprecated Numpy C APIs
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            *((('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')) if debug else ())
        ],
    )


if __name__ == '__main__':
    with open(script_dir/'README.md') as f:
        readme = f.read()

    with open(script_dir/'pyproject.toml', 'rb') as pyproject_file:
        pyproject = tomli.load(pyproject_file)['project']
    exts = [
        *cythonize_extensions([
            ext('core', libraries=['dl', *windows_libraries]),
            ext('_ipc', numpy=True),
            ext('toq', numpy=True),
            ext('_wrappers', numpy=True, extra_compile_args=[
                # Type punning is used to support GUIDs
                '-Wno-strict-aliasing',
                # The default variable tracking size limit is too small for this module
                *(('--param=max-vartrack-size=120000000',) if debug else ()),
            ] if system != 'Windows' else []),
        ]),
    ]
    if py_minor_version >= 8: # python 3.8 or higher is required for NEP-49
        exts.append(ext('_numpy', numpy=True, cython=False, libraries=['dl', *windows_libraries]))
    exts.append(ext('numpy_conversions',
                    numpy=True,
                    cython=False,
                    libraries=['dl', *windows_libraries]))
    exts.append(ext('_pykx_helpers',
                    numpy=False,
                    cython=False,
                    libraries=['dl', *windows_libraries]))
    setup(
        name=pyproject['name'],
        description=pyproject['description'],
        long_description=readme,
        long_description_content_type='text/markdown',
        author=pyproject['authors'][0]['name'],
        author_email=pyproject['authors'][0]['email'],
        url=pyproject['urls']['repository'],
        classifiers=pyproject['classifiers'],
        keywords=pyproject['keywords'],
        # The project metadata above should be gathered automatically from
        # pyproject.toml, but until it is we will provide it to setuptools manually
        packages=['pykx'],
        package_dir={'pykx': str(Path('src/pykx'))},
        cmdclass={
            'build_ext': build_ext,
            'clean': clean,
        },
        include_package_data=True, # makes setuptools use MANIFEST.in
        zip_safe=False, # required by Cython
        ext_modules=exts,
        # `install_requires` and `extras_require` *should* be automatically handled by PIP through
        # it reading the `pyproject.toml` file, but for whatever reason it doesn't seem to be
        # installing the deps or finding the available extras. This workaround allows us to keep
        # the info in `pyproject.toml` while ensuring it gets used properly. It should also be
        # forward compatible with a version of PIP/setuptools/whatever that fixes this problem.
        install_requires=pyproject['dependencies'],
        extras_require={
            **pyproject['optional-dependencies'],
            'all': [y for x in pyproject['optional-dependencies'].values() for y in x],
        },
    )

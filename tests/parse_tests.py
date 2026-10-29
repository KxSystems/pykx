import ast
from collections import namedtuple
from inspect import getmembers, getsource, isclass, isfunction
import os
import re
import sys
from textwrap import dedent

import test_cast
import test_db
import test_license
import test_nbextension
import test_splay
import test_tick
import test_compress_encrypt
import test_config
import test_pandas_agg
import test_pandas_apply
import test_pandas_map
import test_pandas_replace
import test_pandas_reset_index
import test_pandas_set_index
import test_random
import test_register
import test_reimport
import test_remote
import test_schema
import test_sql
import test_system
import test_console
import test_ctx
import test_exceptions
import test_ipc
import test_pandas_api
import test_pykx
import test_q_foreign
import test_q_future
import test_q
import test_query
import test_read
import test_toq
import test_util
import test_wrappers
import test_pandas_sorting
import test_write


py_minor_version = sys.version_info[1]


class ParamTest:
    def __init__(self, x): # noqa
        if callable(x[1]) and not isclass(x[1]):
            self._type = 'func'
            self.func_name = x[0]
            self._func = x[1]
            source = []
            marks = ['ipc', 'isolate', 'licensed', 'nep', 'pandas_api', 'pypy', 'static']
            etc = ['test', 'assert', 'np_ts', 'q_', 'mkt', 'self.v', 'kt', 'self.create_splayed',
                   'self.create_partitioned_table', 'self.reset_sp', 'self.DEFAULT_PATH',
                   'self.make_tab', 'self.errStr']
            for line in getsource(x[1]).split('\n'):
                if line == '' or line == '\n':
                    continue
                elif '@' in line:
                    if any([x in line for x in marks]):
                        continue
                if 'def' in line and 'test' in line:
                    line = line.replace('kx, ', '')\
                        .replace(', kx, ', '')\
                        .replace(', kx)', ')')\
                        .replace('(kx)', '()')\
                        .replace('q, ', '')\
                        .replace(', q, ', '')\
                        .replace(', q)', ')')\
                        .replace('(q)', '()')
                if ('def' not in line and not line.startswith(' ' * 8))\
                    or any([x in line for x in etc]
                ):
                    source.append(line.replace('self,', '')
                                  .replace('self.', '')
                                  .replace('(self)', '()'))
                else:
                    source.append(line)
            # `time` is datetime.time in the generated files, which import sleep directly
            self.source = dedent('\n'.join(source)).replace('time.sleep(', 'sleep(')
            self.decorators = [
                line.strip().split()[0]
                for line in getsource(x[1])[:getsource(x[1]).find("def ")].strip().splitlines()
                if line.strip()[0] == "@"
            ]
            self.decorators = [x for x in self.decorators if 'ignore' not in x]
        else:
            self._type = 'class'
            self.class_name = x[0]
            self._class = x[1]
            self.functions = [ParamTest(x) for x in getmembers(x[1], isfunction)]
            self.source = getsource(x[1])
            self.class_attrs = []
            for line in self.source.split('\n'):
                if 'def' in line:
                    break
                if '@' in line or 'class' in line:
                    continue
                self.class_attrs.append(line)
            first_def = next((i for i, x in enumerate(self.source.split('\n')) if 'def' in x), 0)
            src_lines = self.source.split('\n')
            for node in ast.parse(dedent(self.source)).body[0].body:
                if isinstance(node, ast.Assign) and node.lineno > first_def:
                    self.class_attrs.extend(src_lines[node.lineno - 1:node.end_lineno])
            self.decorators = [
                line.strip().split()[0]
                for line in self.source[:self.source.find("class ")].strip().splitlines()
                if line.strip()[0] == "@"
            ]
            self.decorators = [x for x in self.decorators if 'ignore' not in x]

    def append_to_name(self, text):
        self.func_name += text
        lines = self.source.split('\n')
        new_lines = []
        first = True
        for line in lines:
            if first and 'def' in line:
                split = line.split('(')
                split[0] += text + '('
                new_lines.append(''.join(split))
                first = False
            else:
                new_lines.append(line)
        self.source = '\n'.join(new_lines) + '\n'
        return self

    def add_class_attrs(self, attrs):
        for name in {m.group(1) for a in attrs if (m := re.match(r'\s+(\w+)\s*=', a))}:
            self.source = re.sub(rf'\bself\.{name}\b', name, self.source)
        new_lines = []
        added = False
        for line in self.source.split('\n'):
            if not added and 'def' in line:
                new_lines.append(line)
                new_lines.extend(attrs)
                added = True
                continue
            new_lines.append(line)
        self.source = '\n'.join(new_lines) + '\n'
        return self


ModuleGlobal = namedtuple('ModuleGlobal', ['func_name', 'source'])


# Modules whose tests use module-level values which flattening would otherwise drop
CARRY_GLOBALS = ('test_pandas_sorting',)


class ParseModule:
    def __init__(self, mod):
        self.module = mod
        self.module_str = str(self.module).split(' ')[1]

        self.members = getmembers(self.module, isfunction)
        # pytest>=8.4 wraps fixtures in an object which is no longer a function
        self.members.extend([(x[0], x[1].__wrapped__) for x in getmembers(self.module)
                             if type(x[1]).__name__ == 'FixtureFunctionDefinition'])
        self.members.extend([x for x in
                             getmembers(self.module, isclass)
                             if str(x[1].__module__) in str(self.module)
        ])

        self.parsed = [ParamTest(x) for x in self.members]
        self.helpers = [x for x in self.parsed
                        if x._type == 'func' and 'test' not in x.func_name
                        and 'contextmanager' != x.func_name
                        and 'system' != x.func_name
                        and 'uuid4' != x.func_name
                        and 'gettempdir' != x.func_name
                        and 'dedent' != x.func_name
                        and 'python_implementation' != x.func_name
                        and 'wrapper' != x.func_name
                        and 'patch' != x.func_name
        ]
        self.helpers.extend([x for x in self.parsed if x._type == 'func' and x.func_name[0] == '_'])
        self.functions = [x for x in self.parsed
                          if x._type == 'func' and 'test' in x.func_name
                          and 'unix_test' != x.func_name
                          and x.func_name[0] != '_'
                          and 'QARGS' not in x.source
                          and 'QHOME' not in x.source
                          and 'QLIC' not in x.source
                          and 'PYKX_UNLICENSED' not in x.source
                          and 'PYKX_LIGHT_LOAD' not in x.source
                          and 'memory_domain' not in x.source
                          and 'RawQConnection' not in x.source
                          and 'peach' not in x.source
        ]
        self.classes = [x for x in self.parsed if x._type == 'class']
        self.globals = []
        if self.module.__name__ in CARRY_GLOBALS:
            source = getsource(self.module)
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign):
                    self.globals.append(ModuleGlobal(
                        node.targets[0].id, ast.get_source_segment(source, node) + '\n'))


class ParseTestSuite:
    def __init__(self, modules):
        self.modules = [ParseModule(x) for x in modules]

    def make_tests(self): # noqa
        helpers = []
        licensed = []
        unlicensed = []
        ipc_licensed = []
        ipc_unlicensed = []
        embedded = []
        nep_licensed = []
        nep_unlicensed = []
        pandas_licensed = []

        for mod in self.modules:
            helpers.extend(mod.globals)
            helpers.extend(mod.helpers)
            for func in mod.functions:
                func.append_to_name('_' + mod.module_str[1:-1])
            for cls in mod.classes:
                for func in cls.functions:
                    if any(f'def {helper}' in func.source
                           for helper in ('create_splayed_table', 'reset_sp',
                                          'create_partitioned_table')):
                        func.source = func.source.replace('(self, ', '(', 1)
                        func.add_class_attrs(cls.class_attrs)
                        helpers.append(func)
                    else:
                        func.add_class_attrs(cls.class_attrs)
                        suffix = f'_{cls.class_name}_{mod.module_str[1:-1]}'
                        mod.functions.append(func.append_to_name(suffix))
            for func in mod.functions:
                added = False
                if any(['large' in x for x in func.decorators]) or any(['isolate' in x for x in func.decorators]): # noqa
                    continue
                if any(['.ipc' in x for x in func.decorators]):
                    if not any('(unlicensed_only' in x for x in func.decorators):
                        ipc_licensed.append(func)
                        embedded.append(func)
                        if any(['nep' in x for x in func.decorators]):
                            nep_licensed.append(func)
                        added = True
                    if not any('(licensed_only' in x for x in func.decorators):
                        ipc_unlicensed.append(func)
                        if any(['nep' in x for x in func.decorators]):
                            nep_unlicensed.append(func)
                        added = True

                if any(['.unlicensed' in x for x in func.decorators]):
                    if not any('(unlicensed_only' in x for x in func.decorators):
                        if any(['nep' in x for x in func.decorators]):
                            nep_licensed.append(func)
                        licensed.append(func)
                    if any(['nep' in x for x in func.decorators]):
                        nep_unlicensed.append(func)
                    unlicensed.append(func)
                    added = True
                if any(['pandas' in x for x in func.decorators]):
                    pandas_licensed.append(func)
                    added = True
                if not added:
                    if any(['nep' in x for x in func.decorators]):
                        nep_licensed.append(func)
                    embedded.append(func)

        result = [helpers,
                  licensed,
                  unlicensed,
                  ipc_licensed,
                  ipc_unlicensed,
                  embedded,
                  nep_licensed,
                  nep_unlicensed,
                  pandas_licensed
        ]
        self.check_collisions(result)
        return result

    @staticmethod
    def check_collisions(groups):
        # A repeated top-level name is silently rebound when the generated file is
        # imported, so only the last definition would ever run.
        helper_names = {x.func_name for x in groups[0]}
        collisions = []
        for group in groups[1:]:
            seen = set()
            for func in group:
                if func.func_name in seen or func.func_name in helper_names:
                    collisions.append(func.func_name)
                seen.add(func.func_name)
        if collisions:
            raise RuntimeError('Generated test names are not unique: '
                               + ', '.join(sorted(set(collisions))))


# Excluded on Windows: test_virtual_table (needs kx.pq), test_streamlit (unsupported),
# test_module (class based tests are not supported by this parser)
tests = ParseTestSuite([
    test_read,
    test_write,
    test_cast,
    test_console,
    test_exceptions,
    test_ipc,
    test_q_future,
    test_query,
    test_util,
    test_q_foreign,
    test_ctx,
    test_wrappers,
    test_q,
    test_pykx,
    test_toq,
    test_pandas_api,
    test_compress_encrypt,
    test_db,
    test_license,
    test_nbextension,
    test_splay,
    test_tick,
    test_config,
    test_pandas_agg,
    test_pandas_apply,
    test_pandas_map,
    test_pandas_replace,
    test_pandas_reset_index,
    test_pandas_set_index,
    test_random,
    test_register,
    test_reimport,
    test_remote,
    test_schema,
    test_sql,
    test_system,
    test_pandas_sorting
]).make_tests()
files = [
    ('win_tests/lic/licensed_tests.py', 1, ''),
    ('win_tests/unlic/unlicensed_tests.py', 2, '--unlicensed'),
    ('win_tests/ipc_lic/ipc_licensed_tests.py', 3, ''),
    ('win_tests/ipc_unlic/ipc_unlicensed_tests.py', 4, '--unlicensed'),
    ('win_tests/embedded/embedded_tests.py', 5, ''),
    ('win_tests/pandas_lic/pandas_licensed_tests.py', 8, '--pandas-api')
]

files.extend([
    ('win_tests/nep_lic/nep_licensed_tests.py', 6, '--pykxgc'),
    ('win_tests/nep_unlic/nep_unlicensed_tests.py', 7, '--unlicensed')]
)

if not os.path.exists('win_tests'):
    os.makedirs('win_tests')
    os.makedirs('win_tests/lic')
    os.makedirs('win_tests/unlic')
    os.makedirs('win_tests/ipc_lic')
    os.makedirs('win_tests/ipc_unlic')
    os.makedirs('win_tests/embedded')
    os.makedirs('win_tests/nep_lic')
    os.makedirs('win_tests/nep_unlic')
    os.makedirs('win_tests/pandas_lic')

test_libs = [
    'import pytest',
    'from contextlib import nullcontext, contextmanager',
    'from tempfile import TemporaryDirectory',
    'from pathlib import Path',
    'from platform import system',
    'import re',
    'import shutil',
    'import site',
    'from datetime import date, datetime, timedelta, time',
    'from functools import partial',
    'import math',
    'from sys import getrefcount',
    'from uuid import uuid4, UUID',
    'import sys',
    'import pickle',
    'import numpy as np',
    'from platform import python_implementation',
    'from io import StringIO',
    'import asyncio',
    'import random',
    'from tempfile import gettempdir',
    'from collections import abc',
    'import gc',
    'from textwrap import dedent',
    'from operator import index',
    'import pytz',
    'import pandas as pd',
    'import subprocess',
    'from packaging import version',
    'import uuid',
    'import itertools',
    'import operator',
    'import platform',
    'import toml',
    'import warnings',
    'from threading import Thread',
    'import statistics',
    'from abc import ABCMeta',
    'import base64',
    'from unittest.mock import patch'
]


for test_file in files:
    with open(test_file[0], 'wb') as f:
        f.write(bytes('import os\n'
                      f'os.environ["QARGS"] = "{test_file[2]}"\n'
                      'from time import sleep\n', 'utf-8'))

        if 'ipc' in test_file[0]:
            f.write(bytes(
                '''
original_QHOME = os.environ['QHOME']
from contextlib import closing, contextmanager
import signal
import socket
import subprocess
from platform import system\n
def random_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]\n
def q_proc(q_init):
    port = random_free_port()
    proc = subprocess.Popen(
        (lambda x: x.split() if system() != 'Windows' else x)(f'q -p {port}'),
        stdin=subprocess.PIPE,
        stdout=subprocess.sys.stdout,
        stderr=subprocess.sys.stderr,
        start_new_session=True,
        env={**os.environ, 'QHOME': original_QHOME},
    )
    proc.stdin.write(b'\\n'.join((*q_init, b'')))
    proc.stdin.flush()
    sleep(2) # Windows does not support the signal-based approach used here
    return port\n
import pykx as kx\n
''', 'utf-8'))
            f.write(bytes('q = kx.QConnection(port=q_proc([b""]))\n', 'utf-8'))
        else:
            f.write(bytes('import pykx as kx\nq = kx.q\n', 'utf-8'))

        f.write(bytes('\n'.join(test_libs), 'utf-8'))
        f.write(bytes('\ntest_arr = []\n\n', 'utf-8'))
        f.write(bytes('disposable_env_only = pytest.mark.skipif('
                      'os.environ.get("CI_DISPOSABLE_ENVIRONMENT", "").lower() '
                      'not in ("true", "1"),'
                      'reason="Test must be run in a disposable environment")\n\n', 'utf-8'
        ))
        f.write(
            bytes(
                'q_script_content = \'lambda:{\"this is a test lambda\"}; data:10?10; q:1b;\'\n'
                'k_script_content = \'lambda:{\"this is a test lambda\"}; data:10?10; q:0b;\'\n\n',
                'utf-8'
            )
        )

        f.write(bytes(
            '''
@pytest.fixture(autouse=True)
def _reset_process_state():
    # Linux runs every test in a forked process, Windows shares one so undo state leaks
    cwd = os.getcwd()
    yield
    os.chdir(cwd)
    db = kx.DB._instance
    if db is not None:
        kx.DB._instance = None
        try:
            kx.q('{![`.;();0b;x]}tables[]')
        except BaseException:
            pass

''', 'utf-8'))

        for test in tests[0]:
            f.write(bytes(test.source + '\n', 'utf-8'))
        for test in tests[test_file[1]]:
            f.write(bytes(test.source + '\n\n', 'utf-8'))

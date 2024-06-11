"""Interface to q contexts and scripts which define a namespace.

The context interface provides an easy to way access q contexts (also known as namespaces when at
the top level). For more information about contexts/namespaces in q please refer to
[Chapter 12 of Q for Mortals](https://code.kx.com/q4m3/12_Workspace_Organization/).
"""

from __future__ import annotations

from functools import lru_cache

from pathlib import Path
from typing import Tuple
from weakref import proxy

from . import Q
from .config import ignore_qhome, pykx_lib_dir, qhome
from .exceptions import PyKXException, QError
from .wrappers import Identity, SymbolicFunction


__all__ = [
    'CurrentDirectory',
    'QContext',
    'ZContext',
    'default_paths',
]


def __dir__():
    return __all__


# We resolve the module paths, so if the path of the current directory is actually used, and then
# the directory is changed, the current directory will no longer be searched for q modules. This
# class represents the current directory no matter where the program is currently executing, while
# still allowing the current directory to be removed from the search path by altering the module
# paths attribute.
class CurrentDirectory(type(Path())):
    """``pathlib.Path`` instance for the current directory regardless of directory changes."""
    def __init__(self):
        super().__init__()

    def resolve(self, *args, **kwargs):
        return self


default_paths = [
    CurrentDirectory(),
]

if not ignore_qhome:
    default_paths.append(qhome)
    default_paths.append(pykx_lib_dir)
else:
    default_paths.append(qhome.resolve(strict=True))
    default_paths.append((Path(__file__).parent/'lib').resolve(strict=True))


def _fully_qualified_name(name: str, parent: QContext) -> str:
    """Constructs the fully qualified name of a context given its name and parent."""
    names = [name]
    while parent is not None:
        names.append(parent._name)
        parent = parent._parent
    return '.'.join(reversed(names))


class QContext:
    def __init__(self, q: Q, name: str, parent: QContext, no_ctx=False):
        """Interface to a q context.

        Members of the context be accessed as if the `QContext` object was a dictionary, or by
        dotting off of the `QContext` object.

            q: The q instance in which the context exists.
            name: The name of the context.
            parent: The parent context as a `QContext`, or `None` in the case of the global context.
        """
        super().__setattr__('_q', q)
        super().__setattr__('_name', name)
        super().__setattr__('_parent', parent)
        super().__setattr__('_fqn', _fully_qualified_name(name, parent))
        super().__setattr__('_prev_ctx_stack', [])
        super().__setattr__('no_ctx', no_ctx)
        super().__setattr__('__getattr__', lru_cache(maxsize=None)(self.__getattr__))

    _unsupported_keys_with_msg = {
        'select': 'Usage of \'select\' function directly via \'q\' context not supported, please '
                  'consider using \'pykx.q.qsql.select\'',
        'exec': 'Usage of \'exec\' function directly via \'q\' context not supported, please '
                'consider using \'pykx.q.qsql.exec\'',
        'update': 'Usage of \'update\' function directly via \'q\' context not supported, please '
                  'consider using \'pykx.q.qsql.update\'',
        'delete': 'Usage of \'delete\' function directly via \'q\' context not supported, please '
                  'consider using \'pykx.q.qsql.delete\'',
    }

    @property
    def _context_keys(self) -> Tuple[str]:
        return (
            *(x for x in self._q._call('key', self._fqn, wait=True).py() if x),
            *(() if self._fqn else ('z',)),
        )

    def __iter__(self):
        return iter(self._context_keys)

    def _invalidate_cache(self):
        """Clears the cached context, forcing it to be reloaded the next time it is accessed."""
        if hasattr(self.__getattr__, 'cache_clear'):
            self.__getattr__.cache_clear()

    def __getattr__(self, key): # noqa
        if key == "__objclass__":
            raise AttributeError
        if key == 'z' and self._fqn == '':
            return ZContext(proxy(self))
        elif self._fqn in {'', '.q'} and key in self._unsupported_keys_with_msg:
            raise AttributeError(f'{key}: {self._unsupported_keys_with_msg[key]}')
        if self._fqn in {'', '.q'} and key in self._q.reserved_words:
            # Reserved words aren't actually part of the `.q` context dict
            if 'QConnection' in str(self._q._call):
                return lambda *args: self._q._call(key, *args, wait=True)
            else:
                return self._q._call(key, wait=True)
        if 'no_ctx=True' in str(self.__dict__['_q']) or self.no_ctx:
            raise PyKXException('Attempted to use context interface after disabling it.')
        fqn_with_key = f'{self._fqn}.{key}'
        try:
            attr = self._q._call(
                'k){x:. x;$[99h<@x;:`$"_pykx_fn_marker";99h~@x;if[` in!x;if[(::)~x`;:`$"_pykx_ctx_marker"]]]x}', # noqa: E501
                fqn_with_key,
                wait=True,
                skip_debug=True
            )
        except QError as err:
            if '_' in str(key):
                try:
                    return self.__getattr__(''.join([x for x in str(key) if x != '_']))
                except BaseException:
                    pass
            raise AttributeError(
                f"'{type(self).__module__}.{type(self).__name__}' object has no attribute {key!r}\n"
                f'QError: \'{err}'
            ) from None
        if attr.t == -11:
            maybe_marker = attr.py()
            if maybe_marker == '_pykx_fn_marker':
                return SymbolicFunction(fqn_with_key).with_execution_ctx(self._q)
            elif maybe_marker == '_pykx_ctx_marker':
                return QContext(self._q, key, proxy(self))
        return attr

    def __setattr__(self, key, value):
        if key in self._q.reserved_words or (self._fqn == '' and key in self._q.q):
            raise PyKXException('Cannot assign to reserved word or overwrite q namespace.')
        self._q._call('set', f'{self._fqn}.{key}', value, wait=True)
        self._invalidate_cache()

    def __delattr__(self, key):
        if self._fqn == '':
            raise PyKXException('Cannot delete from the global context.')
        self._q._call('{![x;();0b;enlist y]}', self._fqn, key, wait=True)
        self._invalidate_cache()

    __getitem__ = __getattr__
    __setitem__ = __setattr__
    __delitem__ = __delattr__

    def __enter__(self):
        # HACK: It's difficult to do a type-check here because that would introduce a cyclic import
        # error. Instead we use `hasattr(self._q, 'fileno')` as a proxy for the type check.
        if hasattr(self._q, 'fileno'): # IPC connections have a 'fileno' attribute.
            raise PyKXException('Context cannot be switch over IPC.')
        self._prev_ctx_stack.append(self._q._call(
            '{r:system"d";system "d ",string x;r}',
            self._fqn if self._fqn else '.',
            wait=True
        ))

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self._q._call('{system "d ",string x}', self._prev_ctx_stack.pop(), wait=True)

    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__} of {self._fqn} with ' \
               f'[{", ".join(self._context_keys)}]>'

    def __dir__(self):
        return sorted({*dir(super()), *self._context_keys})


class ZContext(QContext):
    """Special interface to handle the .z context.

    The .z context in q is not a normal context; it lacks a dictionary. To access it one must
    access its attributes directly.
    """
    _no_default = ('ac', 'bm', 'exit', 'pc', 'pd', 'ph', 'pi', 'pm',
                   'po', 'pp', 'pq', 'ps', 'pw', 'vs', 'wc', 'wo', 'ws', 'zd')

    _has_default = ('a', 'b', 'c', 'D', 'd', 'e', 'f', 'H', 'h', 'i', 'K', 'k',
                    'l', 'N', 'n', 'o', 'P', 'p', 'pg', 'q', 'T', 't', 'u', 'W', 'w',
                    'X', 'x')

    _unsupported_keys_with_msg = {
        's': 'cannot refer to self outside of a function, so .z.s is not '
             'exposed through the context interface.',
        'ex': '.z.ex is only available during a debugging session, so it is '
              'not exposed through the context interface.',
        'ey': '.z.ey is only available during a debugging session, so it is '
              'not exposed through the context interface.',
        'ts': '.z.ts is not exposed through the context interface because the '
              'main loop is inactive in PyKX.',
        'z': 'The q datetime type is deprecated, and so the datetime '
             'object .z.z is not exposed through the context interface.',
        'Z': 'The q datetime type is deprecated, and so the datetime '
             'object .z.Z is not exposed through the context interface.'
    }

    _context_keys = {*_has_default, *_no_default}

    def __init__(self, global_context: QContext):
        super().__init__(global_context._q, 'z', global_context)
        self._q('@[value;`.z.pg;{.z.pg:value}]')
        self._q('@[value;`.z.ps;{.z.ps:value}]')

    def __getattr__(self, key):
        if key in self._no_default:
            try:
                return self._q(f'.z.{key}', wait=True)
            except QError:
                pass
            return self._q('::', wait=True)
        elif key in self._has_default:
            return self._q(f'.z.{key}', wait=True)
        elif key in self._unsupported_keys_with_msg:
            raise AttributeError(f'{key}: {self._unsupported_keys_with_msg[key]}')
        raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self._no_default and (value is None or isinstance(value, Identity)):
            self._q._call(f'\\x .z.{key}', wait=True)
        else:
            super().__setattr__(key, value)

    def __delattr__(self, key):
        self.__setattr__(key, None)

    def __dir__(self):
        return sorted({
            *object.__dir__(self),
            *self._has_default,
            *self._no_default,
        })

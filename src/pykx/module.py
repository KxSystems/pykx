"""Loading KDB-X modules.

When `pykx.use` is called KDB-X will attempt to load a module with the given name.
"""
from typing import Optional

from .wrappers import CharVector, Dictionary, Identity, SymbolAtom
from .exceptions import QError

__all__ = [
    'QModule',
    'QNamespaceProxy',
    'SearchPath',
    'use'
]


def __dir__():
    return __all__


def _init(_q):
    global q
    q = _q


class QModule:
    """
    A loaded KDB-X q Module.

    Example loading the pq module and the pq.t submodule within KDB-X Python.
    ```python
    >>> import pykx as kx
    >>> pq = kx.use('kx.pq')
    >>> pq
    pykx.QModule(pykx.q('
    pq| pykx.Lambda
    op| pykx.Lambda
    rd| pykx.Lambda
    '))
    >>> pq.t = kx.use('kx.pq.t')
    >>> pq
    pykx.QModule(pykx.q('
    pq| pykx.Lambda
    op| pykx.Lambda
    rd| pykx.Lambda
    t | pykx.QModule(pykx.q('
        mkT| pykx.Lambda
        mkP| pykx.Lambda
        tt | pykx.Lambda
        mt | pykx.Lambda
        fv | pykx.Lambda
        '))
    '))
    ```
    """
    def __init__(self, name, qns, qkeys, qdict):
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_qns', qns)
        object.__setattr__(self, '_qkeys', qkeys)
        object.__setattr__(self, '_qdict', qdict)
        object.__setattr__(self, '_submods', {})

    def __dir__(self):
        return list(self._qkeys)

    def __getattribute__(self, name):
        if (name in object.__getattribute__(self, "_qkeys")
                and name not in object.__getattribute__(self, '_submods').keys()):
            ns = object.__getattribute__(self, "_qns")
            qd = object.__getattribute__(self, "_qdict")
            return _wrap_if_namespace(_fetch(ns, name, fallback=qd), ns, name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if (name in object.__getattribute__(self, "_qkeys")
                and name not in object.__getattribute__(self, '_submods').keys()):
            ns = object.__getattribute__(self, "_qns")
            q('set', ns + '.' + name, value)
        else:
            if isinstance(value, QModule):
                sub_m = object.__getattribute__(self, '_submods')
                sub_m[name] = value
                object.__setattr__(self, '_submods', sub_m)
                keys = [k.py() for k in object.__getattribute__(self, '_qdict').keys() if k.py()]
                keys.extend(object.__getattribute__(self, '_submods').keys())
                object.__setattr__(
                    self,
                    '_qkeys',
                    frozenset(
                        keys
                    )
                )
            super().__setattr__(name, value)

    def __repr__(self):
        res = 'pykx.QModule(pykx.q(\'\n'
        qd = object.__getattribute__(self, "_qdict")
        all_keys = [x.py() for x in list(qd.keys())]\
            + list(object.__getattribute__(self, '_submods').keys())
        if all_keys == []:
            return f'pykx.QModule(pykx.q(\'{qd}\'))'
        max_len = max([len(x) for x in all_keys])

        for k in all_keys:
            if k in qd.keys():
                res += k + (' ' * (max_len - len(k))) + '| ' +\
                    'pykx.' + str(type(qd[k])).split("'")[-2].split('.')[-1] + '\n'
            else:
                res += k + (' ' * (max_len - len(k))) + '| pykx.QModule(pykx.q(\'\n' +\
                    '\n'.join([
                        ' ' * (max_len + 2) + x for x in
                        object.__getattribute__(self, '_submods')[k].__repr__().split('\n')[1:-1]
                    ]) + '\n' + ' ' * (max_len + 2) + '\'))\n'
        return res + '\'))'


def _is_q_namespace(value: Dictionary) -> bool:
    # Q namespaces, when fetched as dicts, have an empty-symbol key mapping to ::.
    try:
        return any(k.py() == "" for k in value.keys())
    except Exception:
        return False


def _fetch(qns: str, key: str, fallback: Optional[Dictionary] = None):
    # Fetch a key from a q namespace.

    # @[ns; key] returns :: when key is a sub-namespace (not a plain value).
    # In that case we fall back to the dict returned by `use`, which already
    # contains the full sub-namespace contents.
    result = q('@', qns, key)
    if isinstance(result, Identity) and fallback is not None:
        result = fallback[SymbolAtom(key)]
    return result


def _wrap_if_namespace(result, qns: str, key: str):
    # Wrap q namespace dicts in a QNamespaceProxy; leave plain data dicts alone.
    if isinstance(result, Dictionary) and _is_q_namespace(result):
        return QNamespaceProxy(f"{qns}.{key}", result)
    return result


class QNamespaceProxy:
    # Proxy for a nested q namespace returned as a dictionary attribute.

    def __init__(self, qns: str, value: Dictionary):
        object.__setattr__(self, "_qns", qns)
        object.__setattr__(self, "_value", value)

    def _qkeys(self):
        val = object.__getattribute__(self, "_value")
        return frozenset(k.py() for k in val.keys() if k.py())

    def __dir__(self):
        return list(self._qkeys())

    def __getattribute__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        val = object.__getattribute__(self, "_value")
        if name in (k.py() for k in val.keys() if k.py()):
            qns = object.__getattribute__(self, "_qns")
            return _wrap_if_namespace(val[SymbolAtom(name)], qns, name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if not name.startswith("_") and name in self._qkeys():
            qns = object.__getattribute__(self, "_qns")
            q('set', qns + '.' + name, value)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        qns = object.__getattribute__(self, "_qns")
        val = object.__getattribute__(self, "_value")
        return f"<QNamespaceProxy {qns!r}>\n{val}"


def _resolve_qns(module_name) -> str:
    # Look up the q namespace for this module in .Q.m.M after loading.

    module_map = q(".Q.m.M").py()
    for (path_bytes, sub_name), info in module_map.items():
        end_path = path_bytes.decode('utf-8').split('/')[-1].split('\\')[-1]
        mod_name = module_name.split('.')[-1]
        if end_path == mod_name or end_path == mod_name + '.k'\
           or sub_name == module_name.split(':')[-1]:
            ns = info.get('m', '')
            if ns:
                return ns
    return ''


def use(module_name: str, export_argument: object = None) -> QModule:
    """
    Function for loading KDB-X modules.

    Parameters:
        module_name: The name of the KDB-X module to load as a string.
        export_argument: Optional argument to pass to the module export function. Default None.

    Returns:
        A loaded QModule instance.

    Examples:

    Example loading the pq module and the pq.t submodule within KDB-X Python.
    ```python
    >>> import pykx as kx
    >>> from pathlib import Path
    >>> pq = kx.module.use('kx.pq')
    >>> pq.t = kx.module.use('kx.pq.t')
    >>> tab = pq.pq(Path('../types.parquet'))
    >>> tab
    pykx.VirtualTable(pykx.q('`T!`f`m`t!(k){[f;t;c;b;a;v]g:$[s:-1h=@b;0;#b];(bf;b1):$[g;df[t;0,0b;b];(::;()..'))
    >>> tab.select()
    pykx.Table(pykx.q('
    col0 col1 col2
    ---------------
    1    1    "asd"
    1    2    "bsd"
    0    3    "csd"
    1    4    "asd"
    0    5    "bsd"
    0    6    "csd"
    1    7    "asd"
    0    8    "bsd"
    0    9    "csd"
    '))
    ```
    """
    mod = q._get_module(module_name)
    if mod is not None:
        return mod
    try:
        qdict = q.use(module_name if export_argument is None else [module_name, export_argument])
        qns = _resolve_qns(module_name)
    except QError as e:
        raise ImportError(f"Failed to load {module_name}") from e

    qkeys = frozenset(k.py() for k in qdict.keys() if k.py())
    module = QModule(module_name, qns, qkeys, qdict)
    # Seed __dict__ with the exported names so IPython/jedi can discover
    # them via static inspection. __getattribute__ above ensures live q
    # values are always fetched on actual attribute access.
    module.__dict__.update(dict.fromkeys(qkeys))

    if q._get_module('.'.join(module_name.split('.')[:-1])) is not None:
        q._get_module('.'.join(module_name.split('.')[:-1])).__setattr__(
            module_name.split('.')[-1],
            module
        )
    q._add_module(module_name, module)
    return module


def _SP(paths: Optional[list[str]] = None):

    if paths is None:
        return [b.decode() for b in q('.Q.m.SP').py()]
    else:
        q('set', '.Q.m.SP', [CharVector(s.encode()) for s in paths])


def SearchPath(
    entry: Optional[str] = None,
    *,
    add: bool = False,
    remove: bool = False,
    prepend: bool = False,
    strict: bool = False,
    allow_duplicates: bool = False,
) -> Optional[list[str]]:
    """
    Manage the module SearchPath.

    Parameters:
        entry: The path string to add or remove. If omitted, the current
            SearchPath is returned.
        add: If True, add the entry to the SearchPath. Default False.
        remove: If True, remove all occurrences of the entry from the SearchPath. Default False.
        prepend: If True, insert the entry at the front rather than the end. Default False.
        strict: If True, raise a ValueError when attempting to remove an
            entry that is not present. Default False.
        allow_duplicates: If True, allow the same entry to appear more than
            once in the SearchPath. Default False.

    Examples:

    Retrieve the current SearchPath:

    ```python
    >>> kx.module.SearchPath()
    []
    ```

    Append and prepend entries:

    ```python
    >>> kx.module.SearchPath("/home/user/.kx/mod", add=True)
    >>> kx.module.SearchPath("/home/user/mod", add=True)
    >>> kx.module.SearchPath("/other/mod", add=True, prepend=True)
    >>> kx.module.SearchPath()
    ['/other/mod', '/home/user/.kx/mod', '/home/user/mod']
    ```

    Remove an entry:

    ```python
    >>> kx.module.SearchPath("/other/mod", remove=True)
    >>> kx.module.SearchPath()
    ['/home/user/.kx/mod', '/home/user/mod']
    ```
    """
    sp = _SP()
    if entry is None:
        return sp
    if add:
        if not allow_duplicates and entry in sp:
            raise ValueError(
                f"{entry!r} already on SearchPath. Use allow_duplicates=True to override")
        if prepend:
            sp.insert(0, entry)
        else:
            sp.append(entry)
        _SP(sp)
        return None
    if remove:
        if entry not in sp:
            if strict:
                raise ValueError(f"{entry!r} not found in SearchPath.")
            return None
        sp = [p for p in sp if p != entry]
        _SP(sp)
        return None

    raise ValueError("Specify add=True or remove=True, or call with no arguments to retrieve.")

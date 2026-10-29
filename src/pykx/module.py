"""Loading KDB-X modules.

When `pykx.use` is called KDB-X will attempt to load a module with the given name.
"""
from typing import Optional, Union

from .wrappers import CharVector, Dictionary, SymbolAtom
from .config import kdbx
from .exceptions import PyKXException, QError

__all__ = [
    'QModule',
    'QNamespaceProxy',
    'SearchPath',
    'unwrap',
    'use',
    'wrap'
]


def __dir__():
    return __all__


def _init(_q):
    global q
    q = _q


def _require_kdbx():
    if not kdbx():
        raise PyKXException(
            'KDB-X modules require KDB-X 5.0 or later, the q in use reports '
            f'version {q(".z.K").py()} ({q(".z.k").py()})'
        )


def _member(mod, key: str):
    # Resolve one exported name to the object attribute access should hand back.
    value = object.__getattribute__(mod, '_qdict')[SymbolAtom(key)]
    if isinstance(value, Dictionary) and _is_q_namespace(value):
        qns = object.__getattribute__(mod, '_qns')
        return QNamespaceProxy(f'{qns}.{key}' if qns else '', value)
    return value


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
    >>> len(pq)
    3
    >>> pq['pq']                    # a key reaches the same export as the attribute
    pykx.Lambda(pykx.q('locked'))
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

    Assigning to an exported name writes to the module's private q namespace, which is how
    a module's own function or configuration variable is patched from Python:

    ```python
    >>> greet = kx.use('kx.greet')
    >>> greet.hello = kx.q('{"Hi, ",x,"!"}')
    >>> print(greet.hello(b'KDB-X'))
    Hi, KDB-X!
    ```

    Where no q namespace is known, as for a dictionary presented through
    [`wrap`][pykx.module.wrap], assignment raises `#!python TypeError` rather than writing
    somewhere else.
    """
    def __init__(self, qns, qdict):
        object.__setattr__(self, '_qns', qns)
        object.__setattr__(self, '_qdict', qdict)
        object.__setattr__(self, '_names', [k.py() for k in qdict.keys() if k.py()])
        object.__setattr__(self, '_submods', {})

    def __dir__(self):
        return list(object.__getattribute__(self, '_names'))

    def __getattribute__(self, name):
        names = object.__getattribute__(self, '_names')
        if (name.endswith('_') and not name.startswith('_')
                and name not in names and name[:-1] in names):
            name = name[:-1]
        if name in names and name not in object.__getattribute__(self, '_submods'):
            return _member(self, name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        names = object.__getattribute__(self, '_names')
        submods = object.__getattribute__(self, '_submods')
        if isinstance(value, QModule):
            submods[name] = value
            if name not in names:
                names.append(name)
        elif name in names and name not in submods:
            ns = object.__getattribute__(self, '_qns')
            if not ns:
                raise TypeError(
                    f"Cannot assign to {name!r}: this object has no q namespace to write to. "
                    "Use the setter the module provides, or assign into the dictionary from "
                    "kx.module.unwrap."
                )
            q('set', ns + '.' + name, value)
            # Mirror the write into this object's view of `export`, so reading the name back
            # returns what was just written rather than the value the module exported.
            object.__getattribute__(self, '_qdict')[SymbolAtom(name)] = value
            return
        super().__setattr__(name, value)

    def __getitem__(self, key):
        if isinstance(key, SymbolAtom):
            key = key.py()
        if not isinstance(key, str):
            raise TypeError(f'Export name must be a string, received: {type(key)}')
        submods = object.__getattribute__(self, '_submods')
        if key in submods:
            return submods[key]
        if key not in object.__getattribute__(self, '_names'):
            raise KeyError(key)
        return _member(self, key)

    def __contains__(self, key):
        if isinstance(key, SymbolAtom):
            key = key.py()
        return key in object.__getattribute__(self, '_names')

    def __len__(self):
        return len(object.__getattribute__(self, '_names'))

    def __iter__(self):
        return (self[name] for name in list(object.__getattribute__(self, '_names')))

    def __repr__(self):
        cls = 'pykx.' + type(self).__name__
        names = list(object.__getattribute__(self, '_names'))
        submods = object.__getattribute__(self, '_submods')
        if not names:
            return f"{cls}(pykx.q('{object.__getattribute__(self, '_qdict')}'))"
        width = max(len(x) for x in names)
        res = cls + "(pykx.q('\n"
        for k in names:
            pad = ' ' * (width - len(k))
            if k not in submods:
                res += f"{k}{pad}| pykx.{type(_member(self, k)).__name__}\n"
            else:
                nested = '\n'.join(' ' * (width + 2) + x
                                   for x in repr(submods[k]).split('\n')[1:-1])
                res += (f"{k}{pad}| pykx.{type(submods[k]).__name__}(pykx.q('\n"
                        + nested + '\n' + ' ' * (width + 2) + "'))\n")
        return res + "'))"


class QNamespaceProxy(QModule):
    """
    A q namespace nested inside a loaded module.

    q marks a namespace with an empty-symbol key whose value is `#!q ::`. A member carrying
    that entry is presented as one of these rather than as a
    [`pykx.Dictionary`][pykx.Dictionary], so its own members are reachable by attribute and
    by key, and a write reaches the q namespace behind it where the member really is one.

    ```python
    >>> kafka = kx.use('kx.kafka')
    >>> type(kafka.OFFSET)
    <class 'pykx.module.QNamespaceProxy'>
    >>> kafka.OFFSET['BEGINNING']
    pykx.LongAtom(pykx.q('-2'))
    >>> kafka.OFFSET.BEGINNING
    pykx.LongAtom(pykx.q('-2'))
    ```
    """

    def __init__(self, qns: str, value: Dictionary):
        super().__init__(qns, value)


def _is_q_namespace(value: Dictionary) -> bool:
    # Q namespaces, when fetched as dicts, have an empty-symbol key mapping to ::.
    try:
        return any(k.py() == "" for k in value.keys())
    except Exception:
        return False


def _resolve_qns(module_name: str) -> str:
    # Look up the q namespace the loader built for this module in `.Q.m.M`.
    base, _, sub = module_name.partition(':')
    wanted = [part for part in base.split('.') if part]
    if not wanted:
        return ''
    for (path, sub_name), info in q('.Q.m.M').py().items():
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        if isinstance(sub_name, bytes):
            sub_name = sub_name.decode('utf-8')
        if sub_name != sub:
            continue
        segments = [s for s in path.replace('\\', '/').split('/') if s]
        if not segments:
            continue
        stem = segments[-1].split('.')[0]
        segments = segments[:-1] if stem == 'init' else segments[:-1] + [stem]
        if segments[-len(wanted):] != wanted:
            continue
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
    _require_kdbx()
    mod = q._get_module(module_name)
    if mod is not None:
        return mod
    try:
        qdict = q.use(module_name if export_argument is None else [module_name, export_argument])
        qns = _resolve_qns(module_name)
    except QError as e:
        raise ImportError(f"Failed to load {module_name}: {e}") from e

    module = QModule(qns, qdict)
    names = object.__getattribute__(module, '_names')
    module.__dict__.update(dict.fromkeys(names))

    if q._get_module('.'.join(module_name.split('.')[:-1])) is not None:
        q._get_module('.'.join(module_name.split('.')[:-1])).__setattr__(
            module_name.split('.')[-1],
            module
        )
    for key in list(names):
        child = q._get_module(module_name + '.' + key)
        if child is not None:
            module.__setattr__(key, child)
    q._add_module(module_name, module)
    return module


def wrap(qdict: Dictionary) -> QModule:
    """
    Present a q dictionary of names as an attribute-accessible object.

    Some modules expose functionality through a function which returns a dictionary rather
    than through their `#!q export` variable. The KDB-X Logging module is one: its
    `#!python createLog` builds an independent logger and returns it as a dictionary, whose
    members would otherwise have to be reached by key. `#!python wrap` presents such a
    dictionary the way a loaded module is presented, with attribute access and
    `#!python dir` support.

    The result is not a loaded module and has no q namespace behind it, so attribute
    assignment raises `#!python TypeError`. Use whatever setter the dictionary provides.

    Parameters:
        qdict: The q dictionary to wrap.

    Returns:
        A `QModule` whose attributes are the dictionary's keys.

    Raises:
        TypeError: If `qdict` is not a q dictionary.

    Example:

    Wrap a logger returned by the Logging module's `createLog`:

    ```python
    >>> import pykx as kx
    >>> logger = kx.use('kx.log')
    >>> log = kx.module.wrap(logger.createLog())
    >>> dir(log)
    ['add', 'addfmt', 'debug', 'error', 'fatal', 'fmts', 'getfmt', 'getlvl', 'info',
     'remove', 'setfmt', 'setlvl', 'sinks', 'trace', 'warn']
    >>> log.info(b'Hello World!')
    2026.09.04D10:45:45.780488685 info PID[320523] HOST[localhost] Hello World!
    >>> log.setlvl('trace')
    >>> log.getfmt()
    pykx.SymbolAtom(pykx.q('`basic'))
    ```

    Not every entry is callable. Some are symbols naming a q variable rather than a
    function, and q dereferences those when they are applied. Python has no equivalent
    overload, so read them with `#!python kx.q.get`:

    ```python
    >>> log.getlvl
    pykx.SymbolAtom(pykx.q('`.m.kx.0log.log.1.lvl'))
    >>> kx.q.get(log.getlvl)
    pykx.SymbolAtom(pykx.q('`trace'))
    ```
    """
    if not isinstance(qdict, Dictionary):
        raise TypeError(
            f'Supplied object must be a q dictionary, received: {type(qdict)}'
        )
    module = QModule('', qdict)
    module.__dict__.update(dict.fromkeys(object.__getattribute__(module, '_names')))
    return module


def unwrap(module: Union[QModule, QNamespaceProxy]) -> Dictionary:
    """
    Recover the q dictionary behind a `QModule` or a `QNamespaceProxy`.

    This is the inverse of [`wrap`][pykx.module.wrap]. Use it when a member has been
    presented as a [`QNamespaceProxy`][pykx.module.QNamespaceProxy] and you need the
    dictionary itself.

    A q namespace is identified by an empty-symbol key whose value is `#!q ::`. A dictionary
    can carry that same entry deliberately, as a placeholder which keeps the values a general
    list and stops q promoting them to a typed vector. Such a dictionary is indistinguishable
    from a namespace, so it is presented as a `QNamespaceProxy`.

    Parameters:
        module: The `QModule` or `QNamespaceProxy` to unwrap.

    Returns:
        The underlying q dictionary. For a loaded module this is its `#!q export` dictionary.

    Raises:
        TypeError: If `module` is neither a `QModule` nor a `QNamespaceProxy`.

    Example:

    A module exporting a dictionary that uses the placeholder entry:

    ```python
    >>> import pykx as kx
    >>> mod = kx.use('kx.mymodule')
    >>> mod.opts                     # looks like a namespace, so presented as one
    pykx.QNamespaceProxy(pykx.q('
    headers| pykx.Dictionary
    '))
    >>> opts = kx.module.unwrap(mod.opts)
    >>> opts['headers']
    pykx.Dictionary(pykx.q('
    Accept| "a"
    X     | "b"
    '))
    ```

    A nested member reached by attribute arrives as a
    [`kx.module.QNamespaceProxy`][pykx.module.QNamespaceProxy], which `#!python unwrap`
    also accepts:

    ```python
    >>> kafka = kx.use('kx.kafka')
    >>> type(kafka.OFFSET)
    <class 'pykx.module.QNamespaceProxy'>
    >>> kx.module.unwrap(kafka.OFFSET)['BEGINNING']
    pykx.LongAtom(pykx.q('-2'))
    ```
    """
    # QNamespaceProxy is a QModule, so one branch covers both.
    if isinstance(module, QModule):
        return object.__getattribute__(module, '_qdict')
    raise TypeError(
        f'Supplied object must be a QModule or QNamespaceProxy, received: {type(module)}'
    )


def _SP(paths: Optional[list[str]] = None):
    _require_kdbx()
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

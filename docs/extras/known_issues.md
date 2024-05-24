# Known Issues

- Enabling the NEP-49 NumPy allocators will often segfault when running in a multiprocess setting.
- The timeout value is always set to `0` when using `PYKX_Q_LOCK`.
- Enabling `PYKX_ALLOCATOR` and using PyArrow tables can cause segfaults in certain scenarios.
- `kurl` functions require their `options` dictionary to have mixed type values. Add a `None` value to bypass: `{'': None, ...}` (See [docs](https://code.kx.com/insights/core/kurl/kurl.html))
- `None` and `pykx.Identity(pykx.q('::'))` do not pass through to single argument Python functions set under q. See [here](../pykx-under-q/known_issues.md#default-parameter).

    ```python
    >>> def func(n=2):
    ...     return n
    ... 
    >>> kx.q('func', None)
    pykx.LongAtom(pykx.q('2'))
    >>> kx.q('func', kx.q('::'))
    pykx.LongAtom(pykx.q('2'))
    ```

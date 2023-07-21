# Known Issues

- Enabling the NEP-49 numpy allocators will often segfault when running in a multiprocess setting.
- The timeout value is always set to `0` when using `PYKX_Q_LOCK`.
- `pykx.q` fails to load under Windows.
- Enabling `PYKX_ALLOCATOR` and using PyArrow tables can cause segfaults in certain scenarios.
- `kurl` functions require their `options` dictionary to have mixed type values. Add a `None` value to bypass: `{'': None, ...}` (See [docs](https://code.kx.com/insights/core/kurl/kurl.html))

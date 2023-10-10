# FAQ

## How to work around the `'cores` licensing error?

```
>>> import pykx as kx
<frozen importlib._bootstrap>:228: PyKXWarning: Failed to initialize embedded q; falling back to unlicensed mode, which has limited functionality. Refer to https://code.kx.com/pykx/user-guide/advanced/modes.html for more information. Captured output from initialization attempt:
    '2022.09.15T10:32:13.419 license error: cores
```

This error indicates your license is limited to a given number of cores but PyKX tried to use more cores than the license allows.

- On Linux you can use `taskset` to limit the number of cores used by the python process and likewise PyKX and EmbeddedQ:
```
# Example to limit python to the 4 first cores on a 8 cores CPU
$ taskset -c 0-3 python
```

- You can also do this in python before importing PyKX (Linux only):
```
>>> import os
>>> os.sched_setaffinity(0, [0, 1, 2, 3])
>>> import pykx as kx
>>> kx.q('til 10')
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
```

- On Windows you can use the `start` command with its `/affinity` argument (see: `> help start`):
```
> start /affinity f python
```
(above, 0xf = 00001111b, so the python process will only use the four cores for which the mask bits are equal to 1)

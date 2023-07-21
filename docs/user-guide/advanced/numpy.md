# Numpy Integration

PyKX is designed for advanced integration with Numpy. 
This integration is built on three pillars: [NEP-49](https://numpy.org/neps/nep-0049.html), the Numpy [array interface](https://numpy.org/doc/stable/reference/arrays.interface.html), and [universal functions](https://numpy.org/doc/stable/reference/ufuncs.html). 

## Support for NEP-49 and 0-copy data transfer from Numpy to q (when possible)

To use NEP-49 and benefit from 0-copy data transfers from Numpy to q, the following environment variable needs to be set before importing PyKX: `PYKX_ALLOCATOR=1`.
Once enabled, PyKX leverages NEP-49 to replace Numpy's memory allocator with the q/k memory allocator. This makes Numpy arrays directly available to q (by passing only a pointer) and accelerates the conversion time from Numpy arrays to q significantly.

Without NEP-49 (`PYKX_ALLOCATOR=0`):
```python
In [1]: arr = np.random.rand(1000000)
In [2]: %timeit kx.toq(arr)
421 µs ± 9.42 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

With NEP-49 (`PYKX_ALLOCATOR=1`):
```python
In [1]: arr = np.random.rand(1000000)
In [2]: %timeit kx.toq(arr)
5.4 µs ± 150 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
```

In the example above, transferring a Numpy array of one million `float64` numbers runs 80x faster with NEP-49 enabled (`PYKX_ALLOCATOR=1`).

!!! Note

    With NEP-49 enabled, 0-copy data transfer will happen for the following target q types: booleans (`1h`), bytes (`4h`), shorts (`5h`), integers (`6h`), longs (`7h`), reals (`8h`), floats (`9h`), timespan (`16h`), minutes (`17h`), seconds (`18h`) and times (`19h`).
    
    A data copy will happen for the following target q types: guids (`2h`), chars (`10h`), symbols (`11h`), timestamps (`12h`), months (`13h`) and dates (`14h`).

## Support for Numpy array interface and universal functions on pykx/q vectors

PyKX vectors implement the Numpy array interface and are compatible with universal functions. This means all those Numpy functions (and more) can be used directly on PyKX vectors and hence, on q vectors.
The following are a number of useful links relating to universal functions which can be leveraged using this:

* [Numpy universal functions](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs)
* [Scipy universal functions](https://docs.scipy.org/doc/scipy/reference/special.html#available-functions)
* [CuPy universal functions (GPU)](https://docs.cupy.dev/en/stable/reference/ufunc.html) (q vectors will be transferred to GPU)
* [Custom universal functions with Numba](https://numba.readthedocs.io/en/stable/user/vectorize.html)
* [Custom universal functions with C++ and the Boost library](https://www.boost.org/doc/libs/1_65_1/libs/python/doc/html/numpy/tutorial/ufunc.html)

## A little experiment with universal functions

Let's take the Greater Common Divisor problem (GCD) to compare different implementations using Python, q, and some custom universal functions.
The script below implements 5 different solutions for GCD calculation:

* `qgcd`: Naive q implementation, process one pair of integers at a time.
* `qgcd2`: q vectorized implementation.
* `gcd`: Naive python implementation, process one pair of integers at a time.
* `gcd2`: Custom `ufunc` vectorized and compiled JIT with Numba.
* `gcd3`: Custom `ufunc` vectorized, parallelized on all cores and compiled JIT with Numba.

```python
import numpy as np
import pykx as kx
from numba import vectorize, int64


# qgcd: q GCD one pair at a time
kx.q('qgcd: {while[y; y2: y; y: x mod y; x: y2]; x}')
# qgcd2: q GCD vectorized
qgcd2 = kx.q('{while[any y; x2: y; y2: x mod y; x: ?[y>0; x2; x]; y: ?[y>0; y2; y]]; x}')


def gcd(a, b):
    """Calculate the greatest common divisor of a and b (pure python, one pair at a time)"""
    while b:
        a, b = b, a % b
    return a


@vectorize([int64(int64, int64)])
def gcd2(a, b):
    """Calculate the greatest common divisor of a and b (Numba JIT compilation, vectorized)"""
    while b:
        a, b = b, a % b
    return a


@vectorize([int64(int64, int64)], target='parallel')
def gcd3(a, b):
    """Calculate the greatest common divisor of a and b (Numba JIT compilation, vectorized, multicore parallelism)"""
    while b:
        a, b = b, a % b
    return a


a = np.random.randint(0, 1000, size=100000)
b = np.random.randint(0, 1000, size=100000)

qa = kx.toq(a)
qb = kx.toq(b)
```

We can use IPython to load this script and benchmark the different implementations with `%timeit`. We will also compare to `np.gcd`, the Numpy ufunc for GCD calculation.

```bash
$ PYKX_ALLOCATOR=1 ipython -i test_numpy_ufuncs.py
```

```python
# q naive GCD
In [1]: %timeit kx.q('{{qgcd[first x; last x]} each flip (x;y)}', a, b)
297 ms ± 3.29 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# q vectorized GCD
In [2]: %timeit qgcd2(a, b)
19.9 ms ± 306 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Python naive GCD
In [3]: %timeit [gcd(x, y) for x, y in zip(a,b)]
34 ms ± 2.68 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Numpy vectorized GCD ufunc
In [4]: %timeit np.gcd(a, b)
2.86 ms ± 44.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Numpy vectorized GCD ufunc, works as well on pykx vectors
In [5]: %timeit np.gcd(qa, qb)
3.03 ms ± 219 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Numba vectorized GCD ufunc
In [6]: %timeit gcd2(a, b)
3.04 ms ± 16 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Numba vectorized GCD ufunc, works as well on pykx vectors
In [7]: %timeit gcd2(qa, qb)
3.13 ms ± 17.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Numba multicore GCD ufunc
In [8]: %timeit gcd3(a, b)
748 µs ± 80.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# Numba multicore GCD ufunc, works as well on pykx vectors
In [9]: %timeit gcd3(qa, qb)
792 µs ± 27.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

# Quickstart

This quickstart guide provides first time users with instructions for installing this library and make use of the functionality it contains for the first time.

## Prerequisites

To complete the quickstart guide below you will need to have completed the following:

- [Install the PyKX library and a license](installing.md).

## How to import PyKX

To access PyKX and it's functionality import it within your Python code using the following syntax

```python
import pykx as kx
```

The use of the shortened name `kx` is intended to provide a terse convention for interacting with methods and objects from this library.

## How to generate PyKX objects

The generation of PyKX objects is supported pricipally in two ways

1. Execution of q code to create these entities
2. Conversion of Python objects to analagous PyKX objects


### Creation of PyKX objects using inbuilt PyKX functions

Generation of PyKX objects using `pykx` helper functions

```python
>>> kx.random.random([3, 4], 10.0)
pykx.List(pykx.q('
4.976492 4.087545 4.49731   0.1392076
7.148779 1.946509 0.9059026 6.203014 
9.326316 2.747066 0.5752516 2.560658 
'))

>>> kx.Table(data = {'x': kx.random.random(10, 10.0), 'x1': kx.random.random(10, ['a', 'b', 'c'])})
pykx.Table(pykx.q('
x         x1
------------
0.8123546 a 
9.367503  a 
2.782122  c 
2.392341  a 
1.508133  b 
'))
```

### Creation of PyKX objects using q

Generation of PyKX objects using q can be completed through calling `kx.q`

```python
>>> kx.q('10 20 30')
pykx.LongVector(pykx.q('10 20 30'))

>>> kx.q('([]5?1f;5?`4;5?0Ng)')
pykx.Table(pykx.q('
x         x1   x2                                  
---------------------------------------------------
0.439081  ncej 8c6b8b64-6815-6084-0a3e-178401251b68
0.5759051 jogn 5ae7962d-49f2-404d-5aec-f7c8abbae288
0.5919004 ciha 5a580fb6-656b-5e69-d445-417ebfe71994
0.8481567 hkpb ddb87915-b672-2c32-a6cf-296061671e9d
0.389056  aeaj 580d8c87-e557-0db1-3a19-cb3a44d623b1
'))
```

### Creation of PyKX objects from Python data types

Generation of PyKX objects from Python, Numpy, Pandas and PyArrow objects can be completed as follows using the `kx.toq` method.

```python
>>> pylist = [10, 20, 30]
>>> qlist = kx.toq(pylist)
>>> qlist
pykx.LongVector(pykx.q('10 20 30'))

>>> nplist = np.arange(0, 10, 2)
>>> qlist = kx.toq(nplist)
>>> qlist
pykx.LongVector(pykx.q('0 2 4 6 8'))

>>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
>>> df
   col1  col2
0     1     3
1     2     4
>>> qtable = kx.toq(df)
pykx.Table(pykx.q('
col1 col2
---------
1    3   
2    4   
'))

>>> patab = pa.Table.from_pandas(df)
>>> patab
pyarrow.Table
col1: int64
col2: int64
>>> qtable = kx.toq(patab)
>>> qtable
pykx.Table(pykx.q('
col1 col2
---------
1    3   
2    4   
'))
```

## Interacting with PyKX Objects

PyKX objects can be interacted with in a variety of ways, through indexing using Pythonic syntax, passing PyKX objects to q/numpy functions, querying via SQL/qSQL syntax or through the use of q functionality via the context interface. Each of these is described in more depth throughout this documentation but examples of each are provided here


* Create a PyKX list and interact with the list using indexing and slices.

    ```python
    >>> qarray = kx.toq([random() for _ in range(10)], kx.FloatVector)
    >>> qarray
    pykx.FloatVector(pykx.q('0.391543 0.08123546 0.9367503 0.2782122 0.2392341 0.1508133 0.1567317 0.9785 ..'))
    >>> qarray[1]
    pykx.FloatAtom(pykx.q('0.08123546'))
    >>> qarray[1:4]
    pykx.FloatVector(pykx.q('0.08123546 0.9367503 0.2782122'))
    ```

* Assign objects to PyKX lists

    ```python
    >>> qarray = kx.random.random(3, 10.0, seed=10)
    pykx.FloatVector(pykx.q('0.891041 8.345194 3.621949'))
    >>> qarray[1] = 0.1
    >>> qarray
    pykx.FloatVector(pykx.q('0.891041 0.1 3.621949'))
    ```

* Create a PyKX table and manipulate using Pythonic syntax

    ```python
    >>> qtable = kx.Table(
        data={
            'x': [random() for _ in range(100)], 
            'x1': [random() * 5 for _ in range(100)],
            'x2': [['a', 'b', 'c'][randint(0, 2)] for _ in range(100)]
        }
    )
    >>> qtable
    pykx.Table(pykx.q('
    x         x1         x2
    -----------------------
    0.3550381 1.185644   c 
    0.3615143 2.835405   a 
    0.9089531 2.134588   b 
    0.2062569 3.852387   a 
    0.481821  0.07970141 a 
    0.2065625 1.786519   a 
    0.5229178 0.1273692  c 
    0.3338806 3.440445   c 
    0.414621  3.188777   c 
    0.9725813 0.1922818  b 
    0.5422726 4.486179   b 
    0.6116582 3.967756   a 
    0.3414991 1.018642   b 
    0.9516746 3.878809   c 
    0.1169475 0.3469163  c 
    0.8158957 2.050957   a 
    0.6091539 1.168774   a 
    0.9830794 3.562923   b 
    0.7543122 0.6961287  a 
    0.3813679 1.350938   b 
    ..
    '))
    >>> qtable[['x', 'x1']]
    pykx.List(pykx.q('
    0.3550381 0.3615143 0.9089531 0.2062569 0.481821   0.2065625 0.5229178 0.3338..
    1.185644  2.835405  2.134588  3.852387  0.07970141 1.786519  0.1273692 3.4404..
    '))
    >>> qtable[0:5]
    pykx.Table(pykx.q('
    x         x1         x2
    -----------------------
    0.3550381 1.185644   c 
    0.3615143 2.835405   a 
    0.9089531 2.134588   b 
    0.2062569 3.852387   a 
    0.481821  0.07970141 a 
    '))
    ``` 

* Pass a PyKX object to q function

    ```python
    >>> qfunction = kx.q('{x+til 10}')
    >>> qfunction(kx.toq([random() for _ in range(10)], kx.FloatVector))
    pykx.FloatVector(pykx.q('0.3992327 1.726329 2.488636 3.653597 4.028107 5.444905 6.542917 7.00628 8.152..'))
    ```

* Apply a Python function on a PyKX Vector

    ```python
    >>> qvec = kx.random.random(10, 10, seed=42)
    >>> qvec
    pykx.LongVector(pykx.q('4 7 2 2 9 4 2 0 8 0'))
    >>> qvec.apply(lambda x:x+1)
    pykx.LongVector(pykx.q('5 8 3 3 10 5 3 1 9 1'))
    ```
    
    

* Pass a PyKX array objects to a Numpy functions

    ```python
    >>> qarray1 = kx.toq([random() for _ in range(10)], kx.FloatVector)
    >>> qarray1
    pykx.FloatVector(pykx.q('0.7880561 0.9677446 0.9325539 0.6501981 0.4837422 0.5338642 0.5156039 0.31358..'))
    >>> qarray2 = kx.toq([random() for _ in range(10)], kx.FloatVector)
    >>> qarray2
    pykx.FloatVector(pykx.q('0.04164985 0.6417901 0.1608836 0.691249 0.4832847 0.6339534 0.4614883 0.06373..'))

    >>> np.max(qarray1)
    0.9677445779088885
    >>> np.sum(kx.toq([randint(0, 10) for _ in range(10)]))
    43
    >>> np.add(qarray1, qarray2)
    pykx.FloatVector(pykx.q('0.8297059 1.609535 1.093438 1.341447 0.9670269 1.167818 0.9770923 0.3773123 1..'))
    ```

* Query using SQL/qSQL

    ```python
    >>> qtable = kx.Table(
        data={
            'x': [['a', 'b', 'c'][randint(0, 2)] for _ in range(100)]
            'x1': [random() for _ in range(100)], 
            'x2': [random() * 5 for _ in range(100)],
        }
    )
    >>> qtable[0:5]
    pykx.Table(pykx.q('
    x x1        x2        
    ----------------------
    a 0.8236115 0.7306473 
    a 0.3865843 1.01605   
    c 0.9931491 1.155324  
    c 0.9362009 1.569154  
    c 0.4849499 0.09870703
    '))
    >>> kx.q.sql("SELECT * FROM $1 WHERE x='a'", qtable)
    pykx.Table(pykx.q('
    x x1        x2       
    ---------------------
    a 0.8236115 0.7306473
    a 0.3865843 1.01605  
    a 0.259265  2.805719 
    a 0.6140826 1.730398 
    a 0.6212161 3.97236  
    ..
    '))
    >>> kx.q.qsql.select(qtable, where = 'x=`a')
    pykx.Table(pykx.q('
    x x1        x2
    ---------------------
    a 0.8236115 0.7306473
    a 0.3865843 1.01605
    a 0.259265  2.805719
    a 0.6140826 1.730398
    a 0.6212161 3.97236
    ..
    '))
    ```

* Applying q keyword functions

    ```python
    >>> qvec = kx.q.til(10)
    >>> qvec
    pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
    >>> kx.q.mavg(3, qvec)
    pykx.FloatVector(pykx.q('0 0.5 1 2 3 4 5 6 7 8'))
    ```

## Converting PyKX objects to common Python types

Objects generated via the PyKX library can be converted where reasonable to `Python`, `Numpy`, `Pandas` and `PyArrow` types which are analogous to their underlying q representation. For example q tables are converted to Pandas Dataframes and PyArrow tables respectively. This is facilitated in each case through the use of the `py`, `np`, `pd` and `pa` methods.

* Convert PyKX objects to Python

    ```python
    >>> qdictionary = kx.toq({'a': 5, 'b': range(10), 'c': np.random.uniform(low=0.0, high=1.0, size=(5,))})
    >>> qdictionary
    pykx.Dictionary(pykx.q('
    a| 5
    b| 0 1 2 3 4 5 6 7 8 9
    c| 0.01450907 0.9131434 0.5745007 0.961908 0.7609489
    '))
    >>> qdictionary.py()
    {'a': 5, 'b': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'c': [0.014509072760120034, 0.9131434387527406, 0.5745006683282554, 0.9619080068077892, 0.7609488749876618]}
    >>>
    >>> qvec = kx.toq(np.random.randint(5, size=10))
    >>> qvec.py()
    [0, 2, 4, 1, 2, 1, 0, 1, 0, 1]
    ```

* Convert PyKX objects to Numpy

    ```python
    >>> qvec = kx.q('10?5')
    >>> qvec.np()
    array([0, 2, 4, 1, 2, 1, 0, 1, 0, 1])
    >>> qtab = kx.Table([[random(), randint(0, 5)] for _ in range(5)])
    >>> qtab
    pykx.Table(pykx.q('
    x         x1
    ------------
    0.8247812 4 
    0.2149847 0 
    0.1007832 2 
    0.4520411 4 
    0.0196153 0 
    '))
    >>> qtab.np()
    rec.array([(0.82478116, 4), (0.21498466, 0), (0.10078323, 2),
               (0.45204113, 4), (0.0196153 , 0)],
              dtype=[('x', '<f8'), ('x1', '<i8')])
    ```

* Convert PyKX objects to Pandas

    ```python
    >>> qvec = kx.toq(np.random.randint(5, size=10))
    >>> qvec.pd()
    0    0
    1    2
    2    4
    3    1
    4    2
    5    1
    6    0
    7    1
    8    0
    9    1
    dtype: int64
    >>> df = pd.DataFrame(data={'x': [random() for _ in range(5)], 'x1': [randint(0, 4) for _ in range(5)]})
    >>> qtab = kx.toq(df)
    >>> qtab.pd()
              x  x1
    0  0.824781   4
    1  0.214985   0
    2  0.100783   2
    3  0.452041   4
    4  0.019615   0
    ```

* Convert PyKX objects to PyArrow

    ```python
    >>> qvec = kx.q('10?5')
    >>> qvec.pa()
    <pyarrow.lib.Int64Array object at 0x7ffa678f4e80>
    [
      0,
      2,
      4,
      1,
      2,
      1,
      0,
      1,
      0,
      1
    ]
    >>> df = pd.DataFrame(data={'x': [random() for _ in range(5)], 'x1': [randint(0, 4) for _ in range(5)]})
    >>> qtab = kx.toq(df)
    >>> qtab.pa()
    pyarrow.Table
    x: double
    x1: int64
    ----
    x: [[0.707331785506831,0.03695847895120696,0.7024166621644556,0.3955776423810857,0.7539328513313873]]
    x1: [[4,3,2,3,2]]
    ```

## Next steps

- [Interface Overview Notebook](interface_overview.ipynb)
- [PyKX User Guide](../user-guide/index.md)

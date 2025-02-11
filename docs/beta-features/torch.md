---
title: PyTorch Conversions
description: PyTorch Tensor Conversions beta feature in PyKX
date: January 2025
author: KX Systems, Inc.,
tags: PyKX,  PyTorch Tensor
---
# PyTorch Conversions

_This page provides an overview of PyTorch Conversions, a beta feature in PyKX._

!!! Warning

	This functionality is provided as a Beta Feature and is subject to change. To enable this functionality for testing please follow the configuration instructions [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`.

## Introduction

Commonly used in the development of complex machine learning algorithms, PyTorch is a machine learning library based on the Torch library and is used in applications such as computer vision and natural language processing. Originally developed by Meta AI it is now widely used in the open-source community for algorithm development.

This beta feature allows PyKX users to convert PyKX Vector/List objects into their PyTorch [Tensor](https://pytorch.org/docs/stable/tensors.html) equivalents.

## Requirements and limitations


Before you run this functionality, first you must install `torch>2.1` in your local Python session, by using the following command:

```bash
pip install pykx[torch]
```

## Functional walkthrough

This walkthrough demonstrates the following steps:

1. Convert a PyKX Vector object to a Tensor object.
1. Convert a PyKX List object to a Tensor object.
1. Convert a Tensor object to a PyKX equivalent object.

### Vector to Tensor

Use the `*.pt()` methods to convert PyKX numeric data representations to Tensor objects. In the example below we convert PyKX numeric types to their PyTorch Tensor equivalents:

```python
>>> import os
>>> os.environ['PYKX_BETA_FEATURES'] = 'True'
>>> import pykx as kx
>>> svec = kx.q('1 2 3h')
>>> lvec = kx.q('1 2 3j')
>>> rvec = kx.q('1 2 3e')
>>> fvec = kx.q('1 2 3f')
>>> svec.pt()
tensor([1, 2, 3], dtype=torch.int16)
>>> lvec.pt()
tensor([1, 2, 3])
>>> rvec.pt()
tensor([1., 2., 3.])
>>> fvec.pt()
tensor([1., 2., 3.], dtype=torch.float64)
```

In particular note in the above that the data types are converted to their Tensor size equivalent.

### List to Tensor

To convert PyKX List objects to Tensors, two criteria must be met:

1. The `#!python pykx.List` contains only data of a single type.
1. The `#!python pykx.List` is an N-Dimensional regularly shaped/rectangular structure.

By default conversions to a `#!python torch.Tensor` object test for these criteria and it throws an error if they are not met as follows:

```python
>>> import os
>>> os.environ['PYKX_BETA_FEATURES'] = 'True'
>>> import pykx as kx
>>> kx.q('(1 2;2 3f)').pt()
TypeError: Data must be a singular type "rectangular" matrix
```

A working example of this is as follows:

```python
>>> kx.q('100 100 100#1000000?1f').pt()
tensor([[[0.3928, 0.5171, 0.5160,  ..., 0.3410, 0.8618, 0.5549],
         [0.0617, 0.2858, 0.6685,  ..., 0.9234, 0.4016, 0.5619],
         [0.7249, 0.8112, 0.2087,  ..., 0.3187, 0.1873, 0.8416],
         ...,
       dtype=torch.float64)
```

Having to pre-compute the shape of the data can slow down the processing of large matrices. To avoid this, if you already know the final shape of the tensor, you can specify it using the `#!python reshape` keyword in advance.

```python
>>> kx.q('100 100 100#1000000?1f').pt(reshape=[100, 100, 100])
tensor([[[0.3928, 0.5171, 0.5160,  ..., 0.3410, 0.8618, 0.5549],
         [0.0617, 0.2858, 0.6685,  ..., 0.9234, 0.4016, 0.5619],
         [0.7249, 0.8112, 0.2087,  ..., 0.3187, 0.1873, 0.8416],
         ...,
       dtype=torch.float64)
```

While not clear from the above for particularly complex nested `#!python pykx.List` objects setting the data shape can provide significant performance boosts:

```python
lst = kx.q('100 100 100 100#100000000?1f')
%timeit lst.pt()
# 1.22 s ± 24.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit lst.pt(reshape=[100, 100, 100, 100])
# 265 ms ± 4.96 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

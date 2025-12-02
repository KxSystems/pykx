---
title: PyKX Beta Features
description: PyKX features in beta status
date: January 2025
author: KX Systems, Inc.,
tags: PyKX, beta features, 
---
# Beta Features

_This page provides an overview of PyKX Beta Features, including what they are, how to enable them, and what features are available._

## What is a Beta Feature?

As used commonly within software development "Beta Features" within PyKX describe features which have completed an initial development process phase and are being released in an opt-in manner to users of PyKX wishing to test these features. These features are not intended to be for production use while in beta and are subject to change prior to release as full features. Usage of these features will not effect the default behavior of the library outside of the scope of the new functionality being added.

Feedback on Beta Feature development is incredibly helpful and helps to determine when these features are promoted to fully supported production features. If you run into any issues while making use of these features please raise an issue on the PyKX Github [here](https://github.com/KxSystems/pykx/issues).

## How do I enable Beta Features?

Enable PyKX beta features using the `#!python PYKX_BETA_FEATURES` configuration/environment variable. Set this before importing PyKX in a Python session, as shown below, to view available beta features through `#!python kx.beta_features`:

```python
>>> import os
>>> os.environ['PYKX_BETA_FEATURES'] = 'True'
>>> import pykx as kx
>>> kx.beta_features
['PyTorch Conversions']
```

Alternatively you can set beta features to be available at all times by adding `PYKX_BETA_FEATURES` to your `.pykx-config` file as outlined [here](../user-guide/configuration.md#configuration-file). An example of a configuration making use of this is as follows:

```bash
[default]
PYKX_KEEP_LOCAL_TIMES='true'

[beta]
PYKX_BETA_FEATURES='true'
```

## What Beta Features are available?

As mentioned above, the `beta_features` property contains the list of available features. You can retrieve this information in a Python session as follows:

```python
>>> import pykx as kx
>>> kx.beta_features
['PyTorch Conversions']
```

1. [`PyTorch Conversions`](torch.md): Allow users to convert numeric type PyKX vectors and N-Dimensional lists to PyTorch Tensor objects.

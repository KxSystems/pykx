# Beta Features

!!! "Note"

	There are currently no active features in beta status, the following page outlines broadly the concept of beta features within PyKX and how it is managed today

## What is a Beta Feature?

As used commonly within software development "Beta Features" within PyKX describe features which have completed an initial development process phase and are being released in an opt-in manner to users of PyKX wishing to test these features. These features are not intended to be for production use while in beta and are subject to change prior to release as full features. Usage of these features will not effect the default behaviour of the library outside of the scope of the new functionality being added.

Feedback on Beta Feature development is incredibly helpful and helps to determine when these features are promoted to fully supported production features. If you run into any issues while making use of these features please raise an issue on the PyKX Github [here](https://github.com/KxSystems/pykx/issues).

## How do I enable Beta Features?

Within PyKX beta features are enabled through the use of a configuration/environment variable `PYKX_BETA_FEATURES`, within a Python session users can set this prior to importing PyKX as shown below, note that when enabled you will be able to see what features are in beta through access of `kx.beta_features`:

```python
>>> import os
>>> os.environ['PYKX_BETA_FEATURES'] = 'True'
>>> import pykx as kx
>>> kx.beta_features
[]
```

Alternatively you can set beta features to be available at all times by adding `PYKX_BETA_FEATURES` to your `.pykx-config` file as outlined [here](../user-guide/configuration.md#configuration-file). An example of a configuration making use of this is as follows:

```bash
[default]
PYKX_KEEP_LOCAL_TIMES='true'

[beta]
PYKX_BETA_FEATURES='true'
```

## What Beta Features are available?

As mentioned above the list of available features to a user is contained within the `beta_features` property, for users with these features available you can get access to this information as follows within a Python session

```python
>>> import pykx as kx
>>> kx.beta_features
[]
```

There are currently no active features in beta status. This page will be updated when new beta features are added at a future point in time.

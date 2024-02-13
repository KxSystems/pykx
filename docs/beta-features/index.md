# Beta Features

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
['Database Management', 'Remote Functions']
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
['Database Management', 'Remote Functions']
```

The following are the currently available beta features:

- [Database Management](db-management.md) provides users with the ability to create, load and maintain databases and their associated tables including but not limited to:

	- Database table creation and renaming.
	- Enumeration of in-memory tables against on-disk sym file.
	- Column listing, addition, reordering, renaming copying, function application and deletion on-disk.
	- Attribute setting and removal.
	- Addition of missing tables from partitions within a database.

- [Remote Functions](remote-functions.md) let you define functions in Python which interact directly with kdb+ data on a q process. These functions can seamlessly integrate into existing Python infrastructures and also benefit systems that use q processes over Python for performance reasons or as part of legacy applications.
- [PyKX Threading](threading.md) provides users with the ability to call into `EmbeddedQ` from multithreaded python programs and allow any thread to modify global state safely.

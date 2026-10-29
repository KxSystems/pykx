---
title: Load KDB-X modules
description: How to load and use KDB-X q modules from KDB-X Python
date: September 2026
author: KX Systems, Inc.,
tags: KDB-X Python, q, modules, use, SearchPath
---

# Load KDB-X modules

_This page explains how to load KDB-X modules into your Python session with [`kx.use`](../../api/module.md), and gives examples for published modules._

A [KDB-X module](https://code.kx.com/kdb-x/modules/module-index.html) is a unit of encapsulation written in q (or against the C API) which keeps its implementation in a private namespace and publishes a dictionary of public names through an `#!q export` variable. The module decides what to export, and you decide what to call it locally, so modules avoid the namespace collisions that come with loading scripts directly.

KDB-X Python surfaces the same mechanism as a single function:

```python
>>> import pykx as kx
>>> ai = kx.use('kx.ai')
```

The object you get back is a [`kx.QModule`](../../api/module.md#pykx.module.QModule): the exported names are attributes, so you can explore it with `#!python dir` and call its members like any other Python object.

The [Module importing reference](../../api/module.md) covers the full API, including: `#!python kx.use`, `#!python kx.QModule` and `#!python kx.module.SearchPath`.

!!! Note "Modules are not the context interface"

	Modules and [the context interface](context_interface.md) solve different problems. The context interface reaches into q namespaces that already exist (or executes a matching script to create them); `#!python kx.use` loads a packaged module and hands you its exports without publishing them to a q namespace at all. Reach for `#!python kx.use` when a module ships with KDB-X or comes from the module index, and for the context interface when you are importing your own q scripts.

## Load a module

Pass the module name to `#!python kx.use`. The returned `#!python kx.QModule` prints the exported names alongside the type of each:

```python
>>> import pykx as kx
>>> kurl = kx.use('kx.kurl')
>>> dir(kurl)
['async', 'auto', 'aws', 'azr', 'deregister', 'forceRefresh', 'gcp', 'init', 'listRegistered', 'register', 'setLogLevel', 'start', 'stop', 'sync']
>>> ax = kx.use('kx.ax')
>>> ax
pykx.QModule(pykx.q('
gg| pykx.QNamespaceProxy
qp| pykx.QNamespaceProxy
qd| pykx.QNamespaceProxy
'))
```

Loads are cached, so calling `#!python kx.use` twice with the same name returns the same object rather than re-reading the module from disk:

```python
>>> kx.use('kx.pq.t') is kx.use('kx.pq.t')
True
```

A module that cannot be loaded raises `#!python ImportError`, carrying the underlying q error:

```python
>>> kx.use('kx.nosuchmodule')
ImportError: Failed to load kx.nosuchmodule: notfound: kx.nosuchmodule
```

q also writes the list of paths it tried to standard error, which the exception does not capture.

!!! Note "Exports whose names are Python keywords"

	A module can export a name Python cannot use as an attribute. Append a single trailing underscore, or use `#!python getattr`:

	```python
	>>> kx.use('kx.kurl').async_                # a trailing underscore reaches `async`
	>>> getattr(kx.use('kx.kurl'), 'async')     # equivalent; kurl.async is a SyntaxError
	>>> kx.use('kx.kurl')['async']              # a key needs no escaping at all
	```

	The underscore is stripped when the bare name exists and the underscored one does not, so exports which genuinely contain underscores (`#!q max_redirects` in Kurl) are unaffected. This matches the convention [the context interface](context_interface.md) uses, where `#!python kx.q.t.import_` reaches `#!q .t.import`.

### Bind a single exported function

A module's exports are plain attributes, so you can pick out the one function you need instead of keeping the whole module around. This is the Python equivalent of q's destructuring form ``#!q ([printf]):use`kx.printf``:

```python
>>> printf = kx.use('kx.printf').printf
>>> print(printf([b"%'d", 1234567]))
1,234,567
```

### Unpack several exports at once

A `#!python kx.QModule` iterates over its exports in the order the module declares them, so you can unpack it straight into variables:

```python
>>> gg, qp = kx.use('kx.ax.graphics')
```

The equivalent q form is ``#!q ([qp;gg]):use`kx.ax.graphics``, and the two differ in how names are matched. q looks each name up **by key**, so the order you write them in is irrelevant. Python unpacking is **positional**, so your variable names must line up with the module's export order. Check it against the module's `#!python repr` first:

```python
>>> kx.use('kx.ax.graphics')
pykx.QModule(pykx.q('
gg| pykx.QNamespaceProxy
qp| pykx.QNamespaceProxy
'))
```

`#!q gg` is listed first, so `#!python gg, qp = ...` is right for this module. A wrong export count raises `#!python ValueError`. A wrong order does not.

### Load submodules

Modules can nest, and a dotted name loads a submodule directly:

```python
>>> pqt = kx.use('kx.pq.t')
>>> dir(pqt)
['fv', 'mkP', 'mkT', 'mt', 'tt']
```

When a parent already re-exports its children you do not need a second `#!python kx.use` call. Reach through the parent by attribute instead:

```python
>>> ai = kx.use('kx.ai')
>>> dir(ai)
['bm25', 'dtw', 'flat', 'fuzzy', 'hnsw', 'hybrid', 'ivf', 'pq', 'tss']
>>> dir(ai.flat)
['normalize', 'search', 'seq']
```

Loading a submodule explicitly registers it on its parent, whichever order you load them in:

```python
>>> ai = kx.use('kx.ai')
>>> flat = kx.use('kx.ai.flat')
>>> ai.flat is flat
True
```

Some modules use a colon rather than a dot to separate a named sub-component. `#!q kx.fusion` wraps several native libraries this way:

```python
>>> fusion = kx.use('kx.fusion')
>>> fusion.libs()
pykx.SymbolVector(pykx.q('``blas`pcre2`expat'))
>>> dir(kx.use('kx.fusion:pcre2'))
['compile', 'match', 'replace', 'version']
```

!!! Note "When a dictionary is mistaken for a namespace"

	q marks a namespace with an empty-symbol key whose value is `#!q ::`, and a dictionary can carry that entry deliberately, as the placeholder which keeps a dictionary's values a general list and stops q promoting them to a typed vector. Such a dictionary is indistinguishable from a namespace, so it is presented as a [`#!python kx.module.QNamespaceProxy`](../../api/module.md#pykx.module.QNamespaceProxy), which is itself a `#!python kx.QModule`:

	```python
	>>> kafka = kx.use('kx.kafka')
	>>> type(kafka.OFFSET)
	<class 'pykx.module.QNamespaceProxy'>
	>>> kafka.OFFSET['BEGINNING']
	pykx.LongAtom(pykx.q('-2'))
	```

	When you want the dictionary itself rather than a view of it, [`kx.module.unwrap`](../../api/module.md#pykx.module.unwrap) gives it back.

### Pass an argument to `export`

A module's `#!q export` can be a function rather than a dictionary, in which case it takes a configuration argument. Pass it as the second argument to `#!python kx.use`:

```python
>>> blas = kx.use('kx.fusion:blas', {'lib': b'libopenblas.so'})
Successfully loaded BLAS library: libopenblas.so
```

Loading such a module without its argument fails, so check the module's own documentation for what it expects.

## Find modules on disk

Modules are located by searching a list of directories. `#!python kx.module.SearchPath` reads and edits that list, which is the same list q exposes as `#!q .Q.m.SP` and seeds from the `#!bash QPATH` environment variable. By default it holds `#!bash $QHOME/mod`:

```python
>>> kx.module.SearchPath()
['/home/user/.kx/mod']
```

Within each entry, KDB-X looks for a module named `#!q kx.greet` as `#!bash kx/greet.{k,q,k_,q_,<platform>.so}` and then `#!bash kx/greet/init.{k,q,k_,q_,<platform>.so}`.

Add your own directories with `#!python add=True`, and use `#!python prepend=True` to give an entry priority over those already present:

```python
>>> kx.module.SearchPath('/home/user/mymods', add=True)
>>> kx.module.SearchPath('/opt/shared/mod', add=True, prepend=True)
>>> kx.module.SearchPath()
['/opt/shared/mod', '/home/user/.kx/mod', '/home/user/mymods']
>>> kx.module.SearchPath('/opt/shared/mod', remove=True)
```

Adding an entry twice is refused, so a repeated notebook cell does not quietly grow the path. Pass `#!python allow_duplicates=True` if you want the duplicate. Removing an absent entry is a no-op unless you ask for `#!python strict=True`.

## Write your own module

A module is a q script whose last act is to define `#!q export`. Everything else in the file stays private to the module.

```q
/ mymods/kx/greet.q
hello:{"Hello, ",x,"!"}
add:{x+y}
export:([hello;add])
```

Put the directory on the search path and load it:

```python
>>> import pykx as kx
>>> kx.module.SearchPath('/home/user/mymods', add=True)
>>> greet = kx.use('kx.greet')
>>> greet
pykx.QModule(pykx.q('
hello| pykx.Lambda
add  | pykx.Lambda
'))
>>> print(greet.hello(b'KDB-X'))
Hello, KDB-X!
```

Creating modules is covered in full detail in the [Module Framework](https://code.kx.com/kdb-x/modules/module-framework/quickstart.html) documentation.

## Module reference

Each section below gives the `#!python kx.use` call, one worked example, and anything specific to driving that module from Python. Modules ship either with KDB-X itself or from the [KX GitHub organization](https://github.com/KxSystems); install one before loading it, and check its own documentation for licensing and platform prerequisites.

The [module manifest](https://code.kx.com/kdb-x/modules/module-index.html#module-manifest) lists versions and links to each module's own documentation.

| Module | Load with | Download location |
| :----- | :-------- | :---- |
| [qmamba](#qmamba) | `#!python kx.use('kx.qmamba')` | Github |
| [Parquet](#parquet) | `#!python kx.use('kx.pq')`, `#!python kx.use('kx.pq.t')` | KX |
| [AI Libraries](#ai-libraries) | `#!python kx.use('kx.ai')` | KX |
| [GPU](#gpu) | `#!python kx.use('kx.gpu')` | KX |
| [cuVS](#cuvs) | `#!python kx.use('kx.cuvs')` | KX |
| [Object Storage](#object-storage) | `#!python kx.use('kx.objstor')` | KX |
| [OneTick Cloud](#onetick-cloud) | `#!python kx.use('kx.onetickcloud')` | KX |
| [SQL](#sql) | `#!python kx.q('.s.init[]')` | KX |
| [Kurl](#kurl) | `#!python kx.use('kx.kurl')` | KX |
| [REST Server](#rest-server) | `#!python kx.use('kx.rest')` | KX |
| [Datagen](#datagen) | `#!python kx.use('kx.datagen.capmkts')`, `#!python kx.use('kx.datagen.devops')` | Github |
| [Taq](#taq) | `#!python kx.use('kx.taq')` | Github |
| [DBmaint](#dbmaint) | `#!python kx.use('kx.dbmaint')` | Github |
| [Printf](#printf) | `#!python kx.use('kx.printf')` | Github |
| [Logging](#logging) | `#!python kx.use('kx.log')` | Github |
| [ax](#ax) | `#!python kx.use('kx.ax')`, `#!python kx.use('kx.ax.graphics')`, `#!python kx.use('kx.ax.qdoc')` | Github |
| [Fusionx](#fusionx) | `#!python kx.use('kx.fusion')`, `#!python kx.use('kx.fusion:pcre2')` | Github |
| [Kafka](#kafka) | `#!python kx.use('kx.kafka')` | Github |
| [Arrow](#arrow) | `#!python kx.use('kx.arrow')` | Github |
| [Avro](#avro) | `#!python kx.use('kx.avro')` | Github |
| [Protobuf](#protobuf) | `#!python kx.use('kx.protobuf')` | Github |
| [AWS SDK](#aws-sdk) | `#!python kx.use('kx.awssdk')` | Github |
| [Skia](#skia) | `#!python kx.use('kx.skia')` | Github |

### qmamba

[Module documentation](https://code.kx.com/kdb-x/modules/package-management/overview.html). The KDB-X package manager, built on [mamba](https://mamba.readthedocs.io/). It searches for, installs and removes modules and their native dependencies, and keeps them in isolated environments. The source is in the [qmamba repository](https://github.com/KxSystems/qmamba).

Each command takes an options dictionary, which a Python `#!python dict` supplies directly. Commands print their progress to standard output. Searching the `#!python kx` channel looks like this:

```python
>>> import pykx as kx
>>> qmamba = kx.use('kx.qmamba')
>>> qmamba.search({'SPECS': ['q-kx-pr*'], 'CHANNEL': ['kx']})
 Name           Version Build       Channel Subdir
────────────────────────────────────────────────────
 q-kx-printf    1.0.1   h4616a5c_0  kx      noarch
 q-kx-printf    1.0.0   h4616a5c_0  kx      noarch
 q-kx-protobuf  2.0.0   hb0f4dca_0  kx      linux-64
pykx.Identity(pykx.q('::'))
```

To use a package, create an environment, install into it and activate it. Activation puts the environment on the module search path, so `#!python kx.use` finds what you installed:

```python
>>> env = '/home/user/envs/demo'
>>> qmamba.create({'PREFIX': env})
>>> qmamba.install({'PREFIX': env, 'SPECS': ['q-kx-printf'], 'CHANNEL': ['kx'], 'YES': 1})
>>> qmamba.activate({'PREFIX': env})
Activated environment at prefix: /home/user/envs/demo
>>> kx.module.SearchPath()
['/home/user/envs/demo/lib/q/mod', '/home/user/.kx/mod']
>>> printf = kx.use('kx.printf').printf
>>> print(printf([b"%'d", 1234567]))
1,234,567
>>> qmamba.deactivate()
```

- `#!python qmamba.active()` returns the path of the active environment, and `#!python qmamba.env.list` shows every environment mamba knows about. Pass `#!python help` to return the options each command accepts, for example `#!python qmamba.install('help')`.
- The first load creates a qmamba root folder and installs the base libraries it needs from conda-forge. By default the root prefix is the `#!bash root` directory inside the `#!bash pykx/lib` directory of the installed `pykx` package.
- If `#!bash ~/.kx/root` already exists before qmamba is used from `pykx` then that directory will be reused (unless you have `PYKX_IGNORE_QHOME=True` set).
- To choose the folder location yourself, set `#!bash QMAMBA_ROOT_PREFIX` before starting Python. The root prefix is created in a `#!bash root` directory inside the path you give. Set the variable on every launch, since without it qmamba falls back to the default location above:

	```bash
	QMAMBA_ROOT_PREFIX=$HOME/.kx python
	```

	Or from within Python:

	```python
	import os
	os.environ['QMAMBA_ROOT_PREFIX'] = os.path.expanduser('~/.kx')
	import pykx as kx
	qmamba = kx.use('kx.qmamba')
	```

- Modules that depend on native libraries need those libraries on the loader path before the process starts. When `#!bash LD_LIBRARY_PATH` does not begin with the root prefix `#!bash lib` directory, loading prints the value to use. Set it for the Python launch and relaunch Python.

### Parquet

[Module documentation](https://code.kx.com/kdb-x/modules/parquet/introduction.html). Maps [Parquet](https://parquet.apache.org/) files as virtual tables you can query with qSQL, without loading the whole file into memory. `#!q kx.pq` reads files and `#!q kx.pq.t` builds virtual tables.

```python
>>> import pykx as kx
>>> from pathlib import Path
>>> pq = kx.use('kx.pq')
>>> vt = pq.pq(Path('trades.parquet'))
>>> type(vt)
<class 'pykx.wrappers.VirtualTable'>
>>> vt.select(columns=[kx.Column('sym'), kx.Column('px')], where=kx.Column('px') > 5.0)
pykx.Table(pykx.q('
sym    px
---------
"GOOG" 6
"AAPL" 7
"MSFT" 8
"GOOG" 9
'))
>>> vt.select(columns=kx.Column('px').sum(), by=kx.Column('sym'))
pykx.KeyedTable(pykx.q('
sym   | px
------| --
"AAPL"| 12
"GOOG"| 18
"MSFT"| 15
'))
```

Projections, filters and aggregations push down into the read, so only the columns and row groups a query needs are touched. `#!python pq.op` returns a file's footer and column schema without reading data, and `#!python pq.rd` reads one column chunk out of one row group.

`#!q kx.pq.t` builds partitioned virtual tables. `#!python mkP` presents a directory of Parquet files as one queryable table, keyed by a table of partition columns:

```python
>>> pqt = kx.use('kx.pq.t')
>>> parts = [pq.pq(Path('part1.parquet')), pq.pq(Path('part2.parquet'))]
>>> keys = kx.q('([] date:2001.01.01 2001.01.02; a:1 -1)')
>>> p = pqt.mkP(kx.q('!', keys, parts))
>>> p.select()
pykx.Table(pykx.q('
date       a  sym    px
-----------------------
2001.01.01 1  "AAPL" 1
2001.01.01 1  "MSFT" 2
2001.01.02 -1 "GOOG" 3
2001.01.02 -1 "TSLA" 4
'))
```

Filtering on a partition column skips the files that cannot match.

Loading `#!q kx.pq` needs a license carrying the `pq` feature flag; without it the q error is `#!q pq: license must include 'pq' feature flag`. `#!q kx.pq.t` loads regardless, being pure q.

### AI Libraries

[Module documentation](https://code.kx.com/kdb-x/ref/ai-libs/ai-libraries-overview.html). Vector search, keyword search and time-series similarity. One load gives you every algorithm as a submodule attribute:

```python
>>> import pykx as kx
>>> ai = kx.use('kx.ai')
>>> dir(ai)
['bm25', 'dtw', 'flat', 'fuzzy', 'hnsw', 'hybrid', 'ivf', 'pq', 'tss']
>>> vecs = kx.q('{(x;y)#(x*y)?1e}[1000;10]')
>>> ai.flat.search(vecs, vecs[0], 5, 'L2')
pykx.List(pykx.q('
0 0.3443676 0.4946837 0.5293329 0.5341943
0 959       529       596       322
'))
```

`#!q del` removes points from the `#!python hnsw`, `#!python ivf` and `#!python pq.ivf` indexes. It is a Python keyword, so write it with a trailing underscore:

```python
>>> import math
>>> hnsw = ai.hnsw.put([], [], vecs, 'L2', 32, 1/math.log(32), 64)
>>> hnsw = ai.hnsw.del_(hnsw, vecs, 'L2', 13)
```

`#!python fuzzy` takes the same `#!python (data; query; k; metric)` arguments as `#!python flat`, over strings rather than vectors. Its own `#!python utils` lists the metrics it accepts:

```python
>>> words = [b'kitten', b'sitting', b'mitten', b'kite', b'bitten']
>>> ai.fuzzy.search(words, b'kitten', 3, 'levenshtein')
pykx.List(pykx.q('
0        1        1
0        4        2
"kitten" "bitten" "mitten"
'))
>>> ai.fuzzy.dist(b'kitten', b'sitting', 'levenshtein')
pykx.FloatAtom(pykx.q('3f'))
```

### GPU

[Module documentation](https://code.kx.com/kdb-x/modules/gpu/introduction.html). Moves tables onto an Nvidia GPU and runs queries there.

`#!python gpu.to` copies a table into GPU memory. Results of `#!python gpu.select` and `#!python gpu.update` stay on the GPU until you copy them back with `#!python gpu.from_`:

```python
>>> import pykx as kx
>>> gpu = kx.use('kx.gpu')
>>> trade = kx.Table(data={
...     'sym': ['AA', 'BB', 'CC', 'AA', 'BB', 'CC'],
...     'price': [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
...     'size': [100, 200, 300, 150, 250, 350],
... })
>>> Trade = gpu.to(trade)
>>> gpu.from_(gpu.select(Trade, [], {'sym': 'sym'}, {'price_max': [kx.q.max, 'price']}))
pykx.KeyedTable(pykx.q('
sym| price_max
---| ---------
AA | 11       
BB | 21       
CC | 31       
'))
```

`#!python gpu.select(table, where, by, aggregates)` mirrors the q functional form `#!q ?[t;c;b;a]`. Each phrase is a [parse tree](../../api/pykx-execution/q.md#parse): a function followed by its arguments, with a Python `#!python str` naming a column.
`#!python gpu.update` takes the same arguments and returns the whole table with columns added or replaced:

```python
>>> gpu.from_(gpu.select(Trade, [[kx.q('>'), 'price', 15.0]], False, {}))   # where, no grouping, all columns
pykx.Table(pykx.q('
sym price size
--------------
BB  20    200 
CC  30    300 
BB  21    250 
CC  31    350 
'))
>>> gpu.from_(gpu.update(Trade, [], False, {'notional': [kx.q('*'), 'price', 'size']}))
pykx.Table(pykx.q('
sym price size notional
-----------------------
AA  10    100  1000    
BB  20    200  4000    
CC  30    300  9000    
AA  11    150  1650    
BB  21    250  5250    
CC  31    350  10850   
'))
```

`#!python gpu.meta` shows where each column lives:

```python
>>> gpu.meta(Trade)
pykx.KeyedTable(pykx.q('
c    | t f a r  
-----| ---------
sym  | s     gpu
price| f     gpu
size | j     gpu
'))
```

* `#!python from` is a Python keyword. The function which copies data back to the CPU is exported as `#!python from`, so use `#!python gpu.from_` instead.
* GPU data cannot be converted directly to Python. Copy it back with `#!python gpu.from_` first, for example `#!python gpu.from_(Trade).pd()`.
* See the [module reference](https://code.kx.com/kdb-x/modules/gpu/reference.html) for the supported functions and types.
* The module's [troubleshooting](https://code.kx.com/kdb-x/modules/gpu/troubleshooting-errors.html) page lists error codes and their meaning.

### cuVS

[Module documentation](https://code.kx.com/kdb-x/modules/cuvs/introduction.html). CUDA-accelerated approximate vector search, adding the CAGRA index type alongside the AI Libraries indexes.

```python
>>> import pykx as kx
>>> cuvs = kx.use('kx.cuvs')
```

### Object Storage

[Module documentation](https://code.kx.com/kdb-x/modules/objstor/overview.html). Lets KDB-X read directly from cloud object storage, so an on-disk database can live in a bucket.

```python
>>> import pykx as kx
>>> objstor = kx.use('kx.objstor')
>>> objstor.init()
pykx.Identity(pykx.q('::'))
>>> kx.q.key(':s3://mybucket/mydb/')          # list a prefix
pykx.SymbolVector(pykx.q('`s#`2026.01.01`2026.01.02`2026.01.03'))
>>> kx.q.hcount(':s3://mybucket/mydb/2026.01.01/trade/price')
pykx.LongAtom(pykx.q('10128'))
```

After `#!python init`, ordinary KDB-X file operations accept object-storage handles: `#!q :s3://` for AWS, `#!q :gs://` for Google Cloud and `#!q :ms://` for Azure..

Credentials come from the usual cloud environment variables (`#!bash AWS_ACCESS_KEY_ID`, `#!bash AZURE_STORAGE_ACCOUNT`, `#!bash GCP_TOKEN` and so on), which must be set before the process starts. Publicly readable buckets need no credentials, only the region (.e.g set `#!bash AWS_REGION` for AWS).
### OneTick Cloud

[Module documentation](https://code.kx.com/kdb-x/modules/onetick-cloud/introduction.html). Queries OneTick Cloud market data over its REST API and returns q tables, handling OAuth 2.0 and the Arrow IPC response format.

```python
>>> import pykx as kx
>>> otc = kx.use('kx.onetickcloud')
>>> otc.sql_rest(b"select SYMBOL_NAME, TIMESTAMP, TRADE_ID, PRICE, SIZE from LSE_SAMPLE.TRD where SYMBOL_NAME='VOD' and TIMESTAMP >= '2024-01-03 00:00:00 UTC' and TIMESTAMP < '2024-01-04 00:00:00 UTC' limit 10")
pykx.Table(pykx.q('
SYMBOL_NAME TIMESTAMP                     TRADE_ID              PRICE   SIZE
------------------------------------------------------------------------------
VOD         2024.01.03D07:15:10.133000000 "1075835351045197936" 69.76   40000
VOD         2024.01.03D08:00:06.232000000 "911727684223506"     70      184613
VOD         2024.01.03D08:00:06.233000000 "911727684223588"     70.01   140
VOD         2024.01.03D08:00:06.287000000 "911727684223589"     70.01   500
VOD         2024.01.03D08:00:08.113000000 "25706436899852400"   70.104  2800
VOD         2024.01.03D08:00:09.380000000 "892755055723892848"  70.137  49
VOD         2024.01.03D08:00:09.943000000 "50599517591654512"   70.0981 66908
VOD         2024.01.03D08:00:09.944000000 "77621115355877488"   70.1097 214
VOD         2024.01.03D08:00:10.119000000 "390726911293870192"  70.146  840
VOD         2024.01.03D08:00:10.933000000 "953676864715182192"  70.132  344
'))
```

### SQL

[Module documentation](https://code.kx.com/kdb-x/modules/sql/introduction.html). Translates SQL statements into q expressions at runtime. SQL ships with KDB-X and always saves itself in the `.s` namespace. It can be loaded by `#!python kx.q('.s.init[]')` or it will be loaded automatically when you first [query using SQL](../fundamentals/query/sql.md).

```python
>>> import pykx as kx
>>> trades = kx.Table(data={
        'sym': kx.random.random(100, ['AAPL', 'GOOG', 'MSFT']),
        'date': kx.random.random(100, kx.q('2022.01.01') + [0,1,2]),
        'price': kx.random.random(100, 1000.0) 
    })
>>> # The table is referred to by $1, with further arguments using $2, $3 etc.
>>> trades.sql('select * from $1 where price>$2', 950) 
pykx.Table(pykx.q('
sym  date       price   
------------------------
GOOG 2022.01.03 976.1246
GOOG 2022.01.02 962.5156
AAPL 2022.01.03 955.0901
GOOG 2022.01.01 972.5813
MSFT 2022.01.03 951.6746
..
'))
```

### Kurl

[Module documentation](https://code.kx.com/kdb-x/modules/kurl/quickstart.html). An HTTP client with cloud credential handling built in.

`#!python sync` takes a single list of `#!python (url; method; options)` and returns `#!python (status; body)`:

```python
>>> import pykx as kx
>>> kurl = kx.use('kx.kurl')
>>> status, body = kurl.sync(['https://httpbin.org/json', 'GET', None])
>>> status
pykx.IntAtom(pykx.q('200i'))
>>> bytes(body)[:60]
b'{\n  "slideshow": {\n    "author": "Yours Truly", \n    "date":'
```

Pass headers, or any other option, as a dictionary in place of `#!python None`.


### REST Server

[Module documentation](https://code.kx.com/kdb-x/modules/rest-server/overview.html). Exposes q functions as HTTP endpoints. This module is not compatible with KDB-X Python (Python process) as the embedded q process has no main loop to respond to requests. See [Issues and Limitations](../../help/issues.md#ipc). Use from a q process instead.

### Datagen

[Module documentation](https://github.com/KxSystems/datagen). Generates synthetic data. There is no top-level export, so load a domain submodule directly, either `#!q kx.datagen.capmkts` or `#!q kx.datagen.devops`.

```python
>>> import pykx as kx
>>> capmkts = kx.use('kx.datagen.capmkts')
>>> dir(capmkts)
['buildPersistedDB', 'getInMemoryTables']
>>> trade, quote, nbbo, master, exnames = capmkts.getInMemoryTables()
>>> trade[:3]
pykx.Table(pykx.q('
sym  time                 price size stop cond ex
-------------------------------------------------
VWID 0D09:30:00.490257934 33.86 71   0    4    L
AMTM 0D09:30:00.557749093 25.01 50   0    O    B
BAC  0D09:30:00.732427780 50.84 84   0    V    Y
'))
```

`#!python buildPersistedDB` writes the same data out as a partitioned database instead of returning it in memory. `#!q kx.datagen.devops` has the same two entry points, returning `#!python master`, `#!python cpu` and `#!python logmsg`.

### TAQ

[Module documentation](https://github.com/KxSystems/taq). Parses NYSE TAQ PSV files into KDB-X, either in memory, onto disk as a date-partitioned database, or into a tickerplant.

```python
>>> import pykx as kx
>>> from datetime import date
>>> taq = kx.use('kx.taq')
>>> dir(taq)
['getSchemas', 'parseToDisk', 'parseToMemory', 'parseToTP']
>>> trade, quote, master, exnames = taq.parseToMemory(b'/tmp/nysetaqpsv', date(2026, 4, 1))
>>> len(trade), len(quote), len(master)
(15, 2797, 5)
>>> trade[:3]
pykx.Table(pykx.q('
time                 ex sym    cond size price  stop corr seq     tradeId sou..
-----------------------------------------------------------------------------..
0D10:06:26.011628535 D  ZOOZ.W @    211  0.0322      0    2086665 1       N  ..
0D10:06:26.011964035 D  ZOOZ.W @    110  0.0322      0    2086667 2       N  ..
0D10:06:26.013090662 P  ZOOZ.W @F   107  0.032       0    2086668 1       N  ..
'))
```

Every entry point takes an optional trailing options dictionary:

```python
>>> opts = {'letters': b'Y-Z', 'sortcols': 'sym', 'symattr': 'p'}
>>> trade, quote, master, exnames = taq.parseToMemory(b'/tmp/nysetaqpsv', date(2026, 4, 1), opts)
```

### DBmaint

[Module documentation](https://github.com/KxSystems/dbmaint). Maintains splayed and partitioned on-disk tables: add, drop, rename, cast and re-attribute columns across every partition.

```python
>>> import pykx as kx
>>> dbmaint = kx.use('kx.dbmaint')
>>> dbmaint.listCols('db', 'trade')
pykx.SymbolVector(pykx.q('`sym`price`size'))
>>> dbmaint.addCol('db', 'trade', 'flag', kx.q('0b'))
>>> dbmaint.fnCol('db', 'trade', 'price', kx.q('2*'))
>>> dbmaint.castCol('db', 'trade', 'size', kx.CharAtom('f'))
>>> dbmaint.renameCol('db', 'trade', 'flag', 'active')
>>> dbmaint.setAttr('db', 'trade', 'sym', 'p')
>>> dbmaint.listCols('db', 'trade')
pykx.SymbolVector(pykx.q('`sym`price`size`active'))
```

`#!python dir(dbmaint)` lists the full set, including `#!python addTab`, `#!python copyCol`, `#!python reorderCols` and the `#!python check*` inspection functions.

### Printf

[Module documentation](https://github.com/KxSystems/printf). Formats strings following a subset of the C99 `#!c printf` standard. It exports one function, so binding it directly reads best:

```python
>>> import pykx as kx
>>> printf = kx.use('kx.printf').printf
>>> print(printf([b"%'d", 1234567]))
1,234,567
>>> print(printf([b"%s has %d rows", 'trade', 42]))
trade has 42 rows
>>> print(printf([b"%r", kx.q('2025.09.03D12:00:00.000')]))
2025.09.03D12:00:00.000000000
```

The `#!q %r` conversion renders any q value in its own textual form.

The format string must reach q as a char vector. A Python `#!python str` converts to a q symbol, so use a `#!python bytes` literal or `#!python kx.CharVector`.

### Logging

[Module documentation](https://github.com/KxSystems/logging). A logging framework with levels, pluggable formats and multiple sinks. The module exports a factory, and calling it creates an independent logger:

```python
>>> import pykx as kx
>>> logger = kx.use('kx.log')
>>> type(logger.createLog())
<class 'pykx.wrappers.Dictionary'>
```

`#!python createLog` returns a q dictionary rather than a `#!python kx.QModule`, so its members would have to be reached by key. Pass it through [`kx.module.wrap`](../../api/module.md#pykx.module.wrap) for attribute access:

```python
>>> log = kx.module.wrap(logger.createLog())
>>> dir(log)
['add', 'addfmt', 'debug', 'error', 'fatal', 'fmts', 'getfmt', 'getlvl', 'info', 'remove', 'setfmt', 'setlvl', 'sinks', 'trace', 'warn']
>>> log.info(b'Hello World!')
2026.09.04D11:13:31.614026479 info PID[332949] HOST[localhost] Hello World!
pykx.Identity(pykx.q('::'))
>>> log.setfmt('syslog')
pykx.Identity(pykx.q('::'))
>>> log.info(b'syslog formatted')
<6> syslog formatted
pykx.Identity(pykx.q('::'))
```

Levels run `#!q trace`, `#!q debug`, `#!q info`, `#!q warn`, `#!q error`, `#!q fatal`. Message formatting follows [Printf](#printf), so pass a list of format-plus-arguments: `#!python log.info([b'myvar: %r', 'foo'])`. `#!python log.add` and `#!python log.remove` route levels to sinks.

!!! Note "`getlvl`, `fmts` and `sinks` are references, not functions"

	Most entries are callable, but these three are symbols naming a q variable. q dereferences a symbol when it is applied, so `#!q .log.getlvl[]` reads the level. Python has no equivalent overload, so `#!python log.getlvl` is the name itself. Read them with `#!python kx.q.get`:

	```python
	>>> log.getlvl
	pykx.SymbolAtom(pykx.q('`.m.kx.0log.log.2.lvl'))
	>>> kx.q.get(log.getlvl)
	pykx.SymbolAtom(pykx.q('`info'))
	```

### ax

[Module documentation](https://github.com/KxSystems/ax). Exposes the ax-libraries visualization and documentation components: `#!q qp` and `#!q gg` ([Grammar of Graphics](https://code.kx.com/analyst/libraries/grammar-of-graphics/)) plus `#!q qd` ([qDoc](https://code.kx.com/developer/libraries/documentation-generator/)). Load all three, or just the half you need with `#!q kx.ax.graphics` or `#!q kx.ax.qdoc`.

Plots are built as a scenegraph specification, a q table you compose from `#!q qp` layers:

```python
>>> import pykx as kx
>>> ax = kx.use('kx.ax')
>>> t = kx.q('([] x:til 20; y:20?100f)')
>>> spec = ax.qp.title(b'Demo', ax.qp.line(t, 'x', 'y', None))
>>> kx.q.cols(spec)
pykx.SymbolVector(pykx.q('`id`parents`children`item'))
```

`#!q qp` offers the usual marks, among them `#!python line`, `#!python point`, `#!python bar`, `#!python heatmap` and `#!python boxplot`. Combine them with `#!python stack`, `#!python facet`, `#!python grid` and `#!python theme`.

ax requires [Fusionx](#fusionx) PCRE2 to be loadable as `#!python kx.use('kx.fusion:pcre2')`. `#!python ax.qp.go(width, height, spec)` renders a specification by writing terminal graphics to standard output, so use it from a q console, a notebook or a dashboard rather than from a script.

### Fusionx

[Module documentation](https://code.kx.com/kdb-x/modules/fusionx.html). Wraps native libraries for use from q. The top-level module lists what is available; each library is then a colon-separated sub-component.

It currently contains:

- `#!q kx.fusion:blas`: Basic Linear Algebra Subprograms
- `#!q kx.fusion:pcre2`:  Perl Compatible Regular Expressions
- `#!q kx.fusion:expat`:  Expat XML parser library

```python
>>> import pykx as kx
>>> pcre2 = kx.use('kx.fusion:pcre2')
>>> pcre2.match(b'[0-9]+', b'order 123 qty 456', None)
pykx.Dictionary(pykx.q('x0| "123" 6  9  "456" 14 17'))
>>> pcre2.replace(b'([0-9]+)', b'order 123', b'<$1>', None)
pykx.CharVector(pykx.q('"order <123>"'))
```

`#!q blas` needs an [export argument](#pass-an-argument-to-export) naming the shared library to bind against. Pass it as `#!python bytes` so it reaches q as a char vector, and name whichever implementation you have: `#!python b'libopenblas.so'`, `#!python b'libblas.so'`, a vendor build such as MKL, or an absolute path:

```python
>>> blas = kx.use('kx.fusion:blas', {'lib': b'libopenblas.so'})
Successfully loaded BLAS library: libopenblas.so
>>> x = kx.q('1 2 3 4f')
>>> y = kx.q('5 6 7 8f')
>>> blas.ddot(x, y)             # dot product
pykx.FloatAtom(pykx.q('70f'))
>>> blas.daxpy(x, y, 2.0)       # y := alpha * x + y
pykx.FloatVector(pykx.q('7 10 13 16f'))
```

Options dictionaries can be supplied:

```python
>>> A = kx.q('2 2#1 2 3 4f')
>>> B = kx.q('2 2#5 6 7 8f')
>>> C = kx.q('2 2#0 0 0 0f')
>>> blas.dgemm(A, B, C, {'transA':b'N', 'transB':b'N', 'alpha':1.0, 'beta':0.0})
pykx.List(pykx.q('
19 22
43 50
'))
```

### Kafka

[Module documentation](https://github.com/KxSystems/kafka). Publishes to and consumes from Apache Kafka, wrapping the [`librdkafka`](https://github.com/confluentinc/librdkafka) C client.
This module is not compatible with KDB-X Python (Python process) as the embedded q process has no main loop running. See [Issues and Limitations](../../help/issues.md). Use from a q process instead.

### Arrow

[Module documentation](https://github.com/KxSystems/arrowkdb). Reads and writes Apache Arrow, Parquet and ORC files, and serializes tables to the Arrow IPC format.

The KX [Parquet](#parquet) module can also read Parquet files. Reach for Parquet when you want the file exposed as a queryable KDB-X table, and for Arrow when you want ORC, the Arrow IPC format, and ability to write files.

```python
>>> import pykx as kx
>>> arrow = kx.use('kx.arrow')
>>> trade = kx.q('([] sym:`AAPL`MSFT`GOOG; px:187.5 402.1 141.8; sz:100 250 75)')
>>> arrow.pq['writeParquetFromTable'](b'/tmp/trade.parquet', trade, None)
pykx.Identity(pykx.q('::'))
>>> arrow.pq['readParquetToTable'](b'/tmp/trade.parquet', None)
pykx.Table(pykx.q('
sym    px    sz
----------------
"AAPL" 187.5 100
"MSFT" 402.1 250
"GOOG" 141.8 75
'))
```

Arrow exports subdirectories of functions as dictionaries. These can be wrapped as module objects for easier access:

```python
>>> for a in dir(arrow):
...     setattr(arrow, a, kx.module.wrap(getattr(arrow, a)))
... 
>>> arrow.pq.readParquetToTable(b'/tmp/trade.parquet', None)
pykx.Table(pykx.q('
sym    px    sz 
----------------
"AAPL" 187.5 100
"MSFT" 402.1 250
"GOOG" 141.8 75 
'))
```

### Avro

[Module documentation](https://github.com/KxSystems/avrokdb). Encodes and decodes Apache Avro, taking a JSON schema and mapping it onto q types. A schema compiles once into a foreign object, which you then pass to every `#!python encode` and `#!python decode`:

```python
>>> import pykx as kx
>>> avro = kx.use('kx.avro')
>>> js = ('{"type":"record","name":"trade","fields":['
...        '{"name":"sym","type":"string"},{"name":"px","type":"double"}]}')
>>> schema = avro.schemaFromString(kx.CharVector(js))
>>> rec = {'sym': b'AAPL', 'px': 187.5}
>>> enc = avro.encode(schema, rec, None)
>>> enc
pykx.ByteVector(pykx.q('0x084141504c0000000000706740'))
>>> avro.decode(schema, enc, None)
pykx.Dictionary(pykx.q('
   | ::
sym| "AAPL"
px | 187.5
'))
```

`#!python AVRO_FORMAT` switches the wire format, and the JSON forms return a char vector instead of bytes:

```python
>>> print(bytes(avro.encode(schema, rec, {'AVRO_FORMAT': 'JSON'})).decode())
{"sym":"AAPL","px":187.5}
```

### Protobuf

[Module documentation](https://github.com/KxSystems/protobufkdb). Serializes and parses Google Protocol Buffers messages, reading `#!bash .proto` definitions at runtime. Point the module at a directory of `#!bash .proto` files, import one, and the message types become available by name:

```python
>>> import pykx as kx
>>> pb = kx.use('kx.protobuf')
>>> pb.addProtoImportPath(b'/home/user/proto')
>>> pb.importProtoFile(b'trade.proto')
>>> pb.getMessageFields('Trade')
pykx.SymbolVector(pykx.q('`sym`px`sz'))
>>> rec = {'sym': 'AAPL', 'px': 187.5, 'sz': 100}
>>> ser = pb.serializeArrayFromDict('Trade', rec)
>>> pb.parseArrayToDict('Trade', ser)
pykx.Dictionary(pykx.q('
sym| "AAPL"
px | 187.5
sz | 100
'))
```

### AWS SDK

[Module documentation](https://github.com/KxSystems/awssdk). Wraps the official AWS SDK for C++.

```python
>>> import pykx as kx
>>> aws = kx.use('kx.awssdk')
>>> aws.initialize()
pykx.BooleanAtom(pykx.q('1b'))
>>> aws.getCredentials()
pykx.Dictionary(pykx.q('
accessKey   | "AKIAEXAMPLE"
secretKey   | "secretexample"
sessionToken| ""
accountId   | ""
expiration  | 9223372036854
'))
>>> aws.shutDown()
pykx.BooleanAtom(pykx.q('1b'))
```

### Skia

[Module documentation](https://github.com/KxSystems/qskia). Draws two-dimensional graphics through Google's Skia library and renders them to a PNG or a raw pixel buffer. The [ax](#ax) module uses it for rendering, loaded on its own it is a drawing API you call directly.

`#!python init` takes the directory holding the bundled fonts, which the release ships as `#!bash fonts/ivyfonts` under the install prefix. Everything else works on a canvas handle from `#!python new`:

```python
>>> import pykx as kx
>>> skia = kx.use('kx.skia')
>>> skia.init(b'/home/user/.kx/mod/kx')
pykx.Identity(pykx.q('::'))
>>> s = skia.new(320, 160)
>>> skia.setBackgroundColour(s, kx.q('"i"$0x0 sv "x"$255 255 255 255'))
>>> skia.setFillColour(s, kx.q('"i"$0x0 sv "x"$255 0 90 200'))
>>> skia.addRect(s, 20.0, 20.0, 80.0, 60.0)
>>> skia.setFontSize(s, 18)
>>> skia.addText(s, 20, 140, b'KDB-X Python')
>>> png = skia.toPNG(s)
>>> open('/tmp/chart.png', 'wb').write(bytes(png))
2609
```

Colors are ARGB packed into an int, which is why the example builds them from four bytes.

`#!python toRGB` returns the raw pixels as a four-element list of width, height, channel count and the pixel bytes, in BGRA order:

```python
>>> w, h, channels, pixels = skia.toRGB(s)
>>> w.py(), h.py(), channels.py(), len(pixels)
(320, 160, 4, 204800)
```

Every canvas has to be released with `#!python delete`. In a notebook, `#!python IPython.display.Image(bytes(png))` renders the result inline, which saves the trip through a file.

Too few arguments gives you a projection, not an error. Check function arity in the [module reference](https://github.com/KxSystems/qskia/blob/main/docs/reference.md) before calling.

```python
>>> skia.multiFillRect(s, xs, ys, ws, hs)      # one argument short
pykx.Projection(pykx.q('code[foreign;..'))
>>> skia.delete(s)
```

## Next steps

- The [module importing API reference](../../api/module.md) document all of `#!python kx.module` in full.
- The [KDB-X module index](https://code.kx.com/kdb-x/modules/module-index.html) lists every module with its version and its own documentation.
- [Import existing q functions](context_interface.md) shows how to load your own q scripts and namespaces.
- [Use Python in a q process](../../pykx-under-q/intro.md) is the route for modules that need q's event loop.

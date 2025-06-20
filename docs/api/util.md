# PyKX Utilities

The purpose of this page is to provide users with documentation for utility functions located within various modules within PyKX. 

!!! Note

	This functionality presently is not located in a centralized module but it is expected that with the next major release version of PyKX 3.0.0 they 

## `pykx.ssl_info`

```python
pykx.ssl_info()
```

View information relating to the TLS Settings used by PyKX from your process

**Returns:**

| Type              | Description                                          |
|-------------------|------------------------------------------------------|
| `pykx.Dictionary` | A dictionary outlining the TLS settings used by PyKX |

**Example:**

```python
>>> import pykx as kx
>>> kx.ssl_info()
pykx.Dictionary(pykx.q('
SSLEAY_VERSION   | OpenSSL 1.1.1q  5 Jul 2022
SSL_CERT_FILE    | /usr/local/anaconda3/ssl/server-crt.pem
SSL_CA_CERT_FILE | /usr/local/anaconda3/ssl/cacert.pem
SSL_CA_CERT_PATH | /usr/local/anaconda3/ssl
SSL_KEY_FILE     | /usr/local/anaconda3/ssl/server-key.pem
SSL_CIPHER_LIST  | ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:..
SSL_VERIFY_CLIENT| NO
SSL_VERIFY_SERVER| YES
'))
```

## `pykx.util.kill_q_process`

```python
pykx.util.kill_q_process(port)
```

Kill a q process running on a specified port, this allows users to kill sub-processes running q in the case access to the port has been lost due to parent process

**Parameters:**

| Name       | Type | Description                                    | Default |
|------------|------|------------------------------------------------|---------|
| port       | int  | The integer representing the port to be killed |         |


**Returns:**

| Type | Description                                                                          |
|------|--------------------------------------------------------------------------------------|
| bool | Returns `True` if process was successfully killed, `False` if process was not killed |


## `pykx.util.debug_environment`

```python
pykx.util.debug_environment(detailed=False, return_info=False)
```

**Parameters:**

| Name        | Type | Description                                                                                               | Default |
|-------------|------|-----------------------------------------------------------------------------------------------------------|---------|
| detailed    | bool | When returning information about a users license print the content of both `QHOME` and `QLIC` directories | `False` |
| return_info | bool | Should the information returned from the function be printed to console (default) or provided as a str    | `False` |


**Returns:**

| Type               | Description                                                                                         |
|--------------------|-----------------------------------------------------------------------------------------------------|
| `Union[None, str]` | Returns `None` if return information is printed to console otherwise returns a `str` representation |

**Example:**

```python
>>> import pykx as kx
>>> kx.util.debug_environment()
**** PyKX information ****
pykx.args: ()
pykx.qhome: /usr/local/anaconda3/envs/qenv/q
pykx.qlic: /usr/local/anaconda3/envs/qenv/q
pykx.licensed: True
pykx.__version__: 2.4.3
pykx.file: /usr/local/anaconda3/lib/python3.8/site-packages/pykx/util.py

**** Python information ****
sys.version: 3.8.3 (default, Jul  2 2020, 11:26:31) 
[Clang 10.0.0 ]
pandas: 2.0.3
numpy: 1.24.4
pytz: 2023.3.post1
which python: /usr/local/bin/python
which python3: /usr/local/anaconda3/lib/python3.8/bin/python3
find_libpython: /usr/local/anaconda3/lib/libpython3.8.dylib

**** Platform information ****
platform.platform: macOS-10.16-x86_64-i386-64bit

**** PyKX Configuration Variables ****
PYKX_IGNORE_QHOME: False
PYKX_KEEP_LOCAL_TIMES: False
PYKX_ALLOCATOR: False
PYKX_GC: False
PYKX_LOAD_PYARROW_UNSAFE: False
PYKX_MAX_ERROR_LENGTH: 256
PYKX_NOQCE: False
PYKX_RELEASE_GIL: False
PYKX_Q_LIB_LOCATION: /usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib
PYKX_Q_LOCK: False
PYKX_SKIP_UNDERQ: False
PYKX_Q_EXECUTABLE: /usr/local/anaconda3/envs/qenv/q/m64/q
PYKX_THREADING: False
PYKX_4_1_ENABLED: False
PYKX_QDEBUG: False
PYKX_DEBUG_INSIGHTS_LIBRARIES: False
PYKX_DEFAULT_CONVERSION:
PYKX_EXECUTABLE: /usr/local/anaconda3/lib/python3.8/bin/python3.8
PYKX_PYTHON_LIB_PATH:
PYKX_PYTHON_BASE_PATH:
PYKX_PYTHON_HOME_PATH:
PYKX_DIR: /usr/local/anaconda3/lib/python3.8/site-packages/pykx
PYKX_USE_FIND_LIBPYTHON:
PYKX_UNLICENSED:
PYKX_LICENSED:
PYKX_4_1_ENABLED:

**** q Environment Variables ****
QARGS: 
QHOME: /usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib
QLIC: /usr/local/anaconda3/envs/qenv/q
QINIT: 

**** License information ****
pykx.qlic directory: True
pykx.qhome writable: True
pykx.qhome lics: ['k4.lic']
pykx.qlic lics: ['k4.lic']

**** q information ****
which q: /usr/local/anaconda3/envs/qenv/q/q
q info: 
(`m64;4f;2020.05.04)
"insights.lib.embedq insights.lib.pykx..
```

# Troubleshooting

## License issues

The following section outlines practical information useful when dealing with getting access to and managing licenses for PyKX.

### Accessing a license valid for PyKX

A number of trial and enterprise type licenses exist for q/kdb+. Not all licenses for q/kdb+ however are valid for PyKX. In particular users require access to a license which contains the feature flags **pykx** and **embedq** which provide access to the PyKX functionality. The following locations can be used for the retrieval of evaluation/personal licenses

- For non-commercial personal users you can access a 12 month kdb+ license with PyKX enabled [here](https://kx.com/kdb-insights-sdk-personal-edition-download).
- For commercial evaluation, contact your KX sales representative or sales@kx.com requesting a PyKX trial license. Alternately apply through https://kx.com/book-demo.

For non-personal or non-commercial usage please contact sales@kx.com.

Once you have access to your license you can install the license following the steps provided [here](../getting-started/installing.md) or through installation using the function `#!python kx.license.install` as follows

```python
>>> import pykx as kx
>>> kx.license.install('/path/to/downloaded/kc.lic')
```

### Initialization failing with a `#!python embedq` error

Failure to initialize PyKX while raising an error `#!python embedq` indicates that the license you are attempting to use for PyKX in [licensed modality](../user-guide/advanced/modes.md) does not have the sufficient feature flags necessary to run PyKX. To access a license which does allow for running PyKX in this modality please following the instructions [here](#accessing-a-license-valid-for-pykx) to get a new license with appropriate feature flags.

### Initialization failing with a `#!python kc.lic` error

If after initially completing the installation guide for PyKX [here](../getting-started/installing.md) you receive the following error:

```python
pykx.exceptions.PyKXException: Failed to initialize embedded q. Captured output from initialization attempt:
    '2023.09.02T21:28:45.699 licence error: kc.lic
```

It usually indicates that your license was not correctly written to disk or a license could not be found, to check that the installed license matches the license you expect.

=== "License file based checking"

	The following shows a successful check being completed:

	```python
	>>> import pykx as kx
	>>> kx.license.check('/path/to/downloaded/kc.lic')
        True
	```

	The following shows an example of a failed check:

	```python
	>>> import pykx as kx
	>>> kx.license.check('/path/to/incorrect/license.txt')
	Supplied license information does not match.
	Please consider reinstalling your license using pykx.util.install_license

	On disk license:
	b'Atc/wy/gMjZgIdn1KlT3JVWfVmPk55dtb0YJVes5V4ed9Zxt9UVr8G/A1Q3aWiQEkfjGbwvlJU3GXpUergObvzxGN1iyYGZasG5s8vevfAI2ttndt//Y2thrryoQRm9Dy+DIIcmSufwomL+PMJkZacYc9DM6ipnQsL0KvLwLXLrQC1fBLV2pZHCdYC/nX/KM6uslgip4EoTxZTcx1pQPyTx56QKD4K4JBNimO929w/0+v4Hy2x+DIS3n89vpGmtVvjjFRQtsF6Sjnd+6RnFGk13hRL/DlqHTv2XbZgVv++YOCIc7G55KL6PVJYpB66lq9OiZCEdq2GFJLCn2TNWGJPT2s1YDAKsAPI5W3PqJkC2UeV17gPG4gxlCSHr0kfacINbEJ0kSTm/UsuEBZ5B/jvR/jU7rFErcd9PECeQA1kXB19fa4hgvbd+SxWTPxMUKbiHThHk6X0Bi3T7WAQ+sZWsEWwkMncd+mOGS3D+bRav2nfOpKckj8rCdvYum3U8PDv6IHP=S+LaCnJM0yqNjW9xGyog5mlbX2k3mBRyBjbJH/1OWTcIg7uDYxxoMtDOCJjeBdSqI=aK+5FVTVarfowvudv7QsMGeohGaJMyczNWVPPjsbyvsxbAwdXvJUuP0jcFCFVeF'

	Supplied string content:
	b'8nD+HkcJ93xW4oOEtHIZxeWkA1glv5wJ5wE2Fsmbc4lg2ntT9JpsclE1hFeG/Ox/jM4=6GjXD2VNpiCAJ80DNVcXuDB+IPEnP22DMGvBIolJt2pdy9kooGZNQpr6svIkRWX/0m/SbydbQOQUVvfNTxsDjZvvsCiGkdQtygs3sDEJbxsT+KfjqJ7Sd6RQ/47HJHG4JyIWdhmvEBVGSLBa5mdAaCLWdCrga3hHZbW3F4e/l3K4nOQvU91WEiMd6PT061r66AOYmjGACCXqmQ9kSsJfMTXPRi9M2i93Oyv895kFVKdZCLCdKdaow790RcjwnKjFFOERGcge=lZdRtp2BLA+JbixvTIKTObmfqr7uPYsGQLfXSFnQCq7jbt3yxv1ZPjvjYLPTx7YKIvgo+ITG6vyYe+cfwaW1g0tlvFTcVSVb/sxUvvLCLiWMdxGjt5JUxV3GaSm9ysHVk5MrTDpp/5qqXes1/BOXsD2DmS/QSZr/Mt+Vc2baKuxPw1w5YnGVuY6vHxHffABzkn+WPcguabr86JcmIAcC0zc2TLkbufBPJewYka9PIt1Ng283NKe13huPUohnryYVIMPyjrTWpDid+yC5kSGVeP0/5+rJvLmFZUB/n0RUjgMZU5V++GPU1QnCBa+'
	False
	```

=== "Encoded string based checking"

        The following shows a successful check being completed:

        ```python
        >>> import pykx as kx
        >>> license_string = 'Atc/wy/gMjZgIdn1KlT3JVWfVmPk55dtb0YJVes5V4ed9Zxt9UVr8G/A1Q3aWiQEkfjGbwvlJU3GXpUergObvzxGN1iyYGZasG5s8vevfAI2ttndt//Y2thrryoQRm9Dy+DIIcmSufwomL+PMJkZacYc9DM6ipnQsL0KvLwLXLrQC1fBLV2pZHCdYC/nX/KM6uslgip4EoTxZTcx1pQPyTx56QKD4K4JBNimO929w/0+v4Hy2x+DIS3n89vpGmtVvjjFRQtsF6Sjnd+6RnFGk13hRL/DlqHTv2XbZgVv++YOCIc7G55KL6PVJYpB66lq9OiZCEdq2GFJLCn2TNWGJPT2s1YDAKsAPI5W3PqJkC2UeV17gPG4gxlCSHr0kfacINbEJ0kSTm/UsuEBZ5B/jvR/jU7rFErcd9PECeQA1kXB19fa4hgvbd+SxWTPxMUKbiHThHk6X0Bi3T7WAQ+sZWsEWwkMncd+mOGS3D+bRav2nfOpKckj8rCdvYum3U8PDv6IHP=S+LaCnJM0yqNjW9xGyog5mlbX2k3mBRyBjbJH/1OWTcIg7uDYxxoMtDOCJjeBdSqI=aK+5FVTVarfowvudv7QsMGeohGaJMyczNWVPPjsbyvsxbAwdXvJUuP0jcFCFVeF'
        >>> kx.license.check(license_string, format = 'STRING')
        True
        ```

        The following shows an example of a failed check:

        ```python
        >>> import pykx as kx
	>>> license_string = '8nD+HkcJ93xW4oOEtHIZxeWkA1glv5wJ5wE2Fsmbc4lg2ntT9JpsclE1hFeG/Ox/jM4=6GjXD2VNpiCAJ80DNVcXuDB+IPEnP22DMGvBIolJt2pdy9kooGZNQpr6svIkRWX/0m/SbydbQOQUVvfNTxsDjZvvsCiGkdQtygs3sDEJbxsT+KfjqJ7Sd6RQ/47HJHG4JyIWdhmvEBVGSLBa5mdAaCLWdCrga3hHZbW3F4e/l3K4nOQvU91WEiMd6PT061r66AOYmjGACCXqmQ9kSsJfMTXPRi9M2i93Oyv895kFVKdZCLCdKdaow790RcjwnKjFFOERGcge=lZdRtp2BLA+JbixvTIKTObmfqr7uPYsGQLfXSFnQCq7jbt3yxv1ZPjvjYLPTx7YKIvgo+ITG6vyYe+cfwaW1g0tlvFTcVSVb/sxUvvLCLiWMdxGjt5JUxV3GaSm9ysHVk5MrTDpp/5qqXes1/BOXsD2DmS/QSZr/Mt+Vc2baKuxPw1w5YnGVuY6vHxHffABzkn+WPcguabr86JcmIAcC0zc2TLkbufBPJewYka9PIt1Ng283NKe13huPUohnryYVIMPyjrTWpDid+yC5kSGVeP0/5+rJvLmFZUB/n0RUjgMZU5V++GPU1QnCBa+'
        >>> kx.license.check(license_string, format = 'STRING')
        Supplied license information does not match.
        Please consider reinstalling your license using pykx.util.install_license

        On disk license:
        b'Atc/wy/gMjZgIdn1KlT3JVWfVmPk55dtb0YJVes5V4ed9Zxt9UVr8G/A1Q3aWiQEkfjGbwvlJU3GXpUergObvzxGN1iyYGZasG5s8vevfAI2ttndt//Y2thrryoQRm9Dy+DIIcmSufwomL+PMJkZacYc9DM6ipnQsL0KvLwLXLrQC1fBLV2pZHCdYC/nX/KM6uslgip4EoTxZTcx1pQPyTx56QKD4K4JBNimO929w/0+v4Hy2x+DIS3n89vpGmtVvjjFRQtsF6Sjnd+6RnFGk13hRL/DlqHTv2XbZgVv++YOCIc7G55KL6PVJYpB66lq9OiZCEdq2GFJLCn2TNWGJPT2s1YDAKsAPI5W3PqJkC2UeV17gPG4gxlCSHr0kfacINbEJ0kSTm/UsuEBZ5B/jvR/jU7rFErcd9PECeQA1kXB19fa4hgvbd+SxWTPxMUKbiHThHk6X0Bi3T7WAQ+sZWsEWwkMncd+mOGS3D+bRav2nfOpKckj8rCdvYum3U8PDv6IHP=S+LaCnJM0yqNjW9xGyog5mlbX2k3mBRyBjbJH/1OWTcIg7uDYxxoMtDOCJjeBdSqI=aK+5FVTVarfowvudv7QsMGeohGaJMyczNWVPPjsbyvsxbAwdXvJUuP0jcFCFVeF'

        Supplied string content:
        b'8nD+HkcJ93xW4oOEtHIZxeWkA1glv5wJ5wE2Fsmbc4lg2ntT9JpsclE1hFeG/Ox/jM4=6GjXD2VNpiCAJ80DNVcXuDB+IPEnP22DMGvBIolJt2pdy9kooGZNQpr6svIkRWX/0m/SbydbQOQUVvfNTxsDjZvvsCiGkdQtygs3sDEJbxsT+KfjqJ7Sd6RQ/47HJHG4JyIWdhmvEBVGSLBa5mdAaCLWdCrga3hHZbW3F4e/l3K4nOQvU91WEiMd6PT061r66AOYmjGACCXqmQ9kSsJfMTXPRi9M2i93Oyv895kFVKdZCLCdKdaow790RcjwnKjFFOERGcge=lZdRtp2BLA+JbixvTIKTObmfqr7uPYsGQLfXSFnQCq7jbt3yxv1ZPjvjYLPTx7YKIvgo+ITG6vyYe+cfwaW1g0tlvFTcVSVb/sxUvvLCLiWMdxGjt5JUxV3GaSm9ysHVk5MrTDpp/5qqXes1/BOXsD2DmS/QSZr/Mt+Vc2baKuxPw1w5YnGVuY6vHxHffABzkn+WPcguabr86JcmIAcC0zc2TLkbufBPJewYka9PIt1Ng283NKe13huPUohnryYVIMPyjrTWpDid+yC5kSGVeP0/5+rJvLmFZUB/n0RUjgMZU5V++GPU1QnCBa+'
        False
        ```

## Environment issues

### Using PyKX under q is raising a `'libpython` error

If you are getting a `'libpython` error when starting PyKX within a q session, this may indicate that PyKX has been unable to source the Python shared libraries that are required to run Python within an embedded setting. To fix this issue users can either

- Find the absolute path to the appropriate shared object and set the environment variable `PYKX_PYTHON_LIB_PATH` with this location.
- Set the environment variable `PYKX_USE_FIND_LIBPYTHON` to `"true"`, this will use the Python library [`find-libpython`](https://pypi.org/project/find-libpython/) to locate the `libpython` shared library and automatically set `PYKX_PYTHON_LIB_PATH` to the returned location.

### Getting more information about your environment

The following section outlines how a user can get access to a verbose set of environment configuration associated with PyKX. This information is helpful when debugging your environment and should be provided if possible with support requests.

```python
>>> import pykx as kx
>>> kx.util.debug_environment()  # see below for output
```

??? output

	```python
	>>> kx.util.debug_environment()
	**** PyKX information ****
	pykx.args: ()
	pykx.qhome: /usr/local/anaconda3/envs/qenv/q
	pykx.qlic: /usr/local/anaconda3/envs/qenv/q
	pykx.licensed: True
	pykx.__version__: 3.1.3
	pykx.file: /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pykx/util.py

	**** Python information ****
	sys.version: 3.12.3 (v3.12.3:f6650f9ad7, Apr  9 2024, 08:18:48) [Clang 13.0.0 (clang-1300.0.29.30)]
	pandas: 1.5.3
	numpy: 1.26.2
	pytz: 2024.1
	which python: /usr/local/bin/python
	which python3: /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
	find_libpython: /Library/Frameworks/Python.framework/Versions/3.12/Python

	**** Platform information ****
	platform.platform: macOS-13.0.1-x86_64-i386-64bit

	**** PyKX Configuration File ****
	File location: /usr/local/.pykx-config
	Used profile: default
	Profile content: {'PYKX_Q_EXECUTABLE': '/usr/local/anaconda3/envs/qenv/q/m64/q'}

	**** PyKX Configuration Variables ****
	PYKX_IGNORE_QHOME: False
	PYKX_KEEP_LOCAL_TIMES: False
	PYKX_ALLOCATOR: False
	PYKX_GC: False
	PYKX_LOAD_PYARROW_UNSAFE: False
	PYKX_MAX_ERROR_LENGTH: 256
	PYKX_NOQCE: False
	PYKX_RELEASE_GIL: False
	PYKX_Q_LIB_LOCATION: /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pykx/lib
	PYKX_Q_LOCK: False
	PYKX_SKIP_UNDERQ: False
	PYKX_Q_EXECUTABLE: /usr/local/anaconda3/envs/qenv/q/m64/q
	PYKX_THREADING: False
	PYKX_4_1_ENABLED: False
	PYKX_QDEBUG: False
	PYKX_DEBUG_INSIGHTS_LIBRARIES: False
	PYKX_CONFIGURATION_LOCATION: .
	PYKX_NO_SIGNAL: False
	PYKX_CONFIG_PROFILE: default
	PYKX_BETA_FEATURES: True
	PYKX_JUPYTERQ: False
	PYKX_SUPPRESS_WARNINGS: False
	PYKX_DEFAULT_CONVERSION: 
	PYKX_EXECUTABLE: /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
	PYKX_PYTHON_LIB_PATH: 
	PYKX_PYTHON_BASE_PATH: 
	PYKX_PYTHON_HOME_PATH: 
	PYKX_DIR: /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pykx
	PYKX_USE_FIND_LIBPYTHON: 
	PYKX_UNLICENSED: 
	PYKX_LICENSED: 
    PYKX_4_1_ENABLED: 

	**** q Environment Variables ****
	QARGS: 
	QHOME: /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pykx/lib
	QLIC: /usr/local/anaconda3/envs/qenv/q
	QINIT: 

	**** License information ****
	pykx.qlic directory: True
	pykx.qhome writable: True
	pykx.qhome lics: ['k4.lic']
	pykx.qlic lics: ['k4.lic']

	**** q information ****
	which q: /usr/local/bin/q
	q info: 
	(`m64;4.1;2024.10.16)
	"insights.lib.embedq insights.lib.pykx insights.lib.sql insights.lib.qlog insights.lib.kurl insights.lib.objstore insights.lib.bigquery insights.lib.restserver insights.app.rt"
	```

## Development issues

### Debugging q code issues

If you are developing a library of q code, by default PyKX does not provide the full backtrace on error. As an example assume you have developed the a function and pass it an incorrect input

```python
>>> import pykx as kx
>>> kx.q('{x+1}', 'e')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/anaconda3/lib/python3.12/site-packages/pykx/wrappers.py", line 5124, in __call__
    return _wrappers.function_call(self, args, no_gil)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pykx/_wrappers.pyx", line 383, in pykx._wrappers.function_call
  File "pykx/_wrappers.pyx", line 384, in pykx._wrappers.function_call
  File "pykx/_wrappers.pyx", line 515, in pykx._wrappers.factory
pykx.exceptions.QError: type
```

While this provides you the appropriate `QError`, without setting the configuration value `PYKX_QDEBUG` your error does not indicate where your error has arisen from. Using the same example below you can see the backtrace information provided when setting the configuration value `PYKX_QDEBUG` to True.

```python
>>> import os
>>> os.environ['PYKX_QDEBUG'] = 'True'
>>> import pykx as kx
>>> kx.q('{x+1}', 'e')
backtrace:
  [2]  {x+1}
         ^
  [1]  (.Q.trp)

  [0]  {[pykxquery] .Q.trp[value; pykxquery; {2@"backtrace:
                    ^
",.Q.sbt y;'x}]}
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/anaconda3/lib/python3.12/site-packages/pykx/embedded_q.py", line 246, in __call__
    return factory(result, False, name=query.__str__())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pykx/_wrappers.pyx", line 522, in pykx._wrappers._factory
  File "pykx/_wrappers.pyx", line 515, in pykx._wrappers.factory
pykx.exceptions.QError: type
```

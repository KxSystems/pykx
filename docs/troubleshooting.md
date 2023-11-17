# Troubleshooting

## License issues

The following section outlines practical information useful when dealing with getting access to and managing licenses for PyKX.

### Accessing a license valid for PyKX

A number of trial and enterprise type licenses exist for q/kdb+. Not all licenses for q/kdb+ however are valid for PyKX. In particular users require access to a license which contains the feature flags `pykx` and `embedq` which provide access to the PyKX functionality. The following locations can be used for the retrieval of evaluation/personal licenses

- For non-commercial personal users you can access a 12 month kdb+ license with PyKX enabled [here](https://kx.com/kdb-insights-personal-edition-license-download).
- For commercial evaluation you can download a 30 day PyKX license [here](https://kx.com/kdb-insights-commercial-evaluation-license-download/).

For non-personal or non-commercial usage please contact sales@kx.com.

Once you have access to your license you can install the license following the walkthrough provided [here](getting-started/installing.md) or through installation using the function `kx.license.install` as follows

```python
>>> import pykx as kx
>>> kx.license.install('/path/to/downloaded/kc.lic')
```

### Initialization failing with a 'embedq' error

Failure to initialize PyKX while raising an error `embedq` indicates that the license you are attempting to use for PyKX in [licensed modality](modes.md) does not have the sufficient feature flags necessary to run PyKX. To access a license which does allow for running PyKX in this modality please following the instructions [here](#accessing-a-license-valid-for-pykx) to get a new license with appropriate feature flags.

### Initialization failing with a 'kc.lic' error

If after initially completing the installation guide for PyKX [here](getting-started/installing.md) you receive the following error:

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
	b'Atc/wy/gMjZgIdn1KlT3JVWfVmPk55dtb0YJVes5V4ed9Zxt9UVr8G/A1Q3aWiQEkfjGbwvlJU3GXpUergObvzxGN1iyYG\nZasG5s8vevfAI2ttndt//Y2th\nrryoQRm9Dy+DIIcmSufwomL+\nPMJkZacYc9DM6ipnQsL0KvLwLXLrQC1fBLV2pZHCdYC/nX/KM6uslgip4EoTxZTcx1pQPyTx56QKD4K4JBNimO929w/0+v4Hy2x+DIS3n89vpGmtVvjjFRQtsF6Sjnd+6RnFGk13hRL/DlqHTv2XbZgVv++YOCIc7G55KL6PVJY\npB\n66lq9OiZCEdq2GFJLCn2T\nNWGJPT2s1YDAKsAPI5W3PqJkC2UeV17gPG4gxlCSHr0kfacINbEJ0kSTm/UsuEBZ5B/jvR/jU7rFErcd9PECeQA1kXB19fa4hgvbd+SxWTPxMUKbiHThHk6X0Bi3T7WAQ+sZWsEWwkMncd+mOGS\n3D+bRav2nfOpKckj8rCdvYum3U8PDv6IHP=S+\nLaCnJM0yqNjW9xGyog5ml\nbX2k3mBRyBjbJH/1OWTcIg7uDYxxoMtDOCJjeBdSqI=aK+5FVTVarfowvudv7QsMGeohGaJMyczNWVPPjsbyvsxbAwdXvJUuP0jcFCFVeF\n'

	Supplied string content:
	b'8n\nD+HkcJ93xW4oOEtH\nIZxeWkA1glv5wJ5wE2Fsmbc4lg2ntT9JpsclE1hFeG/Ox/jM4=6GjXD2VNpiCAJ80DNVcXuDB+IPEnP22DMGvBIolJt2pdy9kooGZNQpr6svIkRWX/0m/SbydbQOQUVvfNTxsDjZvvsCiGkdQtygs3sDEJbxsT+KfjqJ7Sd6RQ/47HJHG4JyIWdhmvEBVGSLBa5mdAaCLWdCrga3hHZbW3F4e/l3K4nOQvU91WEiMd6PT061r66AOYmjGACCXqmQ9kSsJfMTXPRi9M2i93Oyv895kFVKdZCLCdKdaow790RcjwnKjFFOERGcge=lZdRtp2BL\nA+JbixvTIKTObmfqr7uPYsGQLfXSFnQCq7jbt3yxv1ZPjvjYLPTx7YKIvgo+ITG6vyY\ne+cfwaW1g0tlvFTcVSVb/sxUvvLCLiWMdxGjt5JUxV3GaSm9ysHVk5MrTDpp/5qqXes1\n/BOXsD\n2DmS/QSZr/Mt+Vc2baKuxPw1w5YnGVuY6vHxHffABzkn+WPcguabr86JcmIAcC0zc2TLkbufBPJewYka9PIt1Ng2\n83NKe13huPU\nohnryYVIMPyjrTWpDid+yC5kSGVeP0/5+r\nJvLmFZUB/n0RUjgMZU5V++GPU1QnCBa+\n"
	False
	```

=== "Encoded string based checking"

        The following shows a successful check being completed:

        ```python
        >>> import pykx as kx
        >>> license_string = 'Atc/wy/gMjZgIdn1KlT3JVWfVmPk55dtb0YJVes5V4ed9Zxt9UVr8G/A1Q3aWiQEkfjGbwvlJU3GXpUergObvzxGN1iyYG\nZasG5s8vevfAI2ttndt//Y2th\nrryoQRm9Dy+DIIcmSufwomL+\nPMJkZacYc9DM6ipnQsL0KvLwLXLrQC1fBLV2pZHCdYC/nX/KM6uslgip4EoTxZTcx1pQPyTx56QKD4K4JBNimO929w/0+v4Hy2x+DIS3n89vpGmtVvjjFRQtsF6Sjnd+6RnFGk13hRL/DlqHTv2XbZgVv++YOCIc7G55KL6PVJY\npB\n66lq9OiZCEdq2GFJLCn2T\nNWGJPT2s1YDAKsAPI5W3PqJkC2UeV17gPG4gxlCSHr0kfacINbEJ0kSTm/UsuEBZ5B/jvR/jU7rFErcd9PECeQA1kXB19fa4hgvbd+SxWTPxMUKbiHThHk6X0Bi3T7WAQ+sZWsEWwkMncd+mOGS\n3D+bRav2nfOpKckj8rCdvYum3U8PDv6IHP=S+\nLaCnJM0yqNjW9xGyog5ml\nbX2k3mBRyBjbJH/1OWTcIg7uDYxxoMtDOCJjeBdSqI=aK+5FVTVarfowvudv7QsMGeohGaJMyczNWVPPjsbyvsxbAwdXvJUuP0jcFCFVeF\n'
        >>> kx.license.check(license_string, format = 'STRING')
        True
        ```

        The following shows an example of a failed check:

        ```python
        >>> import pykx as kx
	>>> license_string = '8n\nD+HkcJ93xW4oOEtH\nIZxeWkA1glv5wJ5wE2Fsmbc4lg2ntT9JpsclE1hFeG/Ox/jM4=6GjXD2VNpiCAJ80DNVcXuDB+IPEnP22DMGvBIolJt2pdy9kooGZNQpr6svIkRWX/0m/SbydbQOQUVvfNTxsDjZvvsCiGkdQtygs3sDEJbxsT+KfjqJ7Sd6RQ/47HJHG4JyIWdhmvEBVGSLBa5mdAaCLWdCrga3hHZbW3F4e/l3K4nOQvU91WEiMd6PT061r66AOYmjGACCXqmQ9kSsJfMTXPRi9M2i93Oyv895kFVKdZCLCdKdaow790RcjwnKjFFOERGcge=lZdRtp2BL\nA+JbixvTIKTObmfqr7uPYsGQLfXSFnQCq7jbt3yxv1ZPjvjYLPTx7YKIvgo+ITG6vyY\ne+cfwaW1g0tlvFTcVSVb/sxUvvLCLiWMdxGjt5JUxV3GaSm9ysHVk5MrTDpp/5qqXes1\n/BOXsD\n2DmS/QSZr/Mt+Vc2baKuxPw1w5YnGVuY6vHxHffABzkn+WPcguabr86JcmIAcC0zc2TLkbufBPJewYka9PIt1Ng2\n83NKe13huPU\nohnryYVIMPyjrTWpDid+yC5kSGVeP0/5+r\nJvLmFZUB/n0RUjgMZU5V++GPU1QnCBa+\n'
        >>> kx.license.check(license_string, format = 'STRING')
        Supplied license information does not match.
        Please consider reinstalling your license using pykx.util.install_license

        On disk license:
        b'Atc/wy/gMjZgIdn1KlT3JVWfVmPk55dtb0YJVes5V4ed9Zxt9UVr8G/A1Q3aWiQEkfjGbwvlJU3GXpUergObvzxGN1iyYG\nZasG5s8vevfAI2ttndt//Y2th\nrryoQRm9Dy+DIIcmSufwomL+\nPMJkZacYc9DM6ipnQsL0KvLwLXLrQC1fBLV2pZHCdYC/nX/KM6uslgip4EoTxZTcx1pQPyTx56QKD4K4JBNimO929w/0+v4Hy2x+DIS3n89vpGmtVvjjFRQtsF6Sjnd+6RnFGk13hRL/DlqHTv2XbZgVv++YOCIc7G55KL6PVJY\npB\n66lq9OiZCEdq2GFJLCn2T\nNWGJPT2s1YDAKsAPI5W3PqJkC2UeV17gPG4gxlCSHr0kfacINbEJ0kSTm/UsuEBZ5B/jvR/jU7rFErcd9PECeQA1kXB19fa4hgvbd+SxWTPxMUKbiHThHk6X0Bi3T7WAQ+sZWsEWwkMncd+mOGS\n3D+bRav2nfOpKckj8rCdvYum3U8PDv6IHP=S+\nLaCnJM0yqNjW9xGyog5ml\nbX2k3mBRyBjbJH/1OWTcIg7uDYxxoMtDOCJjeBdSqI=aK+5FVTVarfowvudv7QsMGeohGaJMyczNWVPPjsbyvsxbAwdXvJUuP0jcFCFVeF\n'

        Supplied string content:
        b'8n\nD+HkcJ93xW4oOEtH\nIZxeWkA1glv5wJ5wE2Fsmbc4lg2ntT9JpsclE1hFeG/Ox/jM4=6GjXD2VNpiCAJ80DNVcXuDB+IPEnP22DMGvBIolJt2pdy9kooGZNQpr6svIkRWX/0m/SbydbQOQUVvfNTxsDjZvvsCiGkdQtygs3sDEJbxsT+KfjqJ7Sd6RQ/47HJHG4JyIWdhmvEBVGSLBa5mdAaCLWdCrga3hHZbW3F4e/l3K4nOQvU91WEiMd6PT061r66AOYmjGACCXqmQ9kSsJfMTXPRi9M2i93Oyv895kFVKdZCLCdKdaow790RcjwnKjFFOERGcge=lZdRtp2BL\nA+JbixvTIKTObmfqr7uPYsGQLfXSFnQCq7jbt3yxv1ZPjvjYLPTx7YKIvgo+ITG6vyY\ne+cfwaW1g0tlvFTcVSVb/sxUvvLCLiWMdxGjt5JUxV3GaSm9ysHVk5MrTDpp/5qqXes1\n/BOXsD\n2DmS/QSZr/Mt+Vc2baKuxPw1w5YnGVuY6vHxHffABzkn+WPcguabr86JcmIAcC0zc2TLkbufBPJewYka9PIt1Ng2\n83NKe13huPU\nohnryYVIMPyjrTWpDid+yC5kSGVeP0/5+r\nJvLmFZUB/n0RUjgMZU5V++GPU1QnCBa+\n'
        False
        ```

## Environment issues

The following section outlines how a user can get access to a verbose set of environment configuration associated with PyKX. This information is helpful when debugging your environment and should be provided if possible with support requests.


```python
>>> import pykx as kx
>>> kx.util.debug_environment()  # see below for output
```

??? output

	```python
	>>> kx.util.debug_environment()
	missing q binary at '/usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib/m64/q'
	**** PyKX information ****
	pykx.args: ()
	pykx.qhome: /usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib
	pykx.qlic: /usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib
	pykx.licensed: True
	pykx.__version__: 1.5.3rc2.dev525+g41f008ad
	pykx.file: /usr/local/anaconda3/lib/python3.8/site-packages/pykx/util.py

	**** Python information ****
	sys.version: 3.8.3 (default, Jul  2 2020, 11:26:31) 
	[Clang 10.0.0 ]
	pandas: 1.5.3
	numpy: 1.24.4
	pytz: 2022.7.1
	which python: /usr/local/anaconda3/bin/python
	which python3: /usr/local/anaconda3/bin/python3

	**** Platform information ****
	platform.platform: macOS-10.16-x86_64-i386-64bit

	**** Environment Variables ****
	IGNORE_QHOME: 
	PYKX_IGNORE_QHOME: 
	PYKX_KEEP_LOCAL_TIMES: 
	PYKX_ALLOCATOR: 
	PYKX_GC: 
	PYKX_LOAD_PYARROW_UNSAFE: 
	PYKX_MAX_ERROR_LENGTH: 
	PYKX_NOQCE: 
	PYKX_Q_LIB_LOCATION: 
	PYKX_RELEASE_GIL: 
	PYKX_Q_LOCK: 
	PYKX_ENABLE_PANDAS_API: 
	QARGS: 
	QHOME: /usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib
	QLIC: 
	PYKX_DEFAULT_CONVERSION: 
	PYKX_SKIP_UNDERQ: 
	PYKX_UNSET_GLOBALS: 
	SKIP_UNDERQ: 
	UNSET_PYKX_GLOBALS: 

	**** License information ****
	pykx.qlic directory: True
	pykx.lic writable: True
	pykx.qhome lics: ['kc.lic']
	pykx.qlic lics: ['kc.lic']

	**** q information ****
	which q: /usr/local/anaconda3/bin/q
	q info: 
	```

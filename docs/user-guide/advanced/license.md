---
title: Manage your license
description: Tips and tricks for managing licenses
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, license, licenses
---

# Manage your license

_Tips and tricks for managing licenses_

In the [install page](../../getting-started/installing.md) you can follow along with how to install your first license, this is an important first step for you to get the most out of the functionality provided by PyKX. However, there are a number of cases where upgrading this license will be necessary:

1. Your license has expired
2. You need to upgrade from a personal to enterprise/commercial license

## When will your license expire?

For personal edition licenses your license will expire one year after initial download, for enterprise/commercial licenses the expiry date will vary. To provide some forewarning of when your license will expire PyKX provides the following:

- For the 10 days before expiry PyKX will print a message at start up indicating the license will expire in N days
- The utility function `#!python kx.license.expires` provides a programmatic method of finding days to expiry.

As an example the following shows you both in action:

=== "Expiring soon"

	```python
	>>> import pykx as kx
	WARNING:root:PyKX license set to expire in 8 days, please consider installing an updated license
	```

=== "Checking expiry"

	```python
	>>> import pykx as kx
	>>> kx.license.expires()
	8
	```

## Update a license

When your license is expired/expiring you will need to update it to ensure you can continue to use the software.

There are three methods by which updating your license is possible with PyKX.

- You have allowed your license to expire and on restart of PyKX you will be presented with an new license install walkthrough similar to [installing](../../getting-started/installing.md).
- You pre-emptively install a newly downloaded license using `#!python kx.license.install`.

=== "After Expiry"

	Now that your license has expired importing PyKX will result in the following walkthrough being presented, following this will allow you to install a new license.

	```python
	>>> import pykx as kx
	Your PyKX license has now expired.

	Captured output from initialization attempt:
	    '2023.10.18T13:27:59.719 licence error: exp

	License location used:
	/usr/local/anaconda3/pykx/kc.lic

	Would you like to renew your license? [Y/n]: Y
	
        Do you have access to an existing license for PyKX that you would like to use? [N/y]:
	```

=== "Pre-emptive install"

	If you have downloaded your new license prior to expiry you can install it with `#!python kx.license.install`.

	- Install an updated `kc.lic` license from a file

		```python
		>>> import pykx as kx
		>>> kx.license.install('/tmp/new/location/kc.lic', force=True)
		```

	- Install an updated `k4.lic` license from the base64 license key

		```python
		>>> import pykx as kx
		# String truncated for display purposes
		>>> b64key = 'dajsi8d9asnhda8sld..'
		>>> kx.license.install(b64key,
		...                    format='STRING',
		...                    license_type='k4.lic',
		...                    force=True)
		```

## Upgrade to a commercial license

If you are currently using a `kc.lic` personal license and need to upgrade to a `k4.lic` license the following steps allow you to ensure this can be done effectively.

1. Delete your existing `kc.lic` license.
2. Install your new license following the [license installation](../../getting-started/installing.md)

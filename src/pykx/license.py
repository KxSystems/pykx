import base64
import os
import shutil
from pathlib import Path
from typing import Optional

from . import licensed
from .config import lic_type, qlic


__all__ = [
    'check',
    'expires',
    'install',
]


def __dir__():
    return sorted(__all__)


def _init(_q):
    global q
    q = _q


def check(license: str,
          *,
          format: Optional[str] = 'FILE',
          license_type: Optional[str] = lic_type
) -> bool:
    """
    Validate the license key information you provided matches the license
        saved to disk which is read by PyKX

    Parameters:
        license: If using "FILE" format this is the location of the file being used for comparison.
            If "STRING" this is the base64 encoded string provided in your license email
        format: Is the license check being completed using a downloaded file or base64
            encoded string. Accepted inputs are "FILE"(default) or "STRING".
        license_type: The license file type/name which is to be checked, by default this
            is 'kc.lic' which is the version provided with personal and commercial
            evaluation licenses but can be changed to 'k4.lic' or 'kx.lic' if appropriate

    Returns:
        A boolean indicating if the license is correct or not and a printed message describing
            the issue

    Examples:

    Validate that a provided license matches an existing persisted license

    ```python
    >>> import pykx as kx
    >>> kx.license.check('/usr/location/kc.lic')
    True
    ```

    Attempt to check a new license against an existing installed license

    ```python
    >>> import pykx as kx
    >>> check = kx.license.check('/usr/location/kc.lic')
    Supplied license information does not match.
    Please consider reinstalling your license using pykx.license.install

    Installed license representation:
    b'iIXSiEWzCNTkkCWK5Gggy..'
    User expected license representation:
    b'IyEvdXNyL2Jpbi9lbngDf..'
    >>> check
    False
    ```

    Attempt to check a license in the case no license is currently installed

    ```python
    >>> import pykx as kx
    >>> check = kx.license.check('setup.py', license_type='kc.lic')
    Unable to find an installed license: kc.lic at location: /usr/local/anaconda3/envs/qenv/q.
    Please consider installing your license again using pykx.license.install
    >>> check
    False
    ```
    """
    format = format.lower()
    if format not in ('file', 'string'):
        raise Exception('Unsupported option provided for format parameter')

    if license_type not in ('kx.lic', 'kc.lic', 'k4.lic'):
        raise Exception(f'License type {license_type} not supported.')

    license_located = False
    installed_lic = qlic/license_type
    if os.path.exists(installed_lic):
        license_located = True

    if not license_located:
        print(f'Unable to find an installed license: {license_type} at location: {str(qlic)}.\n'
              'Please consider installing your license again using pykx.license.install')
        return False

    with open(installed_lic, 'rb') as f:
        license_content = base64.encodebytes(f.read()).decode('utf-8')
        license_content = license_content.replace('\n', '')
        license_content = bytes(license_content, 'utf-8')

    if format == 'file':
        license_path = Path(os.path.expanduser(license))
        if os.path.exists(license_path):
            with open(str(license_path), 'rb') as f:
                license = base64.encodebytes(f.read()).decode('utf-8')
                license = license.replace('\n', '')
        else:
            print(f'Unable to locate license {license_path} for comparison')
            return False

    if isinstance(license, str):
        license = license.replace('\n', '')
        license = bytes(license, 'utf-8')

    if license_content != license:
        print('Supplied license information does not match.\n'
              'Please consider reinstalling your license using pykx.license.install\n\n'
              f'Installed license representation:\n{license_content}\n\n'
              f'User expected license representation:\n{license}')
        return False

    return True


def expires() -> int:
    """
    The number of days until the license is set to expire.

    Returns:
        The number of days until the license is set to expire

    Example:

    The number of days until your license expires:

    ```python
    >>> import pykx as kx
    >>> kx.license.expires()
    265
    ```
    """
    if not licensed:
        raise Exception('Unlicensed user, unable to determine license expiry')
    return (q('"D"$', q.z.l[1]) - q.z.D).py()


def install(license: str,
            *,
            format: Optional[str] = 'FILE',
            license_type: Optional[str] = 'kc.lic',
            force: Optional[bool] = False
) -> bool:
    """
    (Re)install a KX license key, optionally overwriting the currently installed license.

    Parameters:
        license: If using "FILE" this is the location of the file being used for comparison.
            If "STRING" this is the base64 encoded string provided in your license email
        format: Is the license check being completed using a downloaded file or base64
            encoded string. Accepted inputs are "FILE"(default) or "STRING".
        license_type: The license file type/name which is to be checked, by default this
            is 'kc.lic' which is the version provided with personal and commercial
            evaluation licenses but can be changed to 'k4.lic' or 'kx.lic' if appropriate
        force: Enforce overwrite without opt-in message for overwrite

    Returns:
        A boolean indicating if the license has been correctly overwritten

    Examples:

    Install a license using a supplied file location

    ```python
    >>> import pykx as kx
    >>> kx.license.install('/path/to/license')
    True
    ```

    Install a k4.lic base64 encoded string representation of the license

    ```python
    >>> import pykx as kx
    >>> b64_string = 'IdyannfDangfa4FasdjD9fcda' # Example
    >>> kx.license.install(b64_string, format = "STRING", license_type = "k4.lic")
    True
    ```
    """
    format = format.lower()
    if format not in ('file', 'string'):
        raise Exception('Unsupported option provided for format parameter')

    license_located = False
    installed_lic = qlic/license_type
    if os.path.exists(installed_lic):
        license_located = True

    if license_located and not force:
        raise Exception(f'Installed license: {license_type} at location: {str(qlic)} '
                        'detected. to overwrite currently installed license use parameter '
                        'force=True')

    if format == 'file':
        download_location = os.path.expanduser(Path(license))

        if not os.path.exists(download_location):
            raise Exception(f'Download location provided {download_location} does not exist.')

        shutil.copy(download_location, qlic)
    else:
        try:
            lic = base64.b64decode(license)
        except base64.binascii.Error:
            raise Exception('Invalid license copy provided, '
                            'please ensure you have copied the license information correctly')

        with open(qlic/license_type, 'wb') as binary_file:
            binary_file.write(lic)
    print("PyKX license successfully installed!")
    return True

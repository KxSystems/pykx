import base64
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
from warnings import warn
import webbrowser

import toml
import pandas as pd

from .exceptions import PyKXWarning, QError


system = platform.system()
# q expects (m|l|w)64 to exist under whatever QHOME was when qinit was executed.
q_lib_dir_name = {
    'Darwin': 'm64',
    'Linux': 'l64',
    'Windows': 'w64',
}[system]
if 'Darwin' in system and 'arm' in platform.machine():
    q_lib_dir_name = 'm64arm'
if 'Linux' in system and ('arm' in platform.machine() or 'aarch64' in platform.machine()):
    q_lib_dir_name = 'l64arm'
tcore_path_location = bytes(Path(__file__).parent.resolve(strict=True) / '_tcore.so')


# Profile information for user defined config
# If PYKX_CONFIGURATION_LOCATION is not set it will search '.'
pykx_config_location = Path(os.getenv('PYKX_CONFIGURATION_LOCATION', ''))
pykx_config_profile = os.getenv('PYKX_PROFILE', 'default')


def _get_config_value(param, default):
    try:
        default = _pykx_profile_content[param]
    except KeyError:
        pass
    except NameError:
        pass
    return os.getenv(param, default)


def _is_enabled(param, cmdflag=False, deprecated=False):
    env_config = _get_config_value(param, '').lower() in ('1', 'true')
    if deprecated and env_config:
        warn('The environment variable ' + param + ' is deprecated.\n'
             'See https://code.kx.com/pykx/user-guide/configuration.html\n'
             'for more information.',
             DeprecationWarning)
    return env_config or (cmdflag and cmdflag in qargs)


def _is_set(envvar):
    return os.getenv(envvar, None)


pykx_config_locs = [Path('.'), pykx_config_location, Path.home()]
for path in pykx_config_locs:
    config_path = path / '.pykx-config'
    if os.path.isfile(config_path):
        _pykx_config_content = toml.load(config_path)
        try:
            _pykx_profile_content = _pykx_config_content[pykx_config_profile]
            break
        except KeyError:
            print("Unable to locate specified 'PYKX_PROFILE': '" + pykx_config_profile + "' in file '" + config_path + "'") # noqa E501


pykx_dir = Path(__file__).parent.resolve(strict=True)
os.environ['PYKX_DIR'] = str(pykx_dir)
os.environ['PYKX_EXECUTABLE'] = sys.executable
pykx_libs_dir = Path(pykx_dir/'lib'/'4-1-libs') if _is_enabled('PYKX_4_1_ENABLED') else Path(pykx_dir/'lib') # noqa
pykx_lib_dir = Path(_get_config_value('PYKX_Q_LIB_LOCATION', pykx_libs_dir))
pykx_platlib_dir = pykx_lib_dir/q_lib_dir_name
lib_prefix = '' if system == 'Windows' else 'lib'
lib_ext = {
    'Darwin': 'dylib',
    'Linux': 'so',
    'Windows': 'dll',
}[system]

try:
    qhome = Path(_get_config_value('QHOME', pykx_lib_dir)).resolve(strict=True)
except FileNotFoundError: # nocov
    # If QHOME and its fallback weren't set/valid, then q/Python must be
    # running in the same directory as q.k (and presumably other stuff one
    # would expect to find in QHOME).
    qhome = Path().resolve(strict=True)

# License search
_qlic = os.getenv('QLIC', '')
_pwd = os.getcwd()
license_located = False
lic_path = ''
lic_type = ''
for loc in (_pwd, _qlic, qhome):
    if loc=='':
        pass
    for lic in ('kx.lic', 'kc.lic', 'k4.lic'):
        try:
            lic_path = Path(str(loc) + '/' + lic).resolve(strict=True)
            license_located=True
            lic_type = lic
            qlic=Path(loc)
        except FileNotFoundError:
            continue
        if license_located:
            break
    if license_located:
        break

if not license_located:
    qlic = Path(qhome)

qargs = tuple(shlex.split(_get_config_value('QARGS', '')))


def _license_install_B64(license, license_type):
    try:
        lic = base64.b64decode(license)
    except base64.binascii.Error:
        raise Exception('Invalid license copy provided, '
                        'please ensure you have copied the license information correctly')

    with open(qlic/license_type, 'wb') as binary_file:
        binary_file.write(lic)
    return True


def _license_check(lic_type, lic_encoding, lic_variable):
    license_content = None
    lic_name = lic_type + '.lic'
    lic_file = qlic / lic_name
    if os.path.exists(lic_file):
        with open(lic_file, 'rb') as f:
            license_content = base64.encodebytes(f.read()).decode('utf-8')
            license_content = license_content.replace('\n', '')
    if lic_encoding == license_content:
        conflict_message = 'We have been unable to update your license for PyKX using '\
                           'the following information:\n'\
                           f"  Environment variable: {lic_variable} \n"\
                           f'  License location: {qlic}/{lic_type}.lic\n'\
                           'Reason: License content matches supplied Environment variable'
        print(conflict_message)
        return False
    else:
        return _license_install_B64(lic_encoding, lic_name)

def _license_install(intro=None, return_value=False, license_check=False, license_error=None): # noqa: 

    if license_check:
        install_success = False
        kc_b64 = _get_config_value('KDB_LICENSE_B64', None)
        k4_b64 = _get_config_value('KDB_K4LICENSE_B64', None)

        if kc_b64 is not None:
            kx_license_env = 'KDB_LICENSE_B64'
            kx_license_file = 'kc'
            install_success = _license_check(kx_license_file, kc_b64, kx_license_env)
        elif k4_b64 is not None:
            kx_license_env = 'KDB_K4LICENSE_B64'
            kx_license_file = 'k4'
            install_success = _license_check(kx_license_file, k4_b64, kx_license_env)
        if install_success:
            if license_error is not None:
                install_message = f'Initialisation failed with error: {license_error}\n'\
                                  'Your license has been updated using the following '\
                                  'information:\n'\
                                  f'  Environment variable: {kx_license_env}\n'\
                                  f'  License write location: {qlic}/{kx_license_file}.lic'
                print(install_message)
            return True

    modes_url = "https://code.kx.com/pykx/user-guide/advanced/modes.html"
    personal_url = "https://kx.com/kdb-insights-personal-edition-license-download"
    commercial_url = "https://kx.com/book-demo"
    unlicensed_message = '\nPyKX unlicensed mode enabled. To set this as your default behavior '\
                         "set the following environment variable PYKX_UNLICENSED='true'"\
                         '\n\nFor more information on PyKX modes of operation, visit '\
                         f'{modes_url}.\nTo apply for a PyKX license visit '\
                         f'\n\n   Personal License:   {personal_url}'\
                         '\n   Commercial License: Contact your KX sales representative '\
                         f'or sales@kx.com or apply on {commercial_url}'
    first_user = '\nThank you for installing PyKX!\n\n'\
                 'We have been unable to locate your license for PyKX. '\
                 'Running PyKX in unlicensed mode has reduced functionality.\n'\
                 'Would you like to continue with license installation? [Y/n]: '
    continue_license = input(first_user if intro is None else intro)
    if continue_license in ('n', 'N'):
        os.environ['PYKX_UNLICENSED']='true'
        print(unlicensed_message)
        if return_value:
            return False

    elif continue_license in ('y', 'Y', ''):
        commercial = input('\nIs the intended use of this software for:'
                           '\n    [1] Personal use (Default)'
                           '\n    [2] Commercial use'
                           '\nEnter your choice here [1/2]: ').strip().lower()
        if commercial not in ('1', '2', ''):
            raise Exception('User provided option was not one of [1/2]')

        personal = commercial in ('1', '')

        lic_url = personal_url if personal else commercial_url
        lic_type = 'kc.lic' if personal else 'k4.lic'

        if personal:
            redirect = input(f'\nTo apply for your PyKX license, navigate to {lic_url}.\n'
                             'Shortly after you submit your application, you will receive a '
                             'welcome email containing your license information.\n'
                             'Would you like to open this page? [Y/n]: ')
        else:
            redirect = input('\nTo apply for your PyKX license, contact your '
                             'KX sales representative or sales@kx.com.\n'
                             f'Alternately apply through {lic_url}.\n'
                             'Would you like to open this page? [Y/n]: ')

        if redirect.lower() in ('y', ''):
            try:
                webbrowser.open(lic_url)
                time.sleep(2)
            except BaseException:
                raise Exception('Unable to open web browser')

        install_type = input('\nPlease select the method you wish to use to activate your license:'
                             '\n  [1] Download the license file provided in your welcome email and '
                             'input the file path (Default)'
                             '\n  [2] Input the activation key (base64 encoded string) provided in '
                             'your welcome email'
                             '\n  [3] Proceed with unlicensed mode'
                             '\nEnter your choice here [1/2/3]: ').strip().lower()

        if install_type not in ('1', '2', '3', ''):
            raise Exception('User provided option was not one of [1/2/3]')

        if install_type in ('1', ''):
            license = input('\nProvide the download location of your license '
                            f'(for example, ~/path/to/{lic_type}) : ').strip()
            download_location = os.path.expanduser(Path(license))

            if not os.path.exists(download_location):
                raise Exception(f'Download location provided {download_location} does not exist.')

            shutil.copy(download_location, qlic)
            print('\nPyKX license successfully installed. Restart Python for this to take effect.\n') # noqa: E501
        elif install_type == '2':

            license = input('\nProvide your activation key (base64 encoded string) '
                            'provided with your welcome email : ').strip()

            _license_install_B64(license, lic_type)

            print('\nPyKX license successfully installed. Restart Python for this to take effect.\n') # noqa: E501
        elif install_type == '3':
            if return_value:
                return False
    else:
        raise Exception('Invalid input provided please try again')
    if return_value:
        return True


_arglist = ['--unlicensed', '--licensed']
_licenvset = _is_enabled('PYKX_LICENSED', '--licensed') or _is_enabled('PYKX_UNLICENSED', '--unlicensed') # noqa: E501
if any(i in qargs for i in _arglist) or _licenvset or not hasattr(sys, 'ps1'): # noqa: C901
    pass
elif not license_located:
    _license_install()

licensed = False

under_q = _is_enabled('PYKX_UNDER_Q')
qlib_location = Path(_get_config_value('PYKX_Q_LIB_LOCATION', pykx_libs_dir))
pykx_threading = _is_enabled('PYKX_THREADING')
if platform.system() == 'Windows' and pykx_threading:
    pykx_threading = False
    warn('PYKX_THREADING is only supported on Linux / MacOS, it has been disabled.')
no_sigint = _is_enabled('PYKX_NO_SIGINT', deprecated=True)
no_pykx_signal = _is_enabled('PYKX_NO_SIGNAL')

if _is_enabled('PYKX_ENABLE_PANDAS_API', '--pandas-api'):
    warn('Usage of PYKX_ENABLE_PANDAS_API configuration variable was removed in '
         'PyKX 2.0. Pandas API is permanently enabled. See: '
         'https://code.kx.com/pykx/changelog.html#pykx-200')

ignore_qhome = _is_enabled('IGNORE_QHOME', '--ignore-qhome', True) or _is_enabled('PYKX_IGNORE_QHOME') # noqa E501
keep_local_times = _is_enabled('KEEP_LOCAL_TIMES', '--keep-local-times', True) or _is_enabled('PYKX_KEEP_LOCAL_TIMES') # noqa E501
max_error_length = int(_get_config_value('PYKX_MAX_ERROR_LENGTH', 256))

if _is_enabled('PYKX_ALLOCATOR', '--pykxalloc'):
    if sys.version_info[1] <= 7:
        raise PyKXWarning('A python version of at least 3.8 is required to use the PyKX allocators') # noqa nocov
        k_allocator = False  # nocov
    else:
        k_allocator = True
else:
    k_allocator = False

k_gc = _is_enabled('PYKX_GC', '--pykxgc')
release_gil = _is_enabled('PYKX_RELEASE_GIL', '--release-gil')
use_q_lock = _get_config_value('PYKX_Q_LOCK', False)
skip_under_q = _is_enabled('SKIP_UNDERQ', '--skip-under-q') or _is_enabled('PYKX_SKIP_UNDERQ')
no_qce = _is_enabled('PYKX_NOQCE', '--no-qce')
beta_features = _is_enabled('PYKX_BETA_FEATURES', '--beta')
load_pyarrow_unsafe = _is_enabled('PYKX_LOAD_PYARROW_UNSAFE', '--load-pyarrow-unsafe')
pykx_qdebug = _is_enabled('PYKX_QDEBUG', '--q-debug')

pandas_2 = pd.__version__.split('.')[0] == '2'


def find_core_lib(name: str) -> Path:
    suffix = '.dll' if system == 'Windows' else '.so'
    path = pykx_platlib_dir/f'{lib_prefix}{name}'
    try:
        return path.with_suffix(suffix).resolve(strict=True)
    except FileNotFoundError: # nocov
        if system == 'Darwin' and suffix == '.so':
            return path.with_suffix('.dylib').resolve(strict=True)
        raise


def _set_licensed(licensed_):
    global licensed
    licensed = licensed_


def _set_keep_local_times(keep_local_times_):
    global keep_local_times
    keep_local_times = keep_local_times_


def _check_beta(feature_name, *, status=beta_features):
    if status:
        return None
    raise QError(f'Attempting to use a beta feature "{feature_name}'
                 '", please set configuration flag PYKX_BETA_FEATURES=true '
                 'to run these operations')


__all__ = [
    'system',
    'q_lib_dir_name',
    'pykx_dir',
    'pykx_lib_dir',
    'pykx_platlib_dir',
    'lib_prefix',
    'lib_ext',

    'qhome',
    'qlic',
    'qargs',
    'licensed',
    'under_q',

    'ignore_qhome',
    'keep_local_times',
    'max_error_length',

    'k_allocator',
    'k_gc',
    'release_gil',
    'use_q_lock',
    'skip_under_q',
    'no_qce',
    'load_pyarrow_unsafe',

    'find_core_lib',

    'pandas_2',
]


def __dir__():
    return sorted(__all__)

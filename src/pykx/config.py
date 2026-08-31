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
import numpy as np

from .exceptions import QError


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
pykx_config_location = os.path.expanduser(os.getenv('PYKX_CONFIGURATION_LOCATION', ''))
pykx_config_profile = os.getenv('PYKX_PROFILE', 'default')
unlicensed_reason = None


def _get_config_value(param, default):
    try:
        default = pykx_profile_content[param]
    except KeyError:
        pass
    except NameError:
        pass
    return os.getenv(param, default)


def _is_enabled(param, cmdflag=False):
    val = _get_config_value(param, '')
    if isinstance(val, (bool, int)):
        env_config = val
    else:
        env_config = val.lower() in ('1', 'true')
    return env_config or (cmdflag and cmdflag in qargs)


pykx_config_locs = [Path.home()/'.kx/config-pykx']
if pykx_config_location != '':
    pykx_config_locs = [Path(pykx_config_location)] + pykx_config_locs
else:
    pykx_config_location = None
pykx_config_locs = [os.path.abspath(path) for path in pykx_config_locs if os.path.isfile(path)]
pykx_config_locs = list(set(pykx_config_locs))

pykx_config_content = None
pykx_profile_content = {}

for path in pykx_config_locs:
    pykx_config_content = toml.load(path)
    try:
        pykx_profile_content = pykx_config_content[pykx_config_profile]
        pykx_config_location = path
        break
    except KeyError:
        pykx_profile_content = {}
        pykx_config_location = None
        print("Unable to locate specified 'PYKX_PROFILE': '" + pykx_config_profile + "' in file '" + str(path) + "'") # noqa E501


pykx_dir = Path(__file__).parent.resolve(strict=True)
os.environ['PYKX_DIR'] = str(pykx_dir)
pykx_executable = sys.executable
os.environ['PYKX_EXECUTABLE'] = pykx_executable

pykx_libs_dir = Path(pykx_dir/'lib')

pykx_lib_dir = Path(_get_config_value('PYKX_Q_LIB_LOCATION', pykx_libs_dir))
pykx_platlib_dir = pykx_lib_dir/q_lib_dir_name
lib_prefix = '' if system == 'Windows' else 'lib'
lib_ext = {
    'Darwin': 'dylib',
    'Linux': 'so',
    'Windows': 'dll',
}[system]


def _remove_outer_doublequotes(s):
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def _get_qcfg_values():
    qcfg_values = {}
    qcfg_path = None

    qargs_str = _get_config_value('QARGS', '')
    qargs_list = shlex.split(qargs_str)

    for i, arg in enumerate(qargs_list):
        if arg == '-v' and i + 1 < len(qargs_list):
            qcfg_path = qargs_list[i + 1]
            break

    if qcfg_path is None:
        qcfg_path = os.getenv('QCFG')

    if qcfg_path is None:
        qcfg_path = Path.home()/'.kx/config'

    if qcfg_path:
        try:
            qcfg_path = Path(qcfg_path).resolve(strict=True)

            with open(qcfg_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        key = _remove_outer_doublequotes(key)
                        value = _remove_outer_doublequotes(value)

                        if key in ('QHOME', 'QLIC'):
                            qcfg_values[key] = value

        except (IOError):
            # Error or Warning to handle failing QCFG
            pass

    return qcfg_values


_qcfg_values = _get_qcfg_values()


def _get_qhome():
    qhome_value = _get_config_value('QHOME', None)

    if qhome_value:
        try:
            qhome = Path(qhome_value).resolve(strict=True)
            return qhome
        except FileNotFoundError:
            pass

    kx_path = Path.home()/'.kx'

    if kx_path.exists() and kx_path.is_dir():
        return kx_path.resolve(strict=True)

    return Path(pykx_lib_dir).resolve(strict=True)


def _get_qpath(qhome):
    qpath = _get_config_value('QPATH', None)
    mod = str(qhome / 'mod')
    if qpath is None or qpath == '':
        result = mod
    elif mod in qpath.split(os.pathsep):
        result = qpath
    else:
        result = qpath + os.pathsep + mod
    os.environ['QPATH'] = result
    return result


if 'QHOME' in _qcfg_values:
    try:
        qhome = Path(_qcfg_values['QHOME']).resolve(strict=True)
    except FileNotFoundError:
        warn(f"QHOME from QCFG does not exist: {_qcfg_values['QHOME']}")
        qhome = Path(_qcfg_values['QHOME']).resolve(strict=False)
else:
    qhome = _get_qhome()

qpath = _get_qpath(qhome)
os.environ['PYKX_OLD_QHOME'] = str(qhome) if qhome is not None else ""
os.environ['PYKX_OLD_QPATH'] = str(qpath) if qpath is not None else ""

# License search
if 'QLIC' in _qcfg_values:
    _qlic = _qcfg_values['QLIC']
    if not os.path.isdir(_qlic):
        warn(f'QCFG value QLIC set to non directory value: {_qlic}')
else:
    _qlic = _get_config_value('QLIC', '')
    if (_qlic != '') and (not os.path.isdir(_qlic)):
        warn(f'Configuration value QLIC set to non directory value: {_qlic}')

_kc_lic = 'kc.lic'
_k4_lic = 'k4.lic'
_kx_lic = 'kx.lic'

license_located = False
lic_path = ''
lic_type = ''
for loc in (_qlic, qhome):
    if loc=='':
        pass
    for lic in (_kx_lic, _kc_lic, _k4_lic):
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

light_load = _is_enabled('PYKX_LIGHT_LOAD')
under_q = _is_enabled('PYKX_UNDER_Q')
suppress_warnings = _is_enabled('PYKX_SUPPRESS_WARNINGS') or light_load

_unsupported_qargs = {
    '-p': 'KDB-X Python running without a main loop, setting a port in this way is not supported',
    '-t': 'KDB-X Python running without a main loop, setting timers in this way has no effect'
}


def _check_qargs():
    qargs = shlex.split(_get_config_value('QARGS', ''))
    if (not under_q) and (not suppress_warnings):
        for i in list(_unsupported_qargs.keys()):
            if i in qargs:
                warn(f"'{i}' argument unsupported in QARGS configuration: {_unsupported_qargs[i]}",
                     RuntimeWarning)
    return tuple(qargs)


qargs = '' if light_load else _check_qargs()


def _license_install_path(download_location, qlic):
    if not os.path.exists(download_location):
        raise Exception(f'Download location provided {download_location} does not exist.')
    lic_types = ('kx.lic', 'kc.lic', 'k4.lic')
    if os.path.isdir(download_location):
        found_license = None
        for lic in lic_types:
            if lic in os.listdir(download_location):
                found_license = lic
                break
        if found_license is not None:
            lic_type = found_license
            download_location += '/' + lic_type
        else:
            raise ValueError("No license detected in given directory.")
    else:
        lic_type = os.path.basename(download_location)
        if lic_type not in lic_types:
            raise ValueError(f"Supplied licence file {lic_type} not \
                                valid. Must be one of:{lic_types}")

    shutil.copy(download_location, qlic)
    print(f'\nKDB-X Python license successfully installed to: {qlic / lic_type}\n')


def _license_install_B64(license, license_type): # pragma: no cover
    try:
        lic = base64.b64decode(license)
    except base64.binascii.Error:
        raise Exception('Invalid license copy provided, '
                        'please ensure you have copied the license information correctly')

    with open(qlic/license_type, 'wb') as binary_file:
        binary_file.write(lic)
    return True


def _license_check(lic_type, lic_encoding, lic_variable): # pragma: no cover
    global unlicensed_reason
    license_content = None
    lic_name = lic_type + '.lic'
    lic_file = qlic / lic_name
    if os.path.exists(lic_file):
        with open(lic_file, 'rb') as f:
            license_content = base64.encodebytes(f.read()).decode('utf-8')
            license_content = license_content.replace('\n', '')
    if lic_encoding == license_content:
        conflict_message = 'We have been unable to update your license for KDB-X Python using '\
                           'the following information:\n'\
                           f"  Environment variable: {lic_variable} \n"\
                           f'  License location: {qlic}/{lic_type}.lic\n'\
                           'Reason: License content matches supplied Environment variable'
        print(conflict_message)
        unlicensed_reason = f"Updating license failed. {lic_variable} contents matched contents of {qlic}/{lic_type}.lic." # noqa: E501
        return False
    else:
        return _license_install_B64(lic_encoding, lic_name)


def _unlicensed_config(unlicensed_message):
    choice = input('\nWould you like us to remember this choice? [N/y]: ')
    if choice in ('y', 'Y'):
        if pykx_config_location is None:
            fpath = Path.home()/'.kx/config-pykx'
        else:
            fpath = pykx_config_location
        try:
            os.access(fpath, os.W_OK)
        except FileNotFoundError:
            pass
        except PermissionError:
            raise PermissionError(f"You do not have sufficient permissions to write to: {fpath}")
        if os.path.exists(fpath):
            with open(fpath, 'r') as file:
                data = toml.load(file)
        else:
            data = {pykx_config_profile: {}}
        data[pykx_config_profile]['PYKX_UNLICENSED'] = 'True'
        with open(fpath, 'w') as file:
            toml.dump(data, file)
            print(f"\nConfiguration updated at: {fpath}.\n"
                  "Unlicensed mode now set as default behavior.")
    else:
        print(unlicensed_message)
    os.environ['PYKX_UNLICENSED']='true'


def _license_install(intro=None, return_value=False, license_check=False, license_error=None): # noqa:
    global unlicensed_reason
    if not hasattr(sys, 'ps1'):  # Exit if running in a non-interactive session
        return False

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

    lic_url = 'https://developer.kx.com/products/kdb-x/install'
    lic_type = 'kc.lic'
    unlicensed_message = '\nKDB-X Python unlicensed mode enabled. To set this as your default '\
                         "behavior set the following environment variable PYKX_UNLICENSED='true'"
    first_user = '\nThank you for installing KDB-X Python!\n\n'\
                 'We have been unable to locate your license for KDB-X Python.\n\n'\
                 'Paths searched:\n'\
                 f'    QLIC     {_qlic if _qlic else "Not Set"}\n'\
                 f'    QHOME    {qhome if qhome else "Not Set"}\n\n'\
                 'Running KDB-X Python in unlicensed mode has reduced functionality.\n'\
                 'Would you like to install a license? [Y/n]: '
    unlicensed_reason = "No license was found."
    root = 'C:\\path\\to\\' if platform.system() == 'Windows' else '~/path/to/'
    continue_license = input(first_user if intro is None else intro)
    if continue_license in ('n', 'N'):
        _unlicensed_config(unlicensed_message)
        if return_value:
            return False

    elif continue_license in ('y', 'Y', ''):
        existing_license = input('\nDo you have access to an existing license for KDB-X Python '
                                 'that you would like to use? [N/y]: ')
        if existing_license not in ('Y', 'y', 'N', 'n', ''):
            raise Exception('Invalid input provided please try again')
        if existing_license in ('N', 'n', ''):
            redirect = input(f'\nFor instructions and to obtain a license visit: {lic_url}\n'
                             'Would you like to open this page? [Y/n]: ')
            if redirect.lower() in ('y', ''):
                try:
                    webbrowser.open(lic_url)
                    time.sleep(2)
                except BaseException:
                    raise Exception('Unable to open web browser')

            install_type = input('\nPlease select the method you wish to use to activate your '
                                 'license:\n  [1] Input the license key (base64 encoded string) '
                                 'provided on the KX Developer Center'
                                 '\n  [2] Proceed with unlicensed mode'
                                 '\nEnter your choice here [1/2]: ').strip().lower()

            if install_type not in ('1', '2', ''):
                raise Exception('User provided option was not one of [1/2]')

            if install_type in ('1', ''):
                license = input('\nProvide your license key (base64 encoded string) '
                                'provided on the KX Developer Center: ').strip()
                _license_install_B64(license, lic_type)

                print('\nKDB-X Python license successfully installed to: {qlic / lic_type}\n') # noqa: E501
            elif install_type == '2':
                _unlicensed_config(unlicensed_message)
                if return_value:
                    return False
        else:
            install_type = input(
                '\nPlease select the method you wish to use to activate your license:\n'
                '    [1] Provide the location of your license file\n'
                '    [2] Paste the license key\n'
                'Enter your choice here [1/2]: ')

            if install_type not in ('1', '2', ''):
                raise Exception('User provided option was not one of [1/2]')
            if install_type in ('1', ''):
                license = input('\nProvide the download location of your license '
                                f'(for example, {root}kc.lic) : ').strip()
                download_location = os.path.expanduser(Path(license))
                _license_install_path(download_location, qlic)

            else:
                lic_type_choice = input('\nPlease confirm the license type:\n'
                                        f'    [1] {_kc_lic} - The default\n'
                                        f'    [2] {_k4_lic} - Used in some scenarios\n'
                                        'Enter your choice here [1/2]: ')
                if lic_type_choice not in ('1', '2', ''):
                    raise Exception('User provided option was not one of [1/2]')

                lic_type = _k4_lic if lic_type_choice == '2' else _kc_lic
                license = input('f\nProvide your {lic_type} license key (base64 encoded string) : ').strip() # noqa: E501

                _license_install_B64(license, lic_type)

                print(f'\nKDB-X Python license successfully installed to: {qlic / lic_type}\n')  # noqa: E501

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

_pykx_force_unlicensed = ('--unlicensed' in qargs or _is_enabled('PYKX_UNLICENSED', '--unlicensed')) or light_load # noqa: E501

try:
    config_unlicensed_flag = pykx_profile_content['PYKX_UNLICENSED']
except KeyError:
    config_unlicensed_flag = False

if _pykx_force_unlicensed:
    if '--unlicensed' in qargs:
        unlicensed_reason = ("--unlicensed passed to args.")
    elif light_load:
        unlicensed_reason = ("light_load enabled.")
    elif os.environ.get('PYKX_UNLICENSED') and unlicensed_reason is None:
        unlicensed_reason = ("The environment variable PYKX_UNLICENSED is set to True.")
    elif config_unlicensed_flag:
        unlicensed_reason = (f"PYKX_UNLICENSED is set to True in the config file {pykx_config_location}.") # noqa: E501

_pykx_force_licensed = ('--licensed' in qargs or _is_enabled('PYKX_LICENSED', '--licensed')) and not light_load # noqa: E501

qlib_location = Path(_get_config_value('PYKX_Q_LIB_LOCATION', pykx_libs_dir))
pykx_threading = _is_enabled('PYKX_THREADING')

_executable = 'q'
if platform.system() == 'Windows' and pykx_threading:
    _executable += '.exe'
    if pykx_threading:
        pykx_threading = False
        warn('PYKX_THREADING is only supported on Linux / MacOS, it has been disabled.')

q_executable = _get_config_value('PYKX_Q_EXECUTABLE', shutil.which(_executable))
no_pykx_signal = _is_enabled('PYKX_NO_SIGNAL')

ignore_qhome = _is_enabled('PYKX_IGNORE_QHOME', '--ignore-qhome')
keep_local_times = _is_enabled('PYKX_KEEP_LOCAL_TIMES')
max_error_length = int(_get_config_value('PYKX_MAX_ERROR_LENGTH', 256))

k_allocator = not _is_enabled('PYKX_NO_ALLOCATOR', '--pykxnoalloc')
k_gc = _is_enabled('PYKX_GC', '--pykxgc')
release_gil = _is_enabled('PYKX_RELEASE_GIL', '--release-gil')
use_q_lock = _get_config_value('PYKX_Q_LOCK', False)
skip_under_q = _is_enabled('PYKX_SKIP_UNDERQ', '--skip-under-q')
qce = _is_enabled('PYKX_QCE', '--qce')
beta_features = _is_enabled('PYKX_BETA_FEATURES', '--beta')
load_pyarrow_unsafe = _is_enabled('PYKX_LOAD_PYARROW_UNSAFE', '--load-pyarrow-unsafe')
pykx_qdebug = _is_enabled('PYKX_QDEBUG', '--q-debug')
pykx_debug_insights = _is_enabled('PYKX_DEBUG_INSIGHTS_LIBRARIES')

pandas_gt1 = int(pd.__version__.split('.')[0]) > 1
pandas_gt2 = int(pd.__version__.split('.')[0]) > 2
numpy_gt1 = int(np.__version__.split('.')[0]) > 1
jupyterq = _is_enabled('PYKX_JUPYTERQ')


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


def _set_unlicensed_reason(unlicensed_reason_):
    global unlicensed_reason
    unlicensed_reason = unlicensed_reason_


def _get_unlicensed_reason():
    return unlicensed_reason


def _set_qdebug(qdebug_):
    global pykx_qdebug
    pykx_qdebug = qdebug_


def _get_qdebug():
    return pykx_qdebug


def _set_keep_local_times(keep_local_times_):
    global keep_local_times
    keep_local_times = keep_local_times_


def _get_qexecutable():
    return _get_config_value('PYKX_Q_EXECUTABLE', shutil.which(_executable))


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
    'pykx_config_location',
    'pykx_config_content',
    'pykx_profile_content',

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
    'qce',
    'load_pyarrow_unsafe',

    'find_core_lib',

    'pandas_gt1',
    'pandas_gt2',
]


def __dir__():
    return sorted(__all__)

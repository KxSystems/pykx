from libc.stdint cimport *

from pykx cimport core
from ctypes import c_void_p, CDLL

from .core import licensed, keval
from .util import normalize_to_bytes, normalize_to_str

from .exceptions import PyKXException
from ._wrappers cimport factory
from .wrappers import K, List


q = None


def _init(_q):
    global q
    q = _q


def init_handle(host, port, credentials, unix, tls, timeout, large_messages):
    if unix and tls:
        raise PyKXException('Using both tls and unix is not possible with a QConnection.')
    if licensed:
        parts = (
            'unix://' if unix else '',
            'tcps://' if tls else '',
            f'{normalize_to_str(host, "Host")}:' if not unix else '',
            str(port),
            ':',
            credentials
        )
        sym = q.hsym(''.join(parts))
        hsym = (sym, int(timeout * 1000))
        return int(q.hopen(hsym))
    else:
        host = b'0.0.0.0' if unix else normalize_to_bytes(host, 'Host')
        return core.khpunc(
            host,
            port,
            credentials.encode(),
            int(timeout * 1000),
            int(large_messages) | (2 * int(tls))
        )


def delete_ptr(x):
    core.r0(<core.K><uintptr_t>x)


def _close_handle(handle):
    try:
        if licensed:
            q.hclose(handle)
        elif isinstance(handle, int):
            core.kclose(handle)
    except Exception:
        pass


cdef inline core.K r1k(x):
    return core.r1(<core.K><uintptr_t>x._addr)


def _unlicensed_call(handle: int, query: bytes, parameters: List[K], wait: bool) -> K:
    val = <uintptr_t>keval(query, *parameters, handle=handle)
    if wait:
        return factory(<uintptr_t>core.ee(<core.K>val), False)
    # TODO: if not wait, flush the handle
    if <uintptr_t>val == 0:
        raise PyKXException('Async query network error')
    return K(None)
    

cpdef ssl_info():
    """View information relating to the TLS settings used by PyKX from your process

    Returns:
        A dictionary outlining the TLS settings used by PyKX

    Example:

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
    """
    if licensed:
        return q('-26!0')
    cdef uintptr_t info = <uintptr_t>core.sslInfo(<core.K><uintptr_t>0)
    cdef uintptr_t err = <uintptr_t>core.ee(<core.K>info)
    return factory(err, False)

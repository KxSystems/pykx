from datetime import date
import os
import platform

# Do not import pykx here - use the `kx` fixture instead!
import pytest


if platform.system() != 'Linux' or platform.machine() == 'aarch64':
    pytest.skip(
        'PyKX Cloud Edition functionality only supports'
        ' x86_64 Linux',
        allow_module_level=True
    )


skipif_aws_bucket_not_set = pytest.mark.skipif(
    'AWS_BUCKET' not in os.environ,
    reason="Environment variable 'AWS_BUCKET' not set",
)


kurl_q_server_init = pytest.mark.parametrize(
    'q_init',
    [[b'system"l tests/kurl/echo_auth_server.q"']],
    ids=['kurl'],
)


def test_kurl_sync_error(q, kx):
    """Sub-selection of sync tests in base 'kurl' repo."""
    err_url = b"http://localhost:8081"
    with pytest.raises(kx.QError, match='(?i)request is of form'):
        q.kurl.sync([])

    with pytest.raises(kx.QError, match='(?i)parameters must be'):
        q.kurl.sync([err_url, 'GET', b""])

    with pytest.raises(kx.QError, match='(?i)tenant type not supported:'):
        q.kurl.sync([err_url, 'GET', {'': None, 'tenant': 5}])

    with pytest.raises(kx.QError, match='(?i)url must be'):
        q.kurl.sync([{}, 'GET', None])


def test_kurl_register_error(q, kx):
    """Sub-selection of register tests in base 'kurl' repo."""
    with pytest.raises(kx.QError, match='(?i)registrant must be of form'):
        q.kurl.register([])

    with pytest.raises(kx.QError, match='(?i)authinfo must be'):
        q.kurl.register(['aws', b'', b'', 5])


@kurl_q_server_init
def test_kurl_aws_auth(q, q_port):
    """Test authentication with fixed length 40 char key"""
    test_url = bytes(f'http://localhost:{q_port}', 'utf-8')

    # Register URL and validate registration
    b = q.kurl.i.listRegistered()
    q.kurl.register([
        'aws_cred',
        test_url,
        b'',
        q('`AccessKeyId`SecretAccessKey!("foo";40#"k")')])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', a, b)
    assert q('{x like first y`domain}', test_url, e)

    # Mock server to return 200 and own Authorization
    ret = q.kurl.sync([test_url+b'/aws?pass', 'GET', None])
    assert(str(ret[1]).startswith('AWS4-HMAC'))

    # Mock server to use region + service if overwritten
    ret = q.kurl.sync([
        test_url + b'/aws?pass',
        'GET',
        {'region': b'us-west-2', 'service': 's3'}])
    assert ret[0] == 200
    assert 'aws4_request' in str(ret[1])

    # Deregister URL and validate removal
    b = q.kurl.i.listRegistered()
    q.kurl.deregister([test_url, b''])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', b, a)
    assert q('{x like first y`domain}', test_url, e)


@kurl_q_server_init
def test_kurl_aws_minio_small(q, q_port):
    """Test AWS Min.io authentication with small passwords"""
    test_url = bytes(f'http://localhost:{q_port}', 'utf-8')

    # Register URL and validate registration
    b = q.kurl.i.listRegistered()
    q.kurl.register([
        'aws_cred',
        test_url,
        b'',
        q('`AccessKeyId`SecretAccessKey!("foo";1#"k")')])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', a, b)
    assert q('{x like first y`domain}', test_url, e)

    # Mock server to return 200 and own Authorization
    ret = q.kurl.sync([test_url+b'/aws?pass', 'GET', None])
    assert(str(ret[1]).startswith('AWS4-HMAC'))

    # Mock server to use region + service if overwritten
    ret = q.kurl.sync([
        test_url + b'/aws?pass',
        'GET',
        {'region': b'us-west-2', 'service': 's3'}])
    assert ret[0] == 200
    assert 'aws4_request' in str(ret[1])

    # Deregister URL and validate removal
    b = q.kurl.i.listRegistered()
    q.kurl.deregister([test_url, b''])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', b, a)
    assert q('{x like first y`domain}', test_url, e)


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
@kurl_q_server_init
def test_kurl_aws_minio_large(q, q_port):
    """Test AWS Min.io authentication with large passwords"""
    test_url = bytes(f'http://localhost:{q_port}', 'utf-8')

    # Register URL and validate registration
    b = q.kurl.i.listRegistered()
    q.kurl.register([
        'aws_cred',
        test_url,
        b'',
        q('`AccessKeyId`SecretAccessKey!("foo";8192#"k")')])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', a, b)
    assert q('{x like first y`domain}', test_url, e)

    # Mock server to return 200 and own Authorization
    ret = q.kurl.sync([test_url+b'/aws?pass', 'GET', None])
    assert(str(ret[1]).startswith('AWS4-HMAC'))

    # Mock server to use region + service if overwritten
    ret = q.kurl.sync([
        test_url + b'/aws?pass',
        'GET',
        {'region': b'us-west-2', 'service': 's3'}])
    assert ret[0] == 200
    assert 'aws4_request' in str(ret[1])

    # Deregister URL and validate removal
    b = q.kurl.i.listRegistered()
    q.kurl.deregister([test_url, b''])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', b, a)
    assert q('{x like first y`domain}', test_url, e)


@kurl_q_server_init
@pytest.mark.xfail(reason='Flaky AWS auth')
def test_kurl_aws_web_token_auth(q, q_port):
    """Test authentication using aws web token"""
    test_url = bytes(f'http://localhost:{q_port}', 'utf-8')

    # Register URL and validate registration
    b = q.kurl.i.listRegistered()
    sts_xml = q('"c"$read1`:tests/kurl/sample_sts_response.xml')
    q.kurl.register([
        'aws_sts',
        test_url,
        b'',
        {'initial_sts_response': sts_xml}])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', a, b)
    assert q('{x like first y`domain}', test_url, e)

    # Mock server to return 200 and own Authorization
    ret = q.kurl.sync([test_url+b'/aws?pass', 'GET', None])
    assert str(ret[1]).startswith('AWS4-HMAC')

    # Deregister URL and validate removal
    b = q.kurl.i.listRegistered()
    q.kurl.deregister([test_url, b''])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', b, a)
    assert q('{x like first y`domain}', test_url, e)


@kurl_q_server_init
def test_kurl_oauth(q, q_port):
    """Test OAuth2 authentication"""
    test_url = bytes(f'http://localhost:{q_port}', 'utf-8')

    # Register URL and validate registration
    b = q.kurl.i.listRegistered()
    q.kurl.register([
        'oauth2',
        test_url,
        b'',
        {'access_token': b'foobar'}])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', a, b)
    assert q('{x like first y`domain}', test_url, e)

    # Mock server to return 200 and own Authorization
    ret = q.kurl.sync([test_url+b'/oauth2?pass', 'GET', None])
    assert str(ret[1]).startswith('Bearer')

    # Deregister URL and validate removal
    b = q.kurl.i.listRegistered()
    q.kurl.deregister([test_url, b''])
    a = q.kurl.i.listRegistered()
    e = q('{x except y}', b, a)
    assert q('{x like first y`domain}', test_url, e)


@pytest.mark.xfail
@skipif_aws_bucket_not_set
@pytest.mark.xfail(reason="Unavailable AWS bucket", strict=False)
def test_objstor_aws_key(q):
    """List user viewable objects in an s3 bucket."""
    bucket_path = q('hsym `$getenv`AWS_BUCKET')
    res = q.key(bucket_path)
    assert 11 == q.type(res).py()


@pytest.mark.xfail
@skipif_aws_bucket_not_set
@pytest.mark.xfail(reason="Unavailable AWS bucket", strict=False)
def test_objstor_aws_get(q, kx):
    """Retrieve a q object from s3 using get."""
    file_path = q('hsym `$getenv[`AWS_BUCKET],"/qfile"')
    res = q.get(file_path)
    assert isinstance(res, kx.K)


@pytest.mark.xfail
@skipif_aws_bucket_not_set
@pytest.mark.xfail(reason="Unavailable AWS bucket", strict=False)
def test_objstor_aws_read0(q, kx):
    """Read a q object from s3 using read0."""
    file_path = q('hsym `$getenv[`AWS_BUCKET],"/test.txt"')
    res = q.read0(file_path)[0]
    assert isinstance(res, kx.CharVector)


@pytest.mark.xfail
@skipif_aws_bucket_not_set
@pytest.mark.xfail(reason="Unavailable AWS bucket", strict=False)
def test_objstor_aws_read1(q, kx):
    """Read a q object from s3 using read1."""
    file_path = q('hsym `$getenv[`AWS_BUCKET],"/test.txt"')
    res = q.read1(file_path)
    assert isinstance(res, kx.ByteVector)


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='ToDo investigate in KXI-63220'
)
def test_qlog_fd_stdout_endpoint(q):
    """Test STDOUT endpoint creation."""
    q('.com_kx_log.fd.i.write:{.fd.cache,:enlist(x;y)}')
    q('.fd.cache:()')

    # Setup endpoint
    q('id:.com_kx_log.lopen[`:fd://stdout]')
    q('data:.com_kx_log.i.endpoint[id;`data]')
    assert q('-1i~data`handle')
    q('.com_kx_log.lcloseAll[]')


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='ToDo investigate in KXI-63220'
)
def test_qlog_fd_stderr_setup(q):
    """Tests creation of stderr endpoint."""
    q('id:.com_kx_log.lopen[`:fd://stderr]')
    q('data:.com_kx_log.i.endpoint[id; `data]')
    assert q('-2i~data`handle')
    q('.com_kx_log.lcloseAll[]')


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='ToDo investigate in KXI-63220'
)
def test_qlog_fd_stdout_log_string(q):
    """Test default string logging."""
    q('.com_kx_log.fd.i.write:{.fd.cache,:enlist(x;y)}')
    q('.fd.cache:()')

    q('id:.com_kx_log.lopen[`:fd://stdout]')
    q('data:.com_kx_log.i.endpoint[id;`data]')

    # Cache updated
    q('.com_kx_log.msg[m:"Test message 1"]')
    assert q('m~last last .fd.cache')

    # Match default formatter output
    q('ex:.com_kx_log.fd.fmt[m:"Test message 1";()!()]')
    assert q('ex~m')
    q('.com_kx_log.lcloseAll[]')


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='ToDo investigate in KXI-63220'
)
def test_qlog_fd_stdout_custom_log(q):
    """Tests custom string logging."""
    q('.com_kx_log.fd.i.write:{.fd.cache,:enlist(x;y)}')
    q('.fd.cache:()')

    # Published message matches expected format
    q('md:`service`id!(`rdb;rand 0Ng)')
    q('msg:"Test message 2"')
    q('ex:.com_kx_log.fd.fmt[msg;md]')
    q('format:`url`metadata`formatter!(`:fd://stdout;md;`.com_kx_log.fd.fmt)')
    q('id:.com_kx_log.lopen[format]')
    q('.com_kx_log.msg msg')
    assert q('ex~last last .fd.cache')
    q('.com_kx_log.lcloseAll[]')


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='ToDo investigate in KXI-63220'
)
def test_qlog_fd_stdout_dict_log(q):
    """Tests dictionary message logging."""
    q('.com_kx_log.fd.i.write:{.fd.cache,:enlist(x;y)}')
    q('.fd.cache:()')

    q('md:`service`id!(`rdb; rand 0Ng)')
    q('msg:`level`message!(`INFO; "Test message 3")')
    q('ex:.com_kx_log.fd.fmt[msg; md]')
    q('id:.com_kx_log.lopen[`url`metadata!(`:fd://stdout; md)]')
    q('.com_kx_log.msg msg')
    assert q('ex~last last .fd.cache')
    q('.com_kx_log.lcloseAll[]')


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='ToDo investigate in KXI-63220'
)
def test_qlog_fd_file_publish(q):
    """Tests logging message to file."""
    q('.com_kx_log.fd.i.write:{.fd.cache,:enlist(x;y)}')
    q('.fd.cache:()')

    q('id:.com_kx_log.lopen[`:fd://test.log]')
    q('data:.com_kx_log.i.endpoint[id;`data]')

    q('.com_kx_log.msg[m:"Test message 4"]')
    q('hn:@[; `handle] exec first data from .com_kx_log.i.endpoint')
    q('ex:(hn;m)')
    assert q('ex~last .fd.cache')
    q('.com_kx_log.lcloseAll[]')


@pytest.mark.ipc
def test_sql(q):
    if not hasattr(q, 's'):
        pytest.skip('KXIC SQL missing')

    q('trades:([]'
      'sym:`ABC`XYZ`ABC`JKL`JKL;'
      'price:123.4 2.7 123.1 32.5 31.9;'
      'date:2022.03.30 2022.03.30 2022.03.31 2022.03.31 2022.03.31)')
    trades = q('trades')

    assert q.sql('select sym from trades').py() == {'sym': ['ABC', 'XYZ', 'ABC', 'JKL', 'JKL']}

    d = date(2022, 3, 31)
    assert q.sql('select * from trades where date = $1 and price < $2', d, 32.1).py() \
        == q.sql('select * from $1 where date = $2 and price < $3', trades, d, 32.1).py() \
        == {'sym': ['JKL'], 'price': [31.9], 'date': [d]}

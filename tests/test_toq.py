from datetime import date, datetime, time, timedelta
from functools import partial
import math
import os
from pathlib import Path
from sys import getrefcount
from uuid import UUID, uuid4

# Do not import Pandas, PyArrow, or PyKX here - use the pd/pa/kx fixtures instead!
import numpy as np
import pandas as pd
import pytest


def gen_q_datatypes_table(q, table_name: str, num_rows: int = 100) -> str:
    query = '{@[;0;string]x#/:prd[x]?/:(`6;`6;0Ng;.Q.a),("xpdmnuvtbhijef"$\\:0)}'
    t = q(query, q('enlist', num_rows))
    if q('{any(raze null x),raze(v,neg v:0W 2147483647 32767)=\\:raze"j"$x:5_x}', t):
        t = q(query, q('enlist', num_rows)) # The table is regenerated until it has no nulls/infs.
    q[table_name] = t
    return table_name


@pytest.fixture
def q_atom_types(kx):
    return [
        kx.GUIDAtom,
        kx.ShortAtom,
        kx.IntAtom,
        kx.LongAtom,
        kx.RealAtom,
        kx.FloatAtom,
        kx.CharAtom,
        kx.SymbolAtom,
        kx.TimestampAtom,
        kx.MonthAtom,
        kx.DateAtom,
        kx.TimespanAtom,
        kx.MinuteAtom,
        kx.SecondAtom,
        kx.TimeAtom
    ]


@pytest.mark.unlicensed(unlicensed_only=True)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_adapt_k_unlicensed_error(kx):
    a = kx.toq(1.5)
    with pytest.raises(kx.LicenseException):
        kx.toq(a, kx.LongAtom)


@pytest.mark.licensed
def test_adapt_k_numeric_types(kx, q_atom_types):
    numeric_atom_types = [
        kx.BooleanVector,
        kx.ByteVector,
        kx.ShortVector,
        kx.IntVector,
        kx.LongVector,
        kx.RealVector,
        kx.FloatVector,
    ]

    a = [1, 2, 3, 4, 5]
    for ty_1 in numeric_atom_types:
        for ty_2 in numeric_atom_types:
            b = ty_1(a)
            c = ty_2(a)
            if ty_1 != kx.BooleanVector:
                assert b.cast(ty_2).py() == c.py()
            assert isinstance(b.cast(ty_2), ty_2)
            assert isinstance(ty_2(a), ty_2)
            assert isinstance(kx.toq(a, ty_2), ty_2)


@pytest.mark.licensed
def test_adapt_k_q_error(kx, q):
    with pytest.raises(kx.QError):
        q('0Nm').cast(kx.LongAtom)


@pytest.mark.licensed
def test_adapt_k_table_dictionary_conversion(kx, q):
    a = q('([] a: 10?10; b: 10?10)')
    b = kx.toq({'foo': 1, 'bar': None, 'baz': b'apple'})

    assert isinstance(a.cast(kx.Dictionary), kx.Dictionary)
    assert isinstance(kx.Table(b), kx.Table)
    assert kx.Table(kx.Dictionary(a)).py() == a.py()

    with pytest.raises(TypeError):
        kx.LongVector(a)
    with pytest.raises(TypeError):
        kx.LongVector(b)
    with pytest.raises(kx.QError):
        q('`a`b!(1 2 3; 1 2 3 4)').cast(kx.Table)


@pytest.mark.licensed
def test_adapt_k_CharVector_conversions(kx, q_atom_types):
    test_str = kx.toq(b"1")
    assert kx.toq(test_str, kx.LongAtom).py() == 1
    assert isinstance(kx.toq(test_str, kx.LongAtom), kx.LongAtom)

    test_str = kx.toq(b"1.5")
    assert kx.FloatAtom(test_str).py() == 1.5
    assert isinstance(kx.toq(test_str, kx.FloatAtom), kx.FloatAtom)

    test_str = kx.toq(b"abc")
    assert "abc" == kx.SymbolAtom(test_str).py()
    assert isinstance(kx.SymbolAtom(test_str), kx.SymbolAtom)

    with pytest.raises(TypeError):
        kx.toq(test_str, np.uint8)


@pytest.mark.isolate
def test_q_memory_init():
    """Test Python -> q conversion functionality before making an IPC connection."""
    os.environ['QARGS'] = '--testflag'
    import pykx as kx
    assert kx.K('sym').py() == 'sym'
    v = ['list', 'of', 'symbols']
    assert kx.K(v).py() == v
    assert kx.K(1234567890).py() == 1234567890


@pytest.mark.unlicensed
def test_default_conversion(kx):
    with pytest.raises(TypeError):
        kx.K(pytest)


@pytest.mark.unlicensed
def test_from_none(kx):
    assert kx.K(None).t == 101


@pytest.mark.licensed
def test_from_none_licensed(kx, q_atom_types):
    for ty in q_atom_types:
        qnull = ty(None)
        assert isinstance(qnull, ty)
        assert qnull.is_null


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_pandas_nat(kx, pd):
    assert isinstance(kx.K(pd.NaT), kx.TimestampAtom)
    assert pd.isna(kx.K(pd.NaT).pd())

    exclude = {kx.TemporalAtom, kx.TemporalSpanAtom, kx.TemporalFixedAtom, kx.DatetimeAtom}
    for x in kx.util.subclasses(kx.TemporalAtom) - exclude:
        nat_as_x = kx.toq(pd.NaT, ktype=x)
        assert isinstance(nat_as_x, x)
        assert isinstance(nat_as_x.pd(), type(pd.NaT))

    with pytest.raises(NotImplementedError):
        kx.toq(pd.NaT, kx.DatetimeAtom)

    with pytest.raises(TypeError):
        kx.toq(pd.NaT, kx.CharAtom)


def test_from_pykx_k(q, kx):
    g = q('first -1?0Ng')
    assert kx.K(g) == g


@pytest.mark.unlicensed
def test_from_int(kx):
    assert kx.K(1234567890).t == kx.toq(1234567890).t == -7
    assert kx.LongAtom(1234567890).t == kx.toq(1234567890, kx.LongAtom).t == -7
    for x in (-2 ** 65, -9223372036854775809, 9223372036854775808, 2 ** 65):
        with pytest.raises(OverflowError):
            kx.BooleanAtom(x)

    assert kx.IntAtom(12345).t == kx.toq(12345, kx.IntAtom).t == -6
    for x in (-2 ** 65, -2147483649, 2147483648, 2 ** 65):
        with pytest.raises(OverflowError):
            kx.BooleanAtom(x)

    assert kx.ShortAtom(12345).t == kx.toq(12345, kx.ShortAtom).t == -5
    for x in (-2 ** 65, -32769, 32768, 2 ** 65):
        with pytest.raises(OverflowError):
            kx.ShortAtom(x)

    assert kx.ByteAtom(255).t == kx.toq(255, kx.ByteAtom).t == -4
    assert kx.ByteAtom(0).t == kx.toq(0, kx.ByteAtom).t == -4
    for x in (-2 ** 65, -1, 256, 2 ** 65):
        with pytest.raises(OverflowError):
            kx.ByteAtom(x)

    assert kx.BooleanAtom(1).t == kx.toq(1, kx.BooleanAtom).t == -1
    assert kx.BooleanAtom(0).t == kx.toq(0, kx.BooleanAtom).t == -1
    for x in (-2 ** 65, -1, 2, 2 ** 65):
        with pytest.raises(OverflowError):
            kx.BooleanAtom(x)

    assert kx.K(True).t == -1
    assert kx.K(False).py() is False
    assert kx.K(True).py() is True
    with pytest.raises(TypeError):
        kx.Table(-2468)


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_float32(kx):
    assert kx.toq(np.half(1.23456789)).np() == np.float32(1.234375)
    assert kx.toq(np.float16(1.23456789)).np() == np.float32(1.234375)
    assert kx.toq(np.single(1.23456789)).np() == np.float32(1.23456789)
    assert kx.toq(np.float32(1.23456789)).np() == np.float32(1.23456789)


@pytest.mark.unlicensed
def test_from_float(kx):
    assert math.isnan(kx.toq(float('NaN')).py())
    assert kx.toq(float('inf')).py() == float('inf')
    assert kx.toq(float('inf'), -8).py() == float('inf')
    assert kx.toq(-float('inf')).py() == -float('inf')
    assert kx.toq(-float('inf'), -8).py() == -float('inf')
    assert kx.toq(1234.003e42, -9).py() == 1234.003e42
    assert kx.toq(1234.003e42, -8).py() == float('inf')
    with pytest.raises(TypeError):
        kx.Function(-2.468)


@pytest.mark.unlicensed
def test_from_str(kx):
    assert kx.toq('abcdefghijklmnopqrstuvwxyz').py() == 'abcdefghijklmnopqrstuvwxyz'
    assert kx.toq('abcdefghijklmnopqrstuvwxyz', 10).py() == b'abcdefghijklmnopqrstuvwxyz'
    assert kx.toq('').py() == ''
    assert kx.CharVector('').py() == b''
    assert kx.toq('🙃').py() == '🙃'
    assert kx.toq('🙃', 10).py() == b'\xf0\x9f\x99\x83'
    assert isinstance(kx.CharAtom('x'), kx.CharAtom)
    assert kx.List(['aaa']).py() == ['aaa']
    with pytest.raises(TypeError):
        kx.List('aaa')
    for x in ('', '12', '🙃'):
        with pytest.raises(ValueError):
            kx.CharAtom(x)
    with pytest.raises(TypeError):
        kx.GUIDAtom('x')

    # Symbolic function from str:
    for f in (kx.toq('.vvv.f', kx.SymbolicFunction), kx.SymbolicFunction('.vvv.f')):
        assert isinstance(f, kx.SymbolicFunction)
        assert f == f.py()


@pytest.mark.unlicensed
def test_from_bytes(kx):
    assert kx.toq(b'abcdefghijklmnopqrstuvwxyz', -11).py() == 'abcdefghijklmnopqrstuvwxyz'
    assert kx.toq(b'abcdefghijklmnopqrstuvwxyz').py() == b'abcdefghijklmnopqrstuvwxyz'
    assert kx.toq(b'').py() == b''
    assert isinstance(kx.CharAtom(b'x'), kx.CharAtom)
    assert kx.CharAtom(b'x').py() == b'x'
    for x in (b'', b'12'):
        with pytest.raises(ValueError):
            kx.CharAtom(x)
    with pytest.raises(TypeError):
        kx.GUIDAtom(b'x')


@pytest.mark.unlicensed
def test_from_bytes_np(kx):
    b = kx.toq(np.bytes_(''))
    assert isinstance(b, kx.CharVector)
    assert b.py() == b''
    b = kx.toq(np.bytes_('a'))
    assert isinstance(b, kx.CharAtom)
    assert b.py() == b'a'
    b = kx.toq(np.bytes_('aa'))
    assert isinstance(b, kx.CharVector)
    assert b.py() == b'aa'
    assert kx.toq(np.bytes_('abcdefghijklmnopqrstuvwxyz'), -11).py() == 'abcdefghijklmnopqrstuvwxyz'
    for x in (np.bytes_(''), np.bytes_('12')):
        with pytest.raises(ValueError):
            kx.CharAtom(x)
    with pytest.raises(TypeError):
        kx.GUIDAtom(np.bytes_('x'))


@pytest.mark.unlicensed
def test_from_datetime_date(kx):
    d = date(2020, 9, 8)

    kd = kx.TimestampAtom(d)
    assert isinstance(kd, kx.TimestampAtom)
    assert kd.py().date() == d

    kd = kx.MonthAtom(d)
    assert isinstance(kd, kx.MonthAtom)
    assert kd.py() == date(2020, 9, 1)

    kd = kx.DateAtom(d)
    assert isinstance(kd, kx.DateAtom)
    assert kd.py() == d

    kd = kx.K(d)
    assert isinstance(kd, kx.DateAtom)
    assert kd.py() == d

    assert kx.K(kx.K(d).py()).py() == d

    with pytest.raises(NotImplementedError):
        kx.DatetimeAtom(d)


@pytest.mark.unlicensed
def test_from_datetime_time(kx):
    t = time(15, 55, 23)
    t_np = np.datetime64('2005-02-25T03:30')

    kd = kx.K(t)
    assert isinstance(kd, kx.TimespanAtom)
    assert kd.py() == timedelta(seconds=57323)

    kd_np = kx.toq.from_datetime_time(t_np, cast=True)
    assert isinstance(kd_np, kx.TimespanAtom)
    assert kd_np.py() == timedelta(seconds=12600)

    with pytest.raises(TypeError):
        kx.TimeAtom(t_np)


@pytest.mark.unlicensed
def test_from_datetime_datetime(kx):
    d = datetime(2020, 9, 8, 7, 6, 5, 4)

    kd = kx.K(d)
    assert isinstance(kd, kx.TimestampAtom)
    assert kd.py() == d

    kd = kx.TimestampAtom(d)
    assert isinstance(kd, kx.TimestampAtom)
    assert kd.py() == d

    kd = kx.MonthAtom(d)
    assert isinstance(kd, kx.MonthAtom)
    assert kd.py() == datetime(2020, 9, 1).date()

    kd = kx.DateAtom(d)
    assert isinstance(kd, kx.DateAtom)
    assert kd.py() == datetime(2020, 9, 8).date()

    with pytest.raises(NotImplementedError):
        kx.DatetimeAtom(d)


@pytest.mark.unlicensed
def test_from_timedelta(kx):
    d = timedelta(hours=16, minutes=45, seconds=12, milliseconds=222, microseconds=970)

    kd = kx.K(d)
    assert isinstance(kd, kx.TimespanAtom)
    assert kd.py() == d

    kd = kx.MinuteAtom(d)
    assert isinstance(kd, kx.MinuteAtom)
    assert kd.py() == timedelta(hours=16, minutes=45)

    kd = kx.SecondAtom(d)
    assert isinstance(kd, kx.SecondAtom)
    assert kd.py() == timedelta(hours=16, minutes=45, seconds=12)

    kd = kx.TimeAtom(d)
    assert isinstance(kd, kx.TimeAtom)
    assert kd.py() == timedelta(hours=16, minutes=45, seconds=12, milliseconds=222)


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_datetime64(kx):
    d = np.datetime64('2020-09-08T07:06:05.000004')

    kd = kx.K(d)
    assert isinstance(kd, kx.TimestampAtom)
    assert kd.np() == d

    kd = kx.MonthAtom(d)
    assert isinstance(kd, kx.MonthAtom)
    assert kd.np() == np.datetime64('2020-09-01')

    kd = kx.DateAtom(d)
    assert isinstance(kd, kx.DateAtom)
    assert kd.np() == np.datetime64('2020-09-08')

    s = b'2023-03-21T10:46:14.198981339'
    ts = pd.to_datetime(s.decode('utf-8'))
    assert isinstance(kx.K(ts), kx.TimestampAtom)

    with pytest.raises(NotImplementedError):
        kx.DatetimeAtom(d)


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_datetime64_smsusns(kx):
    d = np.array(['2020-09-08T07:06:05.000004', '2020-09-08T07:06:05.000004'],
                 dtype='datetime64[ns]')
    dn = np.array(['', ''], dtype='datetime64[ns]')
    dnm = np.array(['', '2020-09-08T07:06:05.000004'], dtype='datetime64[ns]')
    df = pd.DataFrame(data={'d': d, 'dn': dn, 'dnm': dnm})

    kd = kx.K(d)
    kd_hn = kx.K(d, handle_nulls=True)
    kdn = kx.K(dn)
    kdn_hn = kx.K(dn, handle_nulls=True)
    assert all([isinstance(x, kx.TimestampVector) for x in [kd, kd_hn, kdn, kdn_hn]])
    assert (kd.np() == d.astype(np.dtype('datetime64[ns]'))).all()
    if kx.licensed:
        assert (kx.toq(df) == kx.toq(kx.toq(df).pd())).all().all()
        assert (kx.toq(df, handle_nulls=True)
                == kx.toq(kx.toq(df, handle_nulls=True).pd(), handle_nulls=True)).all().all()
        if kx.config.pandas_2:
            assert (kx.toq(df) == kx.toq(kx.toq(df).pd(as_arrow=True))).all().all()
            assert (kx.toq(df, handle_nulls=True)
                    == kx.toq(kx.toq(df, handle_nulls=True).pd(as_arrow=True),
                              handle_nulls=True)).all().all()

    d = np.array(['2020-09-08T07:06:05.000004', '2020-09-08T07:06:05.000004'],
                 dtype='datetime64[us]')
    dn = np.array(['', ''], dtype='datetime64[us]')
    dnm = np.array(['', '2020-09-08T07:06:05.000004'], dtype='datetime64[us]')
    df = pd.DataFrame(data={'d': d, 'dn': dn, 'dnm': dnm})

    kd = kx.K(d)
    kd_hn = kx.K(d, handle_nulls=True)
    kdn = kx.K(dn)
    kdn_hn = kx.K(dn, handle_nulls=True)
    assert all([isinstance(x, kx.TimestampVector) for x in [kd, kd_hn, kdn, kdn_hn]])
    assert (kd.np() == d.astype(np.dtype('datetime64[ns]'))).all()
    if kx.licensed:
        assert (kx.toq(df) == kx.toq(kx.toq(df).pd())).all().all()
        assert (kx.toq(df, handle_nulls=True)
                == kx.toq(kx.toq(df, handle_nulls=True).pd(), handle_nulls=True)).all().all()
        if kx.config.pandas_2:
            assert (kx.toq(df) == kx.toq(kx.toq(df).pd(as_arrow=True))).all().all()
            assert (kx.toq(df, handle_nulls=True)
                    == kx.toq(kx.toq(df, handle_nulls=True).pd(as_arrow=True),
                              handle_nulls=True)).all().all()

    d = np.array(['2020-09-08T07:06:05.000004', '2020-09-08T07:06:05.000004'],
                 dtype='datetime64[ms]')
    dn = np.array(['', ''], dtype='datetime64[ms]')
    dnm = np.array(['', '2020-09-08T07:06:05.000004'], dtype='datetime64[ms]')
    df = pd.DataFrame(data={'d': d, 'dn': dn, 'dnm': dnm})

    kd = kx.K(d)
    kd_hn = kx.K(d, handle_nulls=True)
    kdn = kx.K(dn)
    kdn_hn = kx.K(dn, handle_nulls=True)
    assert all([isinstance(x, kx.TimestampVector) for x in [kd, kd_hn, kdn, kdn_hn]])
    assert (kd.np() == d.astype(np.dtype('datetime64[ns]'))).all()
    if kx.licensed:
        assert (kx.toq(df) == kx.toq(kx.toq(df).pd())).all().all()
        assert (kx.toq(df, handle_nulls=True)
                == kx.toq(kx.toq(df, handle_nulls=True).pd(), handle_nulls=True)).all().all()
        if kx.config.pandas_2:
            assert (kx.toq(df) == kx.toq(kx.toq(df).pd(as_arrow=True))).all().all()
            assert (kx.toq(df, handle_nulls=True)
                    == kx.toq(kx.toq(df, handle_nulls=True).pd(as_arrow=True),
                              handle_nulls=True)).all().all()

    d = np.array(['2020-09-08T07:06:05.000004', '2020-09-08T07:06:05.000004'],
                 dtype='datetime64[s]')
    dn = np.array(['', ''], dtype='datetime64[s]')
    dnm = np.array(['', '2020-09-08T07:06:05.000004'], dtype='datetime64[s]')
    df = pd.DataFrame(data={'d': d, 'dn': dn, 'dnm': dnm})

    kd = kx.K(d)
    kd_hn = kx.K(d, handle_nulls=True)
    kdn = kx.K(dn)
    kdn_hn = kx.K(dn, handle_nulls=True)
    assert all([isinstance(x, kx.TimestampVector) for x in [kd, kd_hn, kdn, kdn_hn]])
    assert (kd.np() == d.astype(np.dtype('datetime64[ns]'))).all()
    if kx.licensed:
        assert (kx.toq(df) == kx.toq(kx.toq(df).pd())).all().all()
        assert (kx.toq(df, handle_nulls=True)
                == kx.toq(kx.toq(df, handle_nulls=True).pd(), handle_nulls=True)).all().all()
        if kx.config.pandas_2:
            assert (kx.toq(df) == kx.toq(kx.toq(df).pd(as_arrow=True))).all().all()
            assert (kx.toq(df, handle_nulls=True)
                    == kx.toq(kx.toq(df, handle_nulls=True).pd(as_arrow=True),
                              handle_nulls=True)).all().all()


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_timedelta64(kx):
    d = np.timedelta64(60312222971312, 'ns')

    kd = kx.K(d)
    assert isinstance(kd, kx.TimespanAtom)
    assert kd.np() == d

    kd = kx.MinuteAtom(d)
    assert isinstance(kd, kx.MinuteAtom)
    assert kd.np() == np.timedelta64(1005, 'm')

    kd = kx.SecondAtom(d)
    assert isinstance(kd, kx.SecondAtom)
    assert kd.np() == np.timedelta64(60312, 's')

    kd = kx.TimeAtom(d)
    assert isinstance(kd, kx.TimeAtom)
    assert kd.np() == np.timedelta64(60312222, 'ms')

    kd = kx.TimespanAtom(d)
    assert isinstance(kd, kx.TimespanAtom)
    assert kd.np() == np.timedelta64(60312222971000, 'ns')


@pytest.mark.unlicensed
def test_from_UUID(kx):
    u = uuid4()
    assert kx.K(u).py() == u
    if kx.licensed:
        assert str(kx.K(u)) == str(u)
    u = uuid4()
    assert kx.toq(u, kx.GUIDAtom).py() == kx.GUIDAtom(u).py() == u
    if kx.licensed:
        assert str(kx.K(u)) == str(u)

    u = UUID('db712ca2-81b1-0080-95dd-7bdb502da77d')
    assert kx.K(u).py() == u
    if kx.licensed:
        assert str(kx.K(u)) == str(u)
    assert kx.toq(u, kx.GUIDAtom).py() == kx.GUIDAtom(u).py() == u
    if kx.licensed:
        assert str(kx.K(u)) == str(u)

    u = UUID('7ff0bbb9-ee32-42fe-9631-8c5a09a155a2')
    assert kx.K(u).py() == u
    if kx.licensed:
        assert str(kx.K(u)) == str(u)
    assert kx.toq(u, kx.GUIDAtom).py() == kx.GUIDAtom(u).py() == u
    if kx.licensed:
        assert str(kx.K(u)) == str(u)


@pytest.mark.unlicensed
def test_from_UUID_list(kx):
    u = [UUID('db712ca2-81b1-0080-95dd-7bdb502da77d')]
    assert kx.K(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])
    assert kx.toq(u, kx.GUIDVector).py() == kx.GUIDVector(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])

    u = [UUID('7ff0bbb9-ee32-42fe-9631-8c5a09a155a2')]
    assert kx.K(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])
    assert kx.toq(u, kx.GUIDVector).py() == kx.GUIDVector(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])


@pytest.mark.unlicensed
def test_from_UUID_np_array(kx):
    u = np.array([UUID('db712ca2-81b1-0080-95dd-7bdb502da77d')], dtype=object)
    assert kx.K(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])
    assert kx.toq(u, kx.GUIDVector).py() == kx.GUIDVector(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])

    u = np.array([UUID('7ff0bbb9-ee32-42fe-9631-8c5a09a155a2')], dtype=object)
    assert kx.K(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])
    assert kx.toq(u, kx.GUIDVector).py() == kx.GUIDVector(u).py() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])


@pytest.mark.unlicensed
def test_from_UUID_pandas(kx):
    values = [
        (1.3942713164545354e+64 - 7.26060294431316e-266j),
        (3.638224669629338e+199 - 7.695044086357459e-212j)
    ]

    def complex_to_guid(c):
        real_part = c.real
        imag_part = c.imag
        real_bytes = real_part.hex().encode('utf-8')[:8]
        imag_bytes = imag_part.hex().encode('utf-8')[:8]
        guid_bytes = real_bytes.ljust(8, b'\x00') + imag_bytes.ljust(8, b'\x00')
        return UUID(bytes=guid_bytes)

    guid_values = [complex_to_guid(c) for c in values]
    guid_vector = kx.GUIDVector(guid_values)
    u = {'g': guid_vector}
    res = kx.toq(u['g'].pd(raw=True))
    assert isinstance(res, kx.K)


@pytest.mark.unlicensed
def test_to_UUID_np_array(kx):
    u = np.array([UUID('db712ca2-81b1-0080-95dd-7bdb502da77d')], dtype=object)
    assert kx.K(u).np() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])
    assert kx.toq(u, kx.GUIDVector).np() == kx.GUIDVector(u).np() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])

    u = np.array([UUID('7ff0bbb9-ee32-42fe-9631-8c5a09a155a2')], dtype=object)
    assert kx.K(u).np() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])
    assert kx.toq(u, kx.GUIDVector).np() == kx.GUIDVector(u).np() == u
    if kx.licensed:
        assert str(kx.K(u[0])) == str(u[0])


@pytest.mark.unlicensed
def test_from_tuple(kx):
    assert kx.K(()).py() == []
    assert kx.K((1, 2)).py() == [1, 2]
    x = kx.K((42, 'thiswillbecomeasymbolatom'))
    assert isinstance(x, kx.List)
    assert isinstance(list(x)[0], kx.LongAtom)
    assert isinstance(list(x)[1], kx.SymbolAtom)
    assert x.py() == [42, 'thiswillbecomeasymbolatom']
    with pytest.raises(TypeError):
        kx.toq((object,), kx.LongAtom)


@pytest.mark.unlicensed
@pytest.mark.nep49
@pytest.mark.xfail(reason='Flaky NEP-49 testing with datetime', strict=False)
def test_from_list(kx):
    assert kx.K([]).py() == []
    assert kx.K([1, 2]).py() == [1, 2]
    x = kx.K([42, 'thiswillbecomeasymbolatom'])
    assert isinstance(x, kx.List)
    assert isinstance(list(x)[0], kx.LongAtom)
    assert isinstance(list(x)[1], kx.SymbolAtom)
    assert x.py() == [42, 'thiswillbecomeasymbolatom']
    assert type(kx.LongVector([0, 1, 2, 3, 4, 5, 6, 7, 0o10])) is kx.LongVector
    assert isinstance(kx.IntVector([0, 1, 2, 3, 4, 5, 6, 7, 0o10]), kx.IntVector)
    assert isinstance(kx.ShortVector([0, 1, 2, 3, 4, 5, 6, 7, 0o10]), kx.ShortVector)
    assert isinstance(kx.ByteVector([0, 1, 2, 3, 4, 5, 6, 7, 0o10]), kx.ByteVector)
    assert isinstance(kx.BooleanVector([0, 1, 2, 3, 4, 5, 6, 7, 0o10]), kx.BooleanVector)
    assert isinstance(kx.SymbolVector(['a', 'cat']), kx.SymbolVector)
    assert isinstance(kx.BooleanVector(True), kx.BooleanVector)
    assert isinstance(kx.GUIDVector(uuid4()), kx.GUIDVector)
    assert isinstance(kx.ByteVector(0xff), kx.ByteVector)
    assert isinstance(kx.ShortVector(2), kx.ShortVector)
    assert isinstance(kx.IntVector(2), kx.IntVector)
    assert isinstance(kx.LongVector(2), kx.LongVector)
    assert isinstance(kx.RealVector(3.14), kx.RealVector)
    assert isinstance(kx.FloatVector(3.14), kx.FloatVector)
    assert isinstance(kx.CharVector('a'), kx.CharVector)
    assert isinstance(kx.List([b'aaa']), kx.List)
    assert isinstance(kx.SymbolVector(['a']), kx.SymbolVector)
    assert isinstance(kx.TimestampVector(np.datetime64(0, 'ns')), kx.TimestampVector)
    assert isinstance(kx.MonthVector(np.datetime64(0, 'M')), kx.MonthVector)
    assert isinstance(kx.DateVector(np.datetime64(0, 'D')), kx.DateVector)
    assert isinstance(kx.TimespanVector(np.timedelta64(0, 'us')), kx.TimespanVector)
    assert isinstance(kx.TimespanVector(np.timedelta64(0, 'ns')), kx.TimespanVector)
    assert isinstance(kx.MinuteVector(np.timedelta64(0, 'W')), kx.MinuteVector)
    assert isinstance(kx.SecondVector(np.timedelta64(0, 's')), kx.SecondVector)
    assert isinstance(kx.TimeVector(np.timedelta64(0, 'ms')), kx.TimeVector)
    with pytest.raises(TypeError):
        kx.toq([object], kx.LongAtom)
    with pytest.raises(TypeError):
        kx.LongAtom([object])
    with pytest.raises(TypeError):
        kx.ByteVector(b'x')
    with pytest.raises(TypeError):
        kx.List('aaa')


@pytest.mark.unlicensed
def test_from_dict(kx):
    assert kx.K({}).py() == {}
    assert type(kx.K({})._keys) == kx.List

    d = {'a': [b'chars'], 'another key': 'another value'}
    kd = kx.K(d)
    if kx.licensed:
        assert isinstance(kd['a'], kx.List)
    assert kd.py() == d
    with pytest.raises(TypeError):
        kx.toq({}, kx.List)

    guid = uuid4()
    dict_with_non_string_keys = {'abc': 123, guid: 321}
    k_dict_with_non_string_keys = kx.K(dict_with_non_string_keys)
    assert k_dict_with_non_string_keys.py() == dict_with_non_string_keys
    assert isinstance(list(k_dict_with_non_string_keys.keys())[0], kx.SymbolAtom)
    assert isinstance(list(k_dict_with_non_string_keys.keys())[1], kx.GUIDAtom)


@pytest.mark.unlicensed
def test_from_slice(kx):
    assert kx.K(slice(None, 4)).py() == [0, 1, 2, 3]
    assert kx.K(slice(0, 4)).py() == [0, 1, 2, 3]
    assert kx.K(slice(0, 12, 3)).py() == [0, 3, 6, 9]
    assert kx.K(slice(-2, 3, 2)).py() == [-2, 0, 2]

    assert isinstance(kx.LongVector(slice(0, 1)), kx.LongVector)
    assert isinstance(kx.IntVector(slice(0, 1)), kx.IntVector)
    assert isinstance(kx.ShortVector(slice(0, 1)), kx.ShortVector)
    assert isinstance(kx.ByteVector(slice(0, 1)), kx.ByteVector)
    assert isinstance(kx.BooleanVector(slice(0, 1)), kx.BooleanVector)

    for x in (slice(-9223372036854775808, -9223372036854775807),
              slice(9223372036854775807, 9223372036854775808)):
        assert isinstance(kx.LongVector(x), kx.LongVector)
    for x in (slice(-9223372036854775809, -9223372036854775807),
              slice(9223372036854775807, 9223372036854775809)):
        with pytest.raises(OverflowError):
            kx.LongVector(x)

    for x in (slice(-2147483648, -2147483647), slice(2147483647, 2147483648)):
        assert isinstance(kx.IntVector(x), kx.IntVector)
    for x in (slice(-2147483649, -2147483647), slice(2147483647, 2147483649)):
        with pytest.raises(OverflowError):
            kx.IntVector(x)

    for x in (slice(-32768, -32767), slice(32767, 32768)):
        assert isinstance(kx.ShortVector(x), kx.ShortVector)
    for x in (slice(-32769, -32767), slice(32767, 32769)):
        with pytest.raises(OverflowError):
            kx.ShortVector(x)

    assert isinstance(kx.ByteVector(slice(0, 256)), kx.ByteVector)
    for x in (slice(-1, 0), slice(0, 257)):
        with pytest.raises(OverflowError):
            kx.ByteVector(x)

    assert isinstance(kx.BooleanVector(slice(0, 2)), kx.BooleanVector)
    for x in (slice(-1, 0), slice(0, 3)):
        with pytest.raises(OverflowError):
            kx.BooleanVector(x)

    with pytest.raises(ValueError):
        kx.K(slice(None))
    with pytest.raises(ValueError):
        kx.K(slice(0, None, 2))


@pytest.mark.unlicensed
def test_from_range(kx):
    assert kx.K(range(0)).py() == []
    assert kx.K(range(3, 12, 2)).py() == [3, 5, 7, 9, 11]
    assert kx.K(range(-3, -12, -2)).py() == [-3, -5, -7, -9, -11]
    assert isinstance(kx.LongVector(range(8)), kx.LongVector)
    assert isinstance(kx.toq(range(8), kx.LongVector), kx.LongVector)
    assert isinstance(kx.IntVector(range(8)), kx.IntVector)
    assert isinstance(kx.ShortVector(range(8)), kx.ShortVector)
    assert isinstance(kx.ByteVector(range(8)), kx.ByteVector)

    assert isinstance(kx.LongVector(range(0, 1)), kx.LongVector)
    assert isinstance(kx.IntVector(range(0, 1)), kx.IntVector)
    assert isinstance(kx.ShortVector(range(0, 1)), kx.ShortVector)
    assert isinstance(kx.ByteVector(range(0, 1)), kx.ByteVector)
    assert isinstance(kx.BooleanVector(range(0, 1)), kx.BooleanVector)

    for x in (range(-9223372036854775808, -9223372036854775807),
              range(9223372036854775807, 9223372036854775808)):
        assert isinstance(kx.LongVector(x), kx.LongVector)
    for x in (range(-9223372036854775809, -9223372036854775807),
              range(9223372036854775807, 9223372036854775809)):
        with pytest.raises(OverflowError):
            kx.LongVector(x)

    for x in (range(-2147483648, -2147483647), range(2147483647, 2147483648)):
        assert isinstance(kx.IntVector(x), kx.IntVector)
    for x in (range(-2147483649, -2147483647), range(2147483647, 2147483649)):
        with pytest.raises(OverflowError):
            kx.IntVector(x)

    for x in (range(-32768, -32767), range(32767, 32768)):
        assert isinstance(kx.ShortVector(x), kx.ShortVector)
    for x in (range(-32769, -32767), range(32767, 32769)):
        with pytest.raises(OverflowError):
            kx.ShortVector(x)

    assert isinstance(kx.ByteVector(range(0, 256)), kx.ByteVector)
    for x in (range(-1, 0), range(0, 257)):
        with pytest.raises(OverflowError):
            kx.ByteVector(x)

    assert isinstance(kx.BooleanVector(range(0, 2)), kx.BooleanVector)
    for x in (range(-1, 0), range(0, 3)):
        with pytest.raises(OverflowError):
            kx.BooleanVector(x)


def test_from_path(q, kx):
    assert kx.K(Path()) == ':.'
    assert kx.K(Path('this/is/a/path')) == q('`:this/is/a/path')
    assert kx.K(Path(500 * 'a')) == q('`:' + 500 * 'a')


def test_from_ellipsis(q, kx):
    assert kx.K(...).t == 101
    assert str(kx.K(...)) == '::'
    f = q('{[a;b;c] a + b * c}')
    assert f(..., ..., ...)(1, 2, 3) == 7
    assert f(..., 2, 3)(1) == 7
    assert f(1, ..., 3)(2) == 7
    assert f(1, 2, ...)(3) == 7
    assert f(..., c=3, b=2)(1) == 7
    assert f(..., ..., 3)(b=2, a=1) == 7
    with pytest.raises(TypeError):
        kx.IntAtom(...)
    with pytest.raises(TypeError):
        kx.Identity(...)


def test_from_fileno(q, tmp_path, kx):
    with open(tmp_path/'testfile.txt', 'w') as f:
        q('{x .Q.s til 10}', f)
    with open(tmp_path/'testfile.txt', 'r') as f:
        content = f.read()
    assert content == '0 1 2 3 4 5 6 7 8 9\n'

    class File:
        fileno = -1

    f = File()
    assert kx.K(f).py() == -1
    assert kx.IntAtom(f).py() == -1
    with pytest.raises(TypeError):
        kx.LongAtom(f)


@pytest.mark.ipc(licensed_only=True)
def test_from_callable(q, capsys, kx):
    def simple_func(x, y, z):
        return [x, x, y, z, y]

    k_simple_func = kx.K(simple_func)
    assert k_simple_func(1, 2, 3).py() == [1, 1, 2, 3, 2]

    k_partial = kx.K(partial(simple_func, 9, 8))
    assert k_partial(7).py() == [9, 9, 8, 7, 8]

    k_unbound_method = kx.K(kx.Dictionary.keys)
    assert k_unbound_method({'a': None}).py() == ['a']

    with pytest.raises(ValueError):
        kx.K(lambda a, b, c, d, e, f, g, h, i: None)

    d = q('enlist[`a]!enlist[::]')
    k_bound_method = kx.K(d.keys)
    assert k_bound_method().py() == ['a']

    k_args_kwargs_func = kx.K(
        lambda *args, **kwargs: print(*args, **{k: str(v) for k, v in kwargs.items()}))
    k_args_kwargs_func('a b c d'.split(), q('`sep`end!(`$".";`$"|END")'))
    out, _ = capsys.readouterr()
    assert out == 'a.b.c.d|END'

    def kwargs_only_func(*, cats=None, bats=None):
        return [cats, bats]

    k_kwargs_only_func = kx.K(kwargs_only_func)
    assert k_kwargs_only_func(bats='bats', cats='cats').py() == ['cats', 'bats']


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_numpy_ndarray_1(kx):
    assert kx.K(np.array([0.0, 1234.003e42, -12])).py() \
        == [0.0, 1234.003e42, -12]
    assert kx.K(np.array([0.0, 1234.003e42, -12], np.float32)).py() \
        == [0.0, float('inf'), -12]
    np_ints = np.array([1, 2, 12, 9113058913442122903])
    assert kx.K(np_ints).py() == [1, 2, 12, 9113058913442122903]
    assert kx.K(np_ints.astype(np.int32)).py() == [1, 2, 12, 2028784791]
    assert kx.K(np_ints.astype(np.int16)).py() == [1, 2, 12, -13161]
    assert kx.K(np_ints.astype(np.uint8)).py() == [1, 2, 12, 151]
    n = kx.K(np.array([uuid4() for _ in range(8)]))
    assert np.array_equal(kx.K(n.np()).np(), n.np())
    assert np.array_equal(kx.K(n.np(raw=True)).np(), n.np())
    assert kx.K(np.array([True, True, False, True, False, False])).py() == \
        [True, True, False, True, False, False]
    if kx.licensed:
        with pytest.raises(TypeError):
            kx.K(np.array([lambda x:x, pytest]))
    assert isinstance(kx.K(np.array(['abcd', 'wxyz'])), kx.SymbolVector)
    assert kx.K(np.array(['abcd', 'wxyz'])).py() == ['abcd', 'wxyz']
    shaped_array = np.random.randint(0, 1000000, (3, 1, 2, 3))
    sk = kx.K(shaped_array)
    if kx.licensed:
        shape = kx.q('k){*{0h~@x[1]}{(x[0],#*x[1];*x[1])}/(,#x;x)}')
        assert list(shape(sk)) == list(shaped_array.shape)
    assert sk.py() == shaped_array.tolist()
    with pytest.raises(TypeError):
        kx.Function(np.array([]))
    with pytest.raises(TypeError):
        kx.toq(np.array([]), kx.Function)
    assert kx.K(np.array([b'a', b'bc', b'def'], dtype=object)).py() \
        == [b'a', b'bc', b'def']
    assert kx.K(np.array([1.2, 1.3, [1.4, 1.5]], dtype=object)).py() \
        == [1.2, 1.3, [1.4, 1.5]]
    assert kx.K(np.array([b'a', b'ab', 1.3, [1.3, 1.2], 'x'],
                dtype=object)).py() == [b'a', b'ab', 1.3, [1.3, 1.2], 'x']

    ar = np.array([1, 2], np.dtype('uint16'))
    ark = kx.K(ar)
    at = ar[0]
    atk = kx.K(at)
    assert isinstance(ark, kx.IntVector)
    assert ark.py() == [1, 2]
    assert isinstance(atk, kx.LongAtom) # ToDo
    assert atk.py() == 1

    ar = np.array([1, 2], np.dtype('uint32'))
    ark = kx.K(ar)
    at = ar[0]
    atk = kx.K(at)
    assert isinstance(ark, kx.LongVector)
    assert ark.py() == [1, 2]
    assert isinstance(atk, kx.LongAtom)
    assert atk.py() == 1

    ar = np.array([1, 2], np.dtype('int8'))
    ark = kx.K(ar)
    at = ar[0]
    atk = kx.K(at)
    assert isinstance(ark, kx.ShortVector)
    assert ark.py() == [1, 2]
    assert isinstance(atk, kx.LongAtom) # ToDo
    assert atk.py() == 1

    ar = np.array([1, 2], np.dtype('float16'))
    ark = kx.K(ar)
    at = ar[0]
    atk = kx.K(at)
    assert isinstance(ark, kx.RealVector)
    assert ark.py() == [1, 2]
    assert isinstance(atk, kx.RealAtom)
    assert atk.py() == 1


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_numpy_ndarray_2(kx):
    assert kx.K(np.array([], dtype=object)).t == 0
    with pytest.raises(TypeError):
        kx.K(np.array([1, 2, 3], dtype=np.uint64))
    if kx.licensed:
        assert kx.K(np.array([b'x', b'y', b'z'], dtype='|S1')).py() == b'xyz'
        assert kx.K(np.array(['K', 'X'])).py() == ['K', 'X']
        assert kx.CharVector(np.array(['K', 'X'])).py() == b'KX'
        assert kx.CharVector(np.array([b'K', b'X'])).py() == b'KX'
        assert kx.K(np.array([b'K', b'X'])).py() == b'KX'
        assert kx.K(np.array([b'K', 'X'])).py() == ['K', 'X']
        assert kx.K(np.array([b'string', b'test'], dtype='|S7')).py() == [b'string', b'test']
        assert kx.K(np.array([b'string', b'test'], dtype='|S10')).py() == [b'string', b'test']
        assert isinstance(kx.K(np.array(['a', 'b', None, 'c'], dtype=object)), kx.SymbolVector)
        assert kx.K(np.array(['a', 'b', None, 'c'], dtype=object)).py() == ['a', 'b', '', 'c']
        with pytest.raises(TypeError):
            kx.CharVector(np.array(['KX', 'Labs']))
    contiguous = np.arange(16).reshape((4, 4))
    assert contiguous.data.c_contiguous
    non_contiguous = contiguous.T
    assert not non_contiguous.data.c_contiguous
    assert kx.K(contiguous).py() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    assert [list(x) for x in np.asarray(kx.K(contiguous).py()).T] == kx.K(non_contiguous).py()
    assert kx.List(np.array([1, '*', 2], object)).py() == [1, '*', 2]


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_numpy_ndarray_3(kx):
    np_rng = np.random.default_rng()
    time_array_np = np_rng.integers(0, 100000000, 100).astype('timedelta64[ms]')
    time_array = kx.K(time_array_np)
    assert all(time_array.np() == time_array_np)
    date_array_np = np_rng.integers(0, 10000, 100).astype('datetime64[D]')
    date_array = kx.K(date_array_np)
    assert all(date_array.np() == date_array_np)
    byte_array_np = np_rng.integers(0, 10000, 100).astype(np.uint8)
    byte_array = kx.K(byte_array_np)
    assert all(byte_array.np() == byte_array_np)
    month_array_np = np_rng.integers(0, 1000, 100).astype('datetime64[M]')
    month_array = kx.K(month_array_np)
    assert all(month_array.np() == month_array_np)
    tstamp_array_np = np_rng.integers(0, 2 ** 63 - 1, 100).astype('datetime64[ns]')
    tstamp_array = kx.K(tstamp_array_np)
    assert all(tstamp_array.np() == tstamp_array_np)
    tspan_array_np = np_rng.integers(0, 2 ** 63 - 1, 100).astype('timedelta64[ns]')
    tspan_array = kx.K(tspan_array_np)
    assert all(tspan_array.np() == tspan_array_np)
    minute_array_np = np_rng.integers(0, 2 ** 31 - 1, 100).astype('timedelta64[m]')
    minute_array = kx.K(minute_array_np)
    assert all(minute_array.np() == minute_array_np)


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_numpy_incompatible_types(kx):
    for ty in (np.int64, np.uint64):
        with pytest.raises(TypeError):
            kx.ShortVector(np.arange(10).astype(ty))
    with pytest.raises(TypeError):
        kx.RealVector(np.random.rand(10).astype('float64'))
    with pytest.raises(TypeError):
        kx.LongVector(np.array(['1', '2', '34']))
    with pytest.raises(TypeError):
        kx.ByteVector('abcdef')


@pytest.mark.nep49
def test_from_arrays_with_nulls(q, kx, pd):
    assert all(kx.K(np.ma.MaskedArray([1, 2, 3], [0, 1, 0])) == q('1 0N 3'))
    with pytest.raises(TypeError):
        kx.K(np.ma.MaskedArray([object, object(), None], [1, 0, 1]))
    assert all(kx.K(pd.Series([1, pd.NA, 3], dtype=pd.Int64Dtype())) == q('1 0N 3'))
    with pytest.raises(TypeError):
        assert kx.K(pd.Series([object(), pd.NA, 3]))


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_pandas_dataframe(kx, pd):
    d = {'a': [0, 1, 2, 3],
         'b': ['a', 'b', 'c', 'd'],
         'c': [b'w', b'x', b'y', b'z'],
         'd': [0.1, 0.2, 0.3, 0.4]}
    df = pd.DataFrame(d)
    assert kx.K(df).t == 98
    assert all(kx.K(df).pd() == df)
    idxdf = df.set_index(['a', 'b'])
    assert kx.K(idxdf).t == 99
    assert all(kx.K(idxdf).pd() == idxdf)
    assert all(kx.K(idxdf).pd().index == idxdf.index)
    with pytest.raises(TypeError):
        kx.List(df)


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_pandas_NA(kx, pd):
    assert kx.toq(pd.NA).py() is None


@pytest.mark.nep49
def test_from_pandas_dataframe_licensed(q, kx):
    q.system.console_size = [25, 80]
    t = q('([] a:til 4; b:"abcd"; c:`w`x`y`z; d:100+til 4)').pd()
    assert t.equals(kx.K(t).pd())

    mkt = q('([k1:`a`b`a`a`a`b;k2:100+til 6] x:til 6; y:`multi`keyed`table`m`k`t)')
    assert all(kx.K(mkt.pd()) == mkt.pd())  # check round trip accuracy
    assert all(kx.K(mkt.pd()).pd().index == mkt.pd().index)

    # Test that all type conversions are equal
    q('N:100')
    gen_q_datatypes_table(q, 'dset_1D', int(q('N')))
    q('gen_names:{"dset_",/:x,/:string til count y}')
    type_tab = q('flip (`$gen_names["tab";dset_1D])!N#\'dset_1D')
    assert all(q.raze(q.raze(q.value((kx.K(type_tab.pd()) == type_tab).flip))))

    nested_tab = q('([]a:1 2 3;b:("a";"ab";"abc");(1;1 2;("abc";`a)))').pd()
    assert nested_tab.equals(kx.K(nested_tab).pd())
    time_tab = q('([]a:1 2 3;b:("a";"ab";"abc");(1;"t"$1 2;("abc";`a)))').pd()
    assert time_tab.equals(kx.K(time_tab).pd())


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_complex_pandas_dataframe(kx, pd):
    df = pd.DataFrame(
        {
            'StringDTypeCol': pd.arrays.StringArray(np.array(['', '', ' ', 'asdf'], dtype='O')),
            'f32': pd.array([0.0, 0, 2.718281828, 3.14159265359], dtype=np.float32),
            'f64': pd.array([0, 98765.1234, float('nan'), -1010.1010], dtype=np.float64),
            'U': pd.array(['u', 'is', 'for', 'unicode'], dtype='U'),
            'S': pd.array([b'b', b'b', b'b', b'b?'], dtype='S'),
            'int16': pd.array([-11111, 94, 12345, 0], dtype=np.int16),
            'int32': pd.array([0, 0, 2**30, 0], dtype=np.int32),
            'int64': pd.array([-2**60, 0, 0, 0], dtype=np.int64),
        },
        index=pd.MultiIndex.from_arrays(
            [
                pd.array([2**x for x in (1, 38, 59, 62)], dtype='datetime64[ns]'),
                np.arange(4),
            ],
            names=['k1', 'k2'],
        ),
    )
    # for x in ((0, 0), (1, 1), (0, 2), (2, 3), (1, 4), (3, 5), (0, 6), (1, 6), (2, 7), (3, 7)):
    #     df.iloc[x] = pd.NA
    assert str(df.to_dict()) == str(kx.toq(df).pd().to_dict())


@pytest.mark.nep49
def test_from_pandas_dataframe_no_col_names(kx, pd):
    d = [[0, 1, 2, 3],
         ['a', 'b', 'c', 'd'],
         [b'w', b'x', b'y', b'z'],
         [0.1, 0.2, 0.3, 0.4]]
    df = pd.DataFrame(d)
    assert kx.K(df).t == 98
    # Check single column df
    d = [1, 2, 3]
    df = pd.DataFrame(d)
    assert kx.K(df).t == 98


@pytest.mark.nep49
def test_from_pandas_index(q, kx, pd):
    t = q('([] a:til 4; b:"abcd"; c:`w`x`y`z; d:100+til 4)')
    assert all(kx.K(t.pd().index) == q('0 1 2 3'))
    mkt = q('([k1:`a`b`a`a`a`b;k2:100+til 6] x:til 6; y:`multi`keyed`table`m`k`t)')
    assert (kx.K(mkt.pd().index)._values == q('key', mkt)._values).all()
    assert kx.K(pd.MultiIndex.from_product([[0, 1], [0, 1]])).py() == \
        {'0': [0, 0, 1, 1], '1': [0, 1, 0, 1]}
    with pytest.raises(TypeError):
        kx.K(pd.IntervalIndex.from_tuples([(0, 1), (1, 4), (4, 6)]))


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_pandas_series(kx, pd):
    float_vector = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])
    assert all(float_vector == kx.K(float_vector).pd())
    real_vector = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], dtype='float32')
    assert all(real_vector == kx.K(real_vector).pd())
    char_vector = pd.Series([b'a', b'b', b'c'])
    assert all(char_vector == kx.K(char_vector).pd())
    string_vector = pd.Series([b'abc', b'def', b'ghij'])
    assert all(string_vector == kx.K(string_vector).pd())
    symbol_vector = pd.Series(['a', 'b', 'c', 'd'])
    assert all(symbol_vector == kx.K(symbol_vector).pd())
    time_vector = pd.Series([1000000, 2000000, 3000000, 4000000, 5000000], dtype='timedelta64[ns]') # noqa
    assert all(time_vector == kx.K(time_vector).pd())
    time_vector = pd.Series([1000000, 2000000, 3000000, 4000000, 5000000], dtype='timedelta64[us]') # noqa
    assert all(time_vector == kx.K(time_vector).pd())
    timestamp_vector = pd.Series([0, 1, 2, 3, 4], dtype='datetime64[ns]')
    assert all(timestamp_vector == kx.K(timestamp_vector).pd())


@pytest.mark.nep49
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
@pytest.mark.xfail(reason="KXI-33749", strict=False)
def test_from_pandas_series_licensed(q, kx):
    float_vector = q('100?1f')
    assert all(float_vector == kx.K(float_vector.pd()))
    real_vector = q('100?1e')
    assert all(real_vector == kx.K(real_vector.pd()))
    char_vector = q('100?"abc"')
    assert all(char_vector == kx.K(char_vector.pd()))
    string_vector = q('100?("abc";"defg")')
    with pytest.raises(TypeError):
        all(string_vector == kx.K(string_vector.pd()))
    assert (string_vector == kx.K(string_vector.pd())).all()
    symbol_vector = q('100?`a`b`c')
    assert all(symbol_vector == kx.K(symbol_vector.pd()))
    time_vector = q('100?0t')
    assert all(time_vector == kx.K(time_vector.pd()))
    timestamp_vector = q('100?0p')
    assert all(timestamp_vector == kx.K(timestamp_vector.pd()))

    table_name = gen_q_datatypes_table(q, 'dset_1D')
    assert (q(table_name) == kx.K(q(table_name).pd())).all()
    assert (q(f'flip {table_name}') == kx.K(q(f'flip {table_name}').pd())).all()


@pytest.mark.nep49
def test_from_pandas_categorical(q, kx, pd):
    cat = pd.Categorical(['aaa', 'bbb', 'ccc'])
    rez = kx.toq(cat)
    assert isinstance(rez, kx.EnumVector)
    assert isinstance(q('enum0'), kx.SymbolVector)
    assert isinstance(rez.pd(), pd.Series)
    assert isinstance(rez.pd().values, pd.Categorical)
    assert all(rez.pd() == cat)

    cat = pd.Series(['aaa', 'bbb', 'ccc'], dtype='category', name='cat')
    rez = kx.toq(cat)
    enum = q['cat']
    assert isinstance(rez, kx.EnumVector)
    assert isinstance(q('cat'), kx.SymbolVector)
    assert isinstance(rez.pd(), pd.Series)
    assert isinstance(rez.pd().values, pd.Categorical)
    assert all(rez.pd() == cat)

    df = pd.DataFrame()
    df['index'] = cat
    df['series'] = cat
    df['x'] = range(len(df))
    original_df = df.copy()
    rez = kx.toq(df)
    assert isinstance(rez, kx.Table)
    assert isinstance(rez['index'], kx.EnumVector)
    assert isinstance(rez['series'], kx.EnumVector)
    assert isinstance(q('index'), kx.SymbolVector)
    assert isinstance(q('series'), kx.SymbolVector)
    assert isinstance(rez.pd(), pd.DataFrame)
    assert isinstance(rez.pd()['index'].values, pd.Categorical)
    assert isinstance(rez.pd()['series'].values, pd.Categorical)
    assert all(rez.pd()['index'] == cat)
    assert all(rez.pd()['series'] == cat)

    df.set_index('index', inplace=True)
    rez = kx.toq(df)
    assert isinstance(rez, kx.KeyedTable)
    assert isinstance(q('{key[x]`index}', rez), kx.EnumVector)
    assert isinstance(q('{value[x]`series}', rez), kx.EnumVector)
    assert isinstance(q('index'), kx.SymbolVector)
    assert isinstance(q('series'), kx.SymbolVector)
    assert isinstance(rez.pd(), pd.DataFrame)
    assert isinstance(rez.pd().index, pd.CategoricalIndex)
    assert isinstance(rez.pd()['series'].values, pd.Categorical)
    assert all(rez.pd().index == cat)
    assert all(rez.pd()['series'].values == cat)

    df2 = pd.DataFrame()
    # Can re-enumerate with same symbols (or subset) in any order
    cat2 = pd.Series(['bbb', 'aaa', 'ccc'], dtype='category')
    df2['series'] = cat2
    rez = kx.toq(df2)
    assert all(rez.pd()['series'] == cat2)
    # enumerated against same symbol in q
    assert q('key', rez['series']) == q('`series')
    # no mutation on original df (before toq)
    assert all((df.reset_index() == original_df))
    # no mutation of the initial symbol
    assert all(enum==q('series'))

    # Test adding new value to existing enum
    cat = pd.Series(['aaa', 'bbb', 'ccc', 'ddd'], dtype='category', name='index')
    rez = kx.toq(cat)
    assert isinstance(rez, kx.EnumVector)
    assert isinstance(q('cat'), kx.SymbolVector)
    assert isinstance(rez.pd(), pd.Series)
    assert isinstance(rez.pd().values, pd.Categorical)
    assert all(rez.pd() == cat)
    # no mutation of original df
    assert all((df.reset_index() == original_df))

    # Test that we don't overwrite an enum already existing on q side
    sym = q('sym:`aaa`bbb`ccc; a:`sym$10?sym; sym').py()
    assert(sym == ['aaa', 'bbb', 'ccc'])
    df = q('([] sym:`sym$10?enlist `aaa)').pd()
    kx.toq(df)
    sym = q('sym').py()
    assert(sym == ['aaa', 'bbb', 'ccc'])

    assert kx.toq.ENUMS == ['enum0', 'cat', 'index', 'series', 'sym']


@pytest.mark.nep49
def test_toq_pd_tabular_ktype(q, kx):
    df = pd.DataFrame.from_dict({'x': [1, 2], 'y': ['a', 'b']})
    assert kx.toq(df).dtypes['datatypes'].py() == [b'kx.LongAtom', b'kx.SymbolAtom']
    kval = {'x': kx.FloatVector}
    assert kx.toq(df, ktype=kval).dtypes['datatypes'].py() == [b'kx.FloatAtom', b'kx.SymbolAtom']
    kval = {'x': kx.FloatVector, 'y': kx.CharVector}
    assert kx.toq(df, ktype=kval).dtypes['datatypes'].py() == [b'kx.FloatAtom', b'kx.CharVector']
    with pytest.raises(ValueError, match="Column name passed in dictionary not present in df table"): # noqa: E501
        kx.toq(df, ktype={'x1': kx.FloatVector})
    with pytest.raises(kx.QError, match="Not supported:.*"):
        kx.toq(df, ktype={'x': kx.GUIDVector})


@pytest.mark.nep49
def test_toq_pa_tabular_ktype(q, kx, pa):
    pdtab = pd.DataFrame.from_dict({'x': [1, 2], 'y': ['a', 'b']})
    df = pa.Table.from_pandas(pdtab)
    assert kx.toq(df).dtypes['datatypes'].py() == [b'kx.LongAtom', b'kx.SymbolAtom']
    kval = {'x': kx.FloatVector}
    assert kx.toq(df, ktype=kval).dtypes['datatypes'].py() == [b'kx.FloatAtom', b'kx.SymbolAtom']
    kval = {'x': kx.FloatVector, 'y': kx.CharVector}
    assert kx.toq(df, ktype=kval).dtypes['datatypes'].py() == [b'kx.FloatAtom', b'kx.CharVector']
    with pytest.raises(ValueError, match="Column name passed in dictionary not present in df table"): # noqa: E501
        kx.toq(df, ktype={'x1': kx.FloatVector})
    with pytest.raises(kx.QError, match="Not supported:.*"):
        kx.toq(df, ktype={'x': kx.GUIDVector})


@pytest.mark.unlicensed
@pytest.mark.xfail(reason="Windows test execution fails intermittently with license error", strict=False) # noqa: E501
def test_toq_dict_error(q, kx, pa):
    pdSeries = q('1 2 3').pd()
    with pytest.raises(TypeError, match=r"'ktype' .*"):
        kx.toq(pdSeries, {'x': kx.LongVector})
    paArray = q('1 2 3').pa()
    with pytest.raises(TypeError, match=r"'ktype' .*"):
        kx.toq(paArray, {'x': kx.LongVector})
    npArray = q('1 2 3').np()
    with pytest.raises(TypeError, match=r"'ktype' .*"):
        kx.toq(npArray, {'x': kx.LongVector})
    pydict = {'x': 1, 'y': 2}
    with pytest.raises(TypeError, match=r"'ktype' .*"):
        kx.toq(pydict, {'x': kx.LongVector})


# TODO: Add this mark back once this test is consistently passing again, adding more calls to it
# each test pass just increases the chance of the tests failing.
@pytest.mark.nep49
@pytest.mark.xfail(reason="KXI-11980", strict=False)
def test_from_arrow_licensed(q, kx, pa):
    float_vector = q('100?1f')
    assert all(float_vector == kx.K(float_vector.pa()))
    real_vector = q('100?1e')
    assert all(real_vector == kx.K(real_vector.pa()))
    char_vector = q('100?"abc"')
    assert all(char_vector == kx.K(char_vector.pa()))
    string_vector = q('100?("abc";"defg")')
    with pytest.raises(TypeError):
        all(string_vector == kx.K(string_vector.pa()))
    assert (string_vector == kx.K(string_vector.pa())).all()
    symbol_vector = q('100?`a`b`c')
    assert all(symbol_vector == kx.K(symbol_vector.pa()))
    time_vector = q('100?0t')
    assert all(time_vector == kx.K(time_vector.pa()))
    timestamp_vector = q('100?0p')
    assert all(timestamp_vector == kx.K(timestamp_vector.pa()))
    with pytest.raises(TypeError):
        guid_vector = q('100?0Ng')
        kx.K(guid_vector.pa())

    q('N:100')
    gen_q_datatypes_table(q, 'dset_1D', int(q('N')))
    q('gen_names:{"dset_",/:x,/:string til count y}')
    type_tab = q('flip (`$gen_names["tab";dset_1D])!N#\'dset_1D')
    assert (type_tab == kx.K(type_tab.pa())).all()


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_from_arrow(kx, pa, pd):
    floatVector = pa.array([0.0, 0.1, 0.2, 0.3, 0.4])
    assert floatVector == kx.K(floatVector).pa()

    realVector = pa.array([0.0, 0.1, 0.2, 0.3, 0.4], type=pa.float32())
    assert realVector == kx.K(realVector).pa()

    charVector = pa.array([b'a', b'b', b'c', b'd', b'e'])
    assert charVector == kx.K(charVector).pa()

    stringVector = pa.array([b'abc', b'def', b'hij', b'klmn'])
    assert stringVector == kx.K(stringVector).pa()

    symbolVector = pa.array(['a', 'b', 'c', 'd', 'e'])
    assert symbolVector == kx.K(symbolVector).pa()

    timeSeries = pd.Series([1000000, 2000000, 3000000, 4000000, 5000000], dtype='timedelta64[ns]')
    timeVector = pa.Array.from_pandas(timeSeries)
    assert timeVector == kx.K(timeVector).pa()

    timestampVector = pa.Array.from_pandas(pd.Series([0, 1, 2, 3, 4], dtype='datetime64[ns]'))
    assert timestampVector == kx.K(timestampVector).pa()

    a = pa.chunked_array([[1, 2, 3], [4, 5, 6]])
    assert a.combine_chunks() == kx.K(a).pa()


@pytest.mark.licensed
@pytest.mark.nep49
def test_null_roundtrip(kx):
    # TODO: Won't pass for null GUID, to be fixed by KXI-11866
    # kx.q('ty:2 5 6 7 8 9 10 11 12 13 14 16 17 18 19h')
    # kx.q('nulls:(0Ng;0Nh;0Ni;0Nj;0Ne;0n;" ";`;0Np;0Nm;0Nd;0Nn;0Nu;0Nv;0Nt)')
    kx.q('ty:5 6 7 8 9 10 11 12 13 14 16 17 18 19h')
    kx.q('nulls:(0Nh;0Ni;0Nj;0Ne;0n;" ";`;0Np;0Nm;0Nd;0Nn;0Nu;0Nv;0Nt)')
    t = kx.q('flip ({`$.Q.t x} each ty)!{enlist nulls[x]} each til count ty')
    for col in t:
        assert (
            kx.q('{x 0}', t[col])
            == kx.toq(t[col].np(), handle_nulls=True)
        ).all()
    assert (t == kx.toq(t.pd(), handle_nulls=True)).all().all()


@pytest.mark.unlicensed
@pytest.mark.nep49
def test_nep_49_refcount(kx):
    if kx.k_allocator:
        a = np.array([1, 2, 3, 4, 5])
        start_count = getrefcount(a)
        b = kx.toq(a)
        assert getrefcount(a) == start_count + 1
        del b
        assert getrefcount(a) == start_count

        a = np.random.randint(2**30, size=10).astype('timedelta64[s]').astype('timedelta64[ns]')
        start_count = getrefcount(a)
        b = kx.toq(a) # noqa
        assert getrefcount(a) == start_count + 1
        del b
        assert getrefcount(a) == start_count


@pytest.mark.unlicensed
def test_resolve_k_type(kx):
    _resolve_k_type = kx.toq._resolve_k_type
    assert _resolve_k_type(-1) == kx.BooleanAtom
    with pytest.raises(TypeError):
        _resolve_k_type(-99)
    assert _resolve_k_type(19) == kx.TimeVector
    assert _resolve_k_type(kx.Dictionary) == kx.Dictionary
    with pytest.raises(TypeError):
        _resolve_k_type('int')


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.toq), list)
    assert sorted(dir(kx.toq)) == dir(kx.toq)


def test_Float64Index(kx):
    pdFloat64Index = pd.DataFrame(data={'a': [1.0, 2.0, 3.0], 'b': [3, 4, 5]}).set_index('a')
    assert all(kx.q('([a:1 2 3.0] b:3 4 5)') == kx.toq(pdFloat64Index))


def test_str_as_char(kx):
    string = 'qstring'
    str_list = ['qstring0', 'qstring1']
    str_dict = {'a': {'b': 'qstring0'}, 'b': 'qstring1'}
    np_list = np.array(str_list)
    np_list_2d = np.array([str_list, str_list])
    str_tab = pd.DataFrame(data={'x': np_list})

    assert isinstance(kx.toq(string), kx.SymbolAtom)
    qchar_string = kx.toq(string, strings_as_char=True)
    assert isinstance(qchar_string, kx.CharVector)
    assert all(qchar_string == b'qstring')

    assert isinstance(kx.toq(str_list), kx.SymbolVector)
    qchar_list = kx.toq(str_list, strings_as_char=True)
    assert isinstance(qchar_list, kx.List)
    assert isinstance(qchar_list[0], kx.CharVector)
    assert all(qchar_list[0] == b'qstring0')

    qsym_dict = kx.toq(str_dict)
    assert isinstance(qsym_dict['a']['b'], kx.SymbolAtom)
    assert qsym_dict['a']['b'] == 'qstring0'
    assert isinstance(qsym_dict['b'], kx.SymbolAtom)
    assert qsym_dict['b'] == 'qstring1'

    qchar_dict = kx.toq(str_dict, strings_as_char=True)
    assert isinstance(qchar_dict['a']['b'], kx.CharVector)
    assert all(qchar_dict['a']['b'] == b'qstring0')
    assert isinstance(qchar_dict['b'], kx.CharVector)
    assert all(qchar_dict['b'] == b'qstring1')

    qsym_np_list = kx.toq(np_list)
    assert isinstance(qsym_np_list, kx.SymbolVector)
    qchar_np_list = kx.toq(np_list, strings_as_char=True)
    assert isinstance(qchar_np_list, kx.List)
    assert isinstance(qchar_np_list[0], kx.CharVector)
    assert all(qchar_np_list[0] == b'qstring0')

    qsym_np_list_2d = kx.toq(np_list_2d)
    assert isinstance(qsym_np_list_2d, kx.List)
    assert isinstance(qsym_np_list_2d[0], kx.SymbolVector)
    qchar_np_list_2d = kx.toq(np_list_2d, strings_as_char=True)
    assert isinstance(qchar_np_list_2d, kx.List)
    assert isinstance(qchar_np_list_2d[0], kx.List)
    assert isinstance(qchar_np_list_2d[0][0], kx.CharVector)
    assert all(qchar_np_list_2d[0][0] == b'qstring0')

    qsym_tab = kx.toq(str_tab)
    assert isinstance(qsym_tab['x'], kx.SymbolVector)
    qchar_tab = kx.toq(str_tab, strings_as_char=True)
    assert isinstance(qchar_tab['x'], kx.List)
    assert isinstance(qchar_tab['x'][0], kx.CharVector)


def test_column_variable_tree_phrase(kx):
    col = kx.Column('x')
    assert kx.toq(col).py() == 'x'

    col_gt = 1 < kx.Column('x')
    assert kx.toq(col_gt).py() == [kx.Operator('>'), 'x', 1]

    col_max = kx.Column('x').max()
    assert kx.toq(col_max).py() == [kx.q('max'), 'x']

    var = kx.Variable('nvar')
    assert kx.toq(var) == 'nvar'

    tree = kx.ParseTree(kx.q.parse(b'x=`a'))
    assert kx.toq(tree).py() == [kx.Operator('='), 'x', ['a']]

    phrase_0 = kx.QueryPhrase(kx.Column('x') == 'a')
    assert kx.toq(phrase_0).py() == [[kx.Operator('='), 'x', ['a']]]

    phrase_1 = kx.QueryPhrase({'asA': 'a', 'negB': [kx.q('neg'), 'b']})
    assert {'asA': 'a', 'negB': [kx.q('neg'), 'b']} == phrase_1.to_dict()


def test_pyarrow(kx):
    import pyarrow as pa

    def test_pa(arr, karr, kat):
        at = arr[0]
        arr_toq= kx.toq(arr)
        at_toq = kx.toq(at)
        assert kx.q('~', karr, arr_toq).py()
        assert kx.q('~', kat, at_toq).py()

    # pyarrow.int8
    test_pa(pa.array([1], pa.int8()), kx.q('1'), kx.q('1')) # ToDo
    test_pa(pa.array([1, 2], pa.int8()), kx.q('1 2h'), kx.q('1'))

    # pyarrow.int16
    test_pa(pa.array([1], pa.int16()), kx.q('1'), kx.q('1')) # ToDo
    test_pa(pa.array([1, 2], pa.int16()), kx.q('1 2h'), kx.q('1'))

    # pyarrow.int32
    test_pa(pa.array([1], pa.int32()), kx.q('1'), kx.q('1')) # ToDo
    test_pa(pa.array([1, 2], pa.int32()), kx.q('1 2i'), kx.q('1'))

    # pyarrow.int64
    test_pa(pa.array([1], pa.int64()), kx.q('1'), kx.q('1'))
    test_pa(pa.array([1, 2], pa.int64()), kx.q('1 2'), kx.q('1'))

    # pyarrow.uint8
    test_pa(pa.array([0], type=pa.uint8()), kx.q('0'), kx.q('0')) # ToDo
    test_pa(pa.array([0, 1, 2], type=pa.uint8()), kx.q('0x000102'), kx.q('0')) # ToDo

    # pyarrow.uint16
    test_pa(pa.array([0], type=pa.uint16()), kx.q('0'), kx.q('0')) # ToDo
    test_pa(pa.array([0, 1, 2], type=pa.uint16()), kx.q('0 1 2i'), kx.q('0')) # ToDo

    # pyarrow.uint32
    test_pa(pa.array([0], type=pa.uint32()), kx.q('0'), kx.q('0'))
    test_pa(pa.array([0, 1, 2], type=pa.uint32()), kx.q('0 1 2'), kx.q('0'))

    # pyarrow.uint64
    # test_pa(pa.array([0, 1, 2], type=pa.uint64()), kx.q('(),1'), kx.q('1'))

    # pyarrow.float16
    test_pa(pa.array([np.float16(1.0)], pa.float16()), kx.q('1e'), kx.q('1e')) # ToDo
    test_pa(pa.array([np.float16(1.0), np.float16(2.0)], pa.float16()), kx.q('1 2e'),
            kx.q('1e'))

    # pyarrow.float32
    test_pa(pa.array([np.float32(1.0)], pa.float32()), kx.q('1e'), kx.q('1f')) # ToDo
    test_pa(pa.array([np.float32(1.0), np.float32(2.0)], pa.float32()), kx.q('1 2e'),
            kx.q('1f'))

    # pyarrow.float64
    test_pa(pa.array([np.float64(1.0)], pa.float64()), kx.q('1f'), kx.q('1f')) # ToDo
    test_pa(pa.array([np.float64(1.0), np.float64(2.0)], pa.float64()), kx.q('1 2f'),
            kx.q('1f'))

    # pyarrow.time32
    test_pa(pa.array([1], pa.time32('s')), kx.q('0D00:00:01'), kx.q('0D00:00:01')) # ToDo
    test_pa(pa.array([1, 2], pa.time32('s')), kx.q('0D00:00:01 0D00:00:02'), kx.q('0D00:00:01'))

    # ToDo
    test_pa(pa.array([1], pa.time32('ms')), kx.q('0D00:00:00.001'), kx.q('0D00:00:00.001'))
    test_pa(pa.array([1, 2], pa.time32('ms')), kx.q('0D00:00:00.001 0D00:00:00.002'),
            kx.q('0D00:00:00.001'))

    # pyarrow.time64
    test_pa(pa.array([1], pa.time64('us')), kx.q('0D00:00:00.000001'),
            kx.q('0D00:00:00.000001')) # ToDo
    test_pa(pa.array([1, 2], pa.time64('us')), kx.q('0D00:00:00.000001  0D00:00:00.000002'),
            kx.q('0D00:00:00.000001'))

    # ToDo: pyarrow.lib.ArrowInvalid: Value 1 has non-zero nanoseconds
    # test_pa(pa.array([1], pa.time64('ns')), kx.q('(),1'), kx.q('1'))

    test_pa(pa.array([0], pa.time64('ns')), kx.q('0D00'), kx.q('0D00')) # ToDo
    test_pa(pa.array([0, 1000], pa.time64('ns')), kx.q('0D00 0D00:00:00.000001'), kx.q('0D00'))

    # pyarrow.timestamp
    test_pa(pa.array([1], pa.timestamp('ms')), kx.q('1970.01.01D00:00:00.001'),
            kx.q('1970.01.01D00:00:00.001'))
    test_pa(pa.array([1, 2], pa.timestamp('ms')),
            kx.q('1970.01.01D00:00:00.001  1970.01.01D00:00:00.002'),
            kx.q('1970.01.01D00:00:00.001'))

    test_pa(pa.array([1], pa.timestamp('ns')), kx.q('1970.01.01D00:00:00.000000001'),
            kx.q('1970.01.01D00:00:00.000000001'))
    test_pa(pa.array([1, 2], pa.timestamp('ns')),
            kx.q('1970.01.01D00:00:00.000000001  1970.01.01D00:00:00.000000002'),
            kx.q('1970.01.01D00:00:00.000000001'))

    test_pa(pa.array([1], pa.timestamp('us')), kx.q('1970.01.01D00:00:00.000001000'),
            kx.q('1970.01.01D00:00:00.000001000'))
    test_pa(pa.array([1, 2], pa.timestamp('us')),
            kx.q('1970.01.01D00:00:00.000001000  1970.01.01D00:00:00.000002000'),
            kx.q('1970.01.01D00:00:00.000001000'))

    test_pa(pa.array([1], pa.timestamp('s')), kx.q('1970.01.01D00:00:01.000000000'),
            kx.q('1970.01.01D00:00:01.000000000'))
    test_pa(pa.array([1, 2], pa.timestamp('s')),
            kx.q('1970.01.01D00:00:01.000000000 1970.01.01D00:00:02.000000000'),
            kx.q('1970.01.01D00:00:01.000000000'))

    # pyarrow.date32
    test_pa(pa.array([pa.scalar(date(2012, 1, 1), type=pa.date32())]), kx.q('2012.01.01'),
            kx.q('2012.01.01'))
    test_pa(pa.array([pa.scalar(date(2012, 1, 1), type=pa.date32())]*2),
            kx.q('2#(),2012.01.01'), kx.q('2012.01.01'))

    # pyarrow.date64
    test_pa(pa.array([pa.scalar(date(2012, 1, 1), type=pa.date64())]), kx.q('2012.01.01'),
            kx.q('2012.01.01'))
    test_pa(pa.array([pa.scalar(date(2012, 1, 1), type=pa.date64())]*2),
            kx.q('2#(),2012.01.01'), kx.q('2012.01.01'))

    # pyarrow.duration
    test_pa(pa.array([1], pa.duration('ns')), kx.q('0D00:00:00.000000001'),
            kx.q('0D00:00:00.000000001'))
    test_pa(pa.array([1, 2], pa.duration('ns')),
            kx.q('(),0D00:00:00.000000001 0D00:00:00.000000002'), kx.q('0D00:00:00.000000001'))

    test_pa(pa.array([1], pa.duration('ms')), kx.q('0D00:00:00.001000000'),
            kx.q('0D00:00:00.001000000'))
    if kx.config.pandas_2:
        test_pa(pa.array([1, 2], pa.duration('ms')), kx.q('00:00:00.001 00:00:00.002'),
                kx.q('0D00:00:00.001000000'))
    else:
        test_pa(pa.array([1, 2], pa.duration('ms')), kx.q('0D00:00:00.001 0D00:00:00.002'),
                kx.q('0D00:00:00.001000000'))

    test_pa(pa.array([1], pa.duration('us')), kx.q('0D00:00:00.000001000'),
            kx.q('0D00:00:00.000001000'))
    test_pa(pa.array([1, 2], pa.duration('us')),
            kx.q('0D00:00:00.000001000 0D00:00:00.000002000'), kx.q('0D00:00:00.000001000'))

    test_pa(pa.array([1], pa.duration('s')), kx.q('0D00:00:01.000000000'),
            kx.q('0D00:00:01.000000000'))
    if kx.config.pandas_2:
        test_pa(pa.array([1, 2], pa.duration('s')), kx.q('00:00:01 00:00:02'),
                kx.q('0D00:00:01.000000000'))
    else:
        test_pa(pa.array([1, 2], pa.duration('s')), kx.q('0D00:00:01 0D00:00:02'),
                kx.q('0D00:00:01.000000000'))

    # pyarrow.binary
    test_pa(pa.array(['foo'], type=pa.binary()), kx.q('"foo"'),
            kx.q('"foo"'))
    test_pa(pa.array(['foo', 'bar', 'baz'], type=pa.binary()), kx.q('("foo";"bar";"baz")'),
            kx.q('"foo"'))

    # pyarrow.string
    test_pa(pa.array(['foo'], type=pa.string()), kx.q('`foo'),
            kx.q('`foo'))
    test_pa(pa.array(['foo', 'bar', 'baz'], type=pa.string()), kx.q('`foo`bar`baz'),
            kx.q('`foo'))

    # pyarrow.utf8
    test_pa(pa.array(['foo'], type=pa.utf8()), kx.q('`foo'),
            kx.q('`foo'))
    test_pa(pa.array(['foo', 'bar', 'baz'], type=pa.utf8()), kx.q('`foo`bar`baz'),
            kx.q('`foo'))

    # pyarrow.large_binary
    test_pa(pa.array(['foo'], type=pa.large_binary()), kx.q('"foo"'),
            kx.q('"foo"'))
    test_pa(pa.array(['foo', 'bar', 'baz'], type=pa.large_binary()),
            kx.q('("foo";"bar";"baz")'), kx.q('"foo"'))

    # pyarrow.large_string
    test_pa(pa.array(['foo'], type=pa.large_string()), kx.q('`foo'), kx.q('`foo'))
    test_pa(pa.array(['foo', 'bar'], type=pa.large_string()), kx.q('`foo`bar'), kx.q('`foo'))

    # pyarrow.large_utf8
    test_pa(pa.array(['foo'], type=pa.large_utf8()), kx.q('`foo'), kx.q('`foo'))
    test_pa(pa.array(['foo', 'bar'], type=pa.large_utf8()), kx.q('`foo`bar'), kx.q('`foo'))


def test_pandas_timedelta(kx):
    if kx.config.pandas_2:
        assert kx.toq(kx.q('16:36').pd()) == kx.q('16:36:00')
        assert kx.toq(kx.q('16:36:29').pd()) == kx.q('16:36:29')
        assert kx.toq(kx.q('16:36:29.214').pd()) == kx.q('16:36:29.214')
        assert kx.toq(kx.q('16:36:29.214344').pd()) == kx.q('0D16:36:29.214344000')
        assert kx.toq(kx.q('16:36:29.214344678').pd()) == kx.q('0D16:36:29.214344678')
    else:
        assert kx.toq(kx.q('16:36').pd()) == kx.q('0D16:36:00.000000000')
        assert kx.toq(kx.q('16:36:29').pd()) == kx.q('0D16:36:29.000000000')
        assert kx.toq(kx.q('16:36:29.214').pd()) == kx.q('0D16:36:29.214000000')
        assert kx.toq(kx.q('16:36:29.214344').pd()) == kx.q('0D16:36:29.214344000')
        assert kx.toq(kx.q('16:36:29.214344678').pd()) == kx.q('0D16:36:29.214344678')


def test_cast_setting(kx):
    with pytest.raises(TypeError) as err:
        kx.IntAtom(9, cast="Test")
        assert "must be of type" in str(err)

    testVal = "9"
    assert kx.IntAtom(9) == kx.IntAtom(testVal, cast=True)
    with pytest.raises(TypeError):
        kx.IntAtom(testVal, cast=False)

    testVal = "9.3"
    assert kx.FloatAtom(9.3) == kx.FloatAtom(testVal, cast=True)
    with pytest.raises(TypeError):
        kx.FloatAtom(testVal, cast=False)

    assert kx.DateAtom(np.datetime64('2024-11-06T14:38:03')) == date(2024, 11, 6)
    with pytest.raises(TypeError):
        kx.DateAtom(np.datetime64('2024-11-06T14:38:03'), cast=False)

    assert kx.TimespanAtom(np.timedelta64(1, 'D'), cast=True) == timedelta(days=1)
    with pytest.raises(TypeError):
        kx.DateAtom(np.timedelta64(1, 'D'), cast=False)

    assert kx.TimeAtom(np.timedelta64(1, 'h'), cast=True) == time(1, 0, 0)
    with pytest.raises(AttributeError):
        kx.TimeAtom(np.timedelta64(1, 'h'), cast=False)

    assert kx.TimestampAtom(np.datetime64('2024-11-06T16:43:50'), cast=True) == datetime(2024, 11, 6, 16, 43, 50) # noqa: E501
    with pytest.raises(Exception) as e:
        kx.TimestampAtom(np.datetime64('2024-11-06T16:43:50'), cast=False)
        assert "numpy" in str(e)

    testVal = [b'a', b'ab', 1.3, [1.3, 1.2], 'x']
    assert (kx.K(np.array(testVal, dtype=object), cast=False) == [b'a', b'ab', 1.3, [1.3, 1.2], 'x']).all() # noqa: E501
    with pytest.raises(KeyError):
        kx.K(np.array(testVal, dtype=object), cast=True)

    nparray = np.array([1, 2, 3])
    assert (kx.toq(nparray, kx.FloatVector, cast=True) == kx.FloatVector(kx.q('1 2 3f'))).all()


def test_2d_array_from_file(kx):
    df = pd.read_parquet('tests/nestedFloats.parquet')
    kx.toq(df, no_allocator=True)


def test_2d_array_from_file_no_allocator(kx):
    df = pd.read_parquet('tests/nestedFloats.parquet')
    kx.toq(df, no_allocator=True)


def test_embedding_segfault(kx):
    df=pd.DataFrame(dict(embeddings=list(np.random.ranf((500, 10)).astype(np.float32))))
    kx.toq(df)

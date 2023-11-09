"""Tests for the types defined in the wrappers module."""

from collections import abc
from datetime import date
from datetime import datetime
from datetime import timedelta
import gc
import math
from operator import index
import pickle
from platform import python_implementation
from textwrap import dedent
from uuid import UUID

# Do not import Pandas, PyArrow, or PyKX here - use the pd/pa/kx fixtures instead!
import numpy as np
import pytest
import pytz
from packaging import version


pypy = python_implementation() == 'PyPy'


def test_abc_compatibility(kx):
    assert issubclass(kx.Vector, abc.Sequence)
    assert issubclass(kx.Vector, abc.Collection)
    assert issubclass(kx.Vector, abc.Reversible)
    assert hasattr(kx.Vector, 'index')
    assert hasattr(kx.Vector, 'count')

    assert issubclass(kx.Mapping, abc.Mapping)
    assert issubclass(kx.Mapping, abc.Collection)
    assert hasattr(kx.Mapping, 'keys')
    assert hasattr(kx.Mapping, 'values')
    assert hasattr(kx.Mapping, 'items')
    assert hasattr(kx.Mapping, 'get')
    assert hasattr(kx.Mapping, '__eq__')
    assert hasattr(kx.Mapping, '__ne__')


def test_eval_repr(kx):
    pykx = kx
    for x in (pykx.q('`a`b`c'), pykx.q('97'), pykx.q('"this is a char vector"')):
        assert pykx.q('~', x, eval(repr(x)))


@pytest.mark.unlicensed(unlicensed_only=True)
def test_eval_repr_unlicensed(kx):
    pykx = kx
    cases = (
        pykx.SymbolVector(['a', 'b', 'c']),
        pykx.LongAtom(97),
        pykx.CharVector('this is a char vector'),
    )
    for x in cases:
        assert x.py() == eval(repr(x)).py()


@pytest.mark.embedded
def test_pykx_q_get(kx, q):
    with pytest.raises(kx.QError):
        q('.pykx.get`blorp')


@pytest.mark.embedded
def test_pykx_q_getattr(kx, q):
    q("af:.pykx.eval\"type('Car', (object,), {'speed': 200, 'color': 'red'})\"")
    q('af[`:speed]`')
    with pytest.raises(kx.QError):
        q('af[`:boourns]`')

    q('arr:.pykx.eval"[1, 2, 3]"')
    with pytest.raises(kx.QError):
        q('.pykx.getattr[.pykx.unwrap arr;`foobarbaz]')

# TODO: Once PYKX_RELEASE_GIL is fixed this should be uncommented
# @pytest.mark.embedded
# def test_no_gil_allows_peach(kx, q):
#     peach_res = q('{[f] f peach x:10+til 2*1|system"s"}', lambda x: range(x)).py()
#     assert peach_res == [list(range(x + 10)) for x in range(2 * kx.util.num_available_cores())]


class Test_K:
    @pytest.mark.skipif(pypy, reason='PyPy uses gc logic that is difficult to reliably test')
    def test_r0(self, q):
        n = 100000000
        q.Q.gc()
        before_alloc = q.Q.w()['used']
        large_vector = q.til(n)
        after_alloc = q.Q.w()['used']
        # Ensure 8n bytes have been allocated (n.b. long ints take 8 bytes each)
        assert after_alloc - before_alloc > 8 * n
        assert large_vector._k.r == 0
        del large_vector
        gc.collect()
        q.Q.gc()
        after_dealloc = q.Q.w()['used']
        # The memory in use after deallocation should be within 10% of the amount in use before
        assert after_dealloc < 1.1 * before_alloc

    def test_refcounting_q_vars(self, kx, q):
        q('a:til 10')
        assert q('-16!a') == kx._wrappers.k_r(q('a'))== 1
        # Check incrementing with new reference
        q('b:a')
        assert q('-16!a') == kx._wrappers.k_r(q('a'))== 2
        # Remove global referencing 'a'
        q('delete b from `.')
        assert q('-16!a') == kx._wrappers.k_r(q('a'))== 1
        # Check incrementing with new reference in table
        q('c:([]a;10?1f)')
        assert q('-16!a') == kx._wrappers.k_r(q('a'))== 2
        # Remove reference to 'a' in table
        q('![`c;();0b;enlist`a]')
        assert q('-16!a') == kx._wrappers.k_r(q('a'))== 1

    def test_repr(self, q, kx):
        q.system.console_size = [25, 80]
        pykx = kx # noqa: F401
        rand_shorts = q('5?0Wh')
        assert all(rand_shorts == eval(repr(rand_shorts), globals(), locals()))
        assert "pykx.SymbolAtom(pykx.q('`reprtest'))" == repr(q('`reprtest'))
        assert "pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 ..'))" == repr(q('til 1000')) # noqa
        s = dedent("""
            pykx.Table(pykx.q('
            alpha beta
            ----------
            0     a   
            1     b   
            2     c   
            3     d   
            4     e   
            5     f   
            6     g   
            7     h   
            8     i   
            9     j   
            10    k   
            11    l   
            12    m   
            13    n   
            14    o   
            15    p   
            16    q   
            17    r   
            18    s   
            19    t   
            ..
            '))""")[1:].replace('\r\n', '\n') # noqa
        assert s == repr(q('([] alpha:til 100;beta:100#.Q.a)')).replace('\r\n', '\n')

    def test_str(self, q):
        assert str(q('enlist(::)')) == '::'
        assert str(q('til 4')) == '0 1 2 3'
        assert str(q('enlist each til 4')).replace('\r\n', '\n') == '0\n1\n2\n3'
        assert str(q('::')) == '::'
        assert str(q('()')) == ''

    @pytest.mark.unlicensed(unlicensed_only=True)
    def test_repr_str_unlicensed(self, kx):
        x = kx.K([0, None, '', float('NaN')])
        assert str(x) == repr(x)
        assert str(x).startswith('pykx.List._from_addr(0x')

    @pytest.mark.ipc
    def test_pickling(self, kx, q):
        mkt = q('([k1:`a`b`a;k2:100+til 3] x:til 3; y:`multi`keyed`table)')
        pickled_mkt = pickle.dumps(mkt)
        unpickled_mkt = pickle.loads(pickled_mkt)
        assert mkt._addr != unpickled_mkt._addr
        assert q('~', mkt, unpickled_mkt)
        with pytest.raises(TypeError) as err:
            pickle.dumps(kx.Foreign(10))
        assert 'Foreign' in str(err.value)

    @pytest.mark.ipc
    def test_is_atom(self, kx, q):
        assert q('til 2').is_atom is False
        assert q('2').is_atom
        if kx.licensed:
            assert q('{x*y}').is_atom # despite the positive type number

    @pytest.mark.ipc
    def test_t_attr(self, q):
        assert q('til 2').t == 7
        assert q('12407').t == -7
        assert q('::').t == 101
        assert q('(0b;1)').t == 0
        assert q('"abc"!`a`b`c').t == 99

    @pytest.mark.ipc
    def test_t_property(self, kx, q):
        x = q('2')
        x.__class__ = kx.wrappers.K # Force fallback to t property
        assert x.t == -7

    @pytest.mark.ipc
    def test_hash(self, kx, q):
        queries = ['`abc', '0N', '1', '0', '-1', '2*', '0Wp']
        if kx.licensed:
            queries.append('{x+y*z}')
        for query in queries:
            x = q(query)
            assert hash(x) == hash(x) == hash(q(query))
        assert len({q(query) for query in queries}) == len(queries)
        for query in ('til 8', '()!()', '([]"abc")'):
            with pytest.raises(TypeError):
                hash(q(query))

    @pytest.mark.nep49
    def test_richest_interface_applied(self, q, kx, pd):
        x = q('(([] a:til 3; b:2+til 3; c:`a`b`c); til 32; first "p"$1?0Wi; {x*x};([a:til 3; b:2+til 3]; c:`a`b`c; d:(`x;(til 6; "abcd");`z)))') # noqa
        types = (pd.DataFrame, np.ndarray, np.datetime64, kx.Lambda, pd.DataFrame)
        for a, b in zip(x.py(stdlib=False), types):
            assert isinstance(a, b)
        for a, b in zip(x[-1].values().values().py(stdlib=False), (np.ndarray, np.ndarray)):
            assert isinstance(a, b)

    def test_reference_constructor(self, q, kx):
        assert kx.LongAtom(q('196883'))
        with pytest.raises(TypeError):
            kx.Table(q('196883'))

    def test_equality(self, q):
        assert q('{}') == q('{}')
        assert q('{}') != q('{x}')
        assert (q('(::;1 2 3)') == (None, (1, 2, 3))).all()
        assert not (q('(::;1 2 3)') != (None, (1, 8, 3))).all()
        assert (q('(::;1 2 3)') != (None, (1, 8, 3))).any()
        assert not q('(::;1 2 3)') == object()
        assert q('(::;1 2 3)') != object()
        assert q('5') != None # noqa: E711
        assert not q('5') == None # noqa: E711


class Test_Atom:
    def test_char_atom(self, kx, q):
        atom = kx.CharAtom('a')
        assert 1 == len(atom)
        assert atom[0].py() == b'a'
        with pytest.raises(IndexError):
            atom[1]
        with pytest.raises(IndexError):
            atom[-1]

    def test_boolean_atom(self, q):
        t, f = q('1b'), q('0b')
        assert t == True    # noqa
        assert f == False   # noqa
        assert t.py() is True
        assert f.py() is False
        assert t.py(raw=True) is not True
        assert f.py(raw=True) is not False
        assert t.py(raw=True) == 1
        assert f.py(raw=True) == 0
        assert bool(t) is True
        assert bool(f) is False

    def test_null_gen_lic(self, kx):
        qtypes = [kx.GUIDAtom, kx.ShortAtom, kx.IntAtom,
                  kx.LongAtom, kx.RealAtom, kx.FloatAtom,
                  kx.CharAtom, kx.SymbolAtom, kx.TimestampAtom,
                  kx.MonthAtom, kx.DateAtom, kx.DatetimeAtom,
                  kx.TimespanAtom, kx.MinuteAtom, kx.SecondAtom,
                  kx.TimeAtom]
        for i in qtypes:
            null_val = getattr(i, 'null') # noqa: B009
            assert type(null_val) == i
            assert null_val.is_null

    def test_inf_pos_lic(self, kx):
        qtypes = [kx.ShortAtom, kx.IntAtom,
                  kx.LongAtom, kx.RealAtom, kx.FloatAtom,
                  kx.TimestampAtom, kx.MonthAtom, kx.DateAtom,
                  kx.DatetimeAtom, kx.TimespanAtom, kx.MinuteAtom,
                  kx.SecondAtom, kx.TimeAtom]
        for i in qtypes:
            inf_val = getattr(i, 'inf') # noqa: B009
            assert type(inf_val) == i
            assert inf_val>0
            assert inf_val.is_inf

    def test_inf_neg_lic(self, kx):
        qtypes = [kx.ShortAtom, kx.IntAtom,
                  kx.LongAtom, kx.RealAtom, kx.FloatAtom,
                  kx.TimestampAtom, kx.MonthAtom, kx.DateAtom,
                  kx.DatetimeAtom, kx.TimespanAtom, kx.MinuteAtom,
                  kx.SecondAtom, kx.TimeAtom]
        for i in qtypes:
            inf_val = -getattr(i, 'inf') # noqa: B009
            assert type(inf_val) == i
            assert inf_val<0
            assert inf_val.is_inf

    def test_null_fail(self, kx):
        qtypes = [kx.BooleanAtom, kx.ByteAtom]
        for i in qtypes:
            with pytest.raises(NotImplementedError) as err:
                getattr(i, 'null') # noqa: B009
            assert 'Retrieval of null values' in str(err)

    def test_inf_fail(self, kx):
        qtypes = [kx.BooleanAtom, kx.ByteAtom, kx.GUIDAtom,
                  kx.CharAtom, kx.SymbolAtom]
        for i in qtypes:
            with pytest.raises(NotImplementedError) as err:
                getattr(i, 'inf') # noqa: B009
            assert 'Retrieval of infinite values' in str(err)

    @pytest.mark.unlicensed(unlicensed_only=True)
    def test_null_inf_unlic(self, kx):
        qtypes = [kx.ByteAtom, kx.GUIDAtom, kx.ShortAtom,
                  kx.IntAtom, kx.LongAtom, kx.RealAtom,
                  kx.FloatAtom, kx.CharAtom, kx.SymbolAtom,
                  kx.TimestampAtom, kx.MonthAtom, kx.DateAtom,
                  kx.DatetimeAtom, kx.TimespanAtom, kx.MinuteAtom,
                  kx.SecondAtom, kx.TimeAtom]
        for i in qtypes:
            for j in ['null', 'inf']:
                with pytest.raises(kx.QError) as err:
                    getattr(i, j)()
                assert 'not supported in unlicensed mode' in str(err)

    def test_is_null_and_is_inf(self, q):
        assert q('0Ng').is_null
        assert not q('first 1?0Ng').is_null

        assert q('0Nh').is_null
        assert not q('first 1?0h').is_null
        assert q('0Wh').is_inf
        assert q('-0Wh').is_inf
        assert not q('first 1?0h').is_inf

        assert q('0Ni').is_null
        assert not q('first 1?0i').is_null
        assert q('0Wi').is_inf
        assert q('-0Wi').is_inf
        assert not q('first 1?0i').is_inf

        assert q('0Nj').is_null
        assert not q('first 1?0j').is_null
        assert q('0Wj').is_inf
        assert q('-0Wj').is_inf
        assert not q('first 1?0j').is_inf

        assert q('0Ne').is_null
        assert not q('first 1?1e').is_null
        assert q('0We').is_inf
        assert q('-0We').is_inf
        assert not q('first 1?1e').is_inf

        assert q('0Nf').is_null
        assert not q('first 1?1f').is_null
        assert q('0Wf').is_inf
        assert q('-0Wf').is_inf
        assert not q('first 1?1f').is_inf

        assert q('0Np').is_null
        assert not q('first 1?1f').is_null
        assert q('0Wp').is_inf
        assert q('-0Wp').is_inf
        assert not q('first 1?0p').is_inf

        assert q('0Nm').is_null
        assert not q('first 1?2000.01m').is_null
        assert q('0Wm').is_inf
        assert q('-0Wm').is_inf
        assert not q('first 1?2000.01m').is_inf

        assert q('0Nd').is_null
        assert not q('first 1?2000.01.01').is_null
        assert q('0Wd').is_inf
        assert q('-0Wd').is_inf
        assert not q('first 1?2000.01.01').is_inf

        assert q('0Nn').is_null
        assert not q('first "n"$1?0').is_null
        assert q('0Wn').is_inf
        assert q('-0Wn').is_inf
        assert not q('first "n"$1?0').is_inf

        assert q('0Nu').is_null
        assert not q('first 1?0u').is_null
        assert q('0Wu').is_inf
        assert q('-0wu').is_inf
        assert not q('first 1?0u').is_inf

        assert q('0Nv').is_null
        assert not q('first 1?0v').is_null
        assert q('0Wv').is_inf
        assert q('-0Wv').is_inf
        assert not q('first 1?0v').is_inf

        assert q('0Nt').is_null
        assert not q('first 1?0t').is_null
        assert q('0Wt').is_inf
        assert q('-0Wt').is_inf
        assert not q('first 1?0t').is_inf

        assert not q('{x*y+z}').is_null
        assert not q('{x*y+z}').is_inf

    @pytest.mark.nep49
    def test_null_np(self, q, kx):
        for type_char in 'hij':
            with pytest.raises(kx.PyKXException):
                q(f'0N{type_char}').np()

        for type_char in 'ef':
            assert np.isnan(q(f'0N{type_char}').np())

        for type_char in 'pmdnuvt':
            assert np.isnat(q(f'0N{type_char}').np())

    @pytest.mark.nep49
    def test_null_pd(self, q, kx, pd):
        for type_char in 'hij':
            with pytest.raises(kx.PyKXException):
                q(f'0N{type_char}').pd()

        for type_char in 'ef':
            assert pd.isna(q(f'0N{type_char}').pd())

        for type_char in 'pmdnuvt':
            assert isinstance(q(f'0N{type_char}').pd(), type(pd.NaT))

    def test_int_ops(self, kx):
        """Test arithmetic operations between a Atom and a Python int."""
        x = 12
        y = 19
        a = kx.K(x)
        b = kx.K(y)

        assert x + 2 == a + 2
        assert x + y == a + b
        assert 2 + x == 2 + a
        assert y + x == b + a

        assert x - 2 == a - 2
        assert x - y == a - b
        assert 2 - x == 2 - a
        assert y - x == b - a

        assert x * 3 == a * 3
        assert x * y == a * b
        assert 3 * x == 3 * a
        assert y * x == b * a

        assert x / 7 == a / 7
        assert x / y == a / b
        assert 7 / x == 7 / x
        assert y / x == b / x

        assert x / 7.5 == a / 7.5
        assert 7.5 / x == 7.5 / a

        assert x // 7 == a // 7
        assert x // y == a // b
        assert 17 // x == 17 // a
        assert y // x == b // a

        assert x % 7 == a % 7
        assert x % y == a % b
        assert 17 % x == 17 % a
        assert y % x == b % a

        assert divmod(x, 7) == divmod(a, 7)
        assert divmod(x, y) == divmod(a, b)
        assert divmod(17, x) == divmod(17, a)
        assert divmod(y, x) == divmod(b, a)

        # 10 is subtracted from b/y to avoid an overflow
        assert x ** 2 == a ** 2
        assert x ** (y - 10) == a ** (b - 10)
        assert 2 ** x == 2 ** a
        assert (y - 10) ** x == (b - 10) ** a
        assert pow(x, 2, 7) == pow(a, 2, 7)
        assert pow(x, y, 7) == pow(a, b, 7)
        assert pow(y, x, 7) == pow(b, a, 7)
        assert pow(y, 4, x) == pow(b, 4, a)

        # Cython allows for this, but CPython doesn't support it yet
        if python_implementation() != 'CPython':
            assert pow(2, x, 7) == pow(2, a, 7)

        # Cython allows for this, but neither CPython nor PyPy support it yet
        # assert pow(2, 4, x) == pow(2, 4, a)

        assert -x == -a
        assert +x == +a

        assert abs(x) == abs(a) == abs(-a)

        assert bool(x) == bool(a)

        assert ~x == ~a

        assert x << 2 == a << 2
        assert x << y == a << b
        assert 2 << x == 2 << a
        assert y << x == b << a

        assert x >> 2 == a >> 2
        assert x >> y == a >> b
        assert 2 >> x == 2 >> a
        assert y >> x == b >> a

        assert x & 4 == a & 4
        assert x & y == a & b
        assert 4 & x == 4 & a
        assert y & x == b & a

        assert x | 7 == a | 7
        assert x | y == a | b
        assert 7 | x == 7 | a
        assert y | x == b | a

        assert x ^ 7 == a ^ 7
        assert x ^ y == a ^ b
        assert x ^ 7 == 7 ^ a
        assert x ^ y == b ^ a

        assert (x < 12) == (a < 12)
        assert (x < y) == (a < b)
        assert (x == 12) == (a == 12)
        assert (x == y) == (a == b)
        assert (x > 12) == (a > 12)
        assert (x > y) == (a > b)
        assert (x <= 12) == (a <= 12)
        assert (x <= y) == (a <= b)
        assert (x != 12) == (a != 12)
        assert (x != y) == (a != b)
        assert (x >= 12) == (a >= 12)
        assert (x >= y) == (a >= b)

    def test_py(self, q, kx):
        assert q('0b').py() is False
        assert q('1b').py()
        assert q('1b').t == -1

        assert q('0xFF').py() == 2 ** 8 - 1
        assert q('0xFF').t == -4

        assert(q('0Nh').py() == kx.ShortAtom(q('0Nh')))
        try:
            q('0Wh').py()
        except kx.PyKXException:
            pass
        assert q('0Wh').t == -5

        assert (q('0Ni').py() == kx.IntAtom(q('0Ni')))
        try:
            q('0Wi').py()
        except kx.PyKXException:
            pass
        assert q('0Wi').t == -6

        assert (q('0N').py() == kx.LongAtom(q('0N')))
        try:
            q('0Wj').py()
        except kx.PyKXException:
            pass
        assert q('0Wj').t == -7

        assert math.isnan(q('0Ne').py())
        assert q('0We').py() == float('inf')
        assert q('0We').t == -8

        assert math.isnan(q('0Nf').py())
        assert q('0Wf').py() == float('inf')
        assert q('0Wf').t == -9

        assert q('"a"').py() == b'a'
        assert q('"a"').py(raw=True) == ord('a')
        assert q('"a"').t == -10

    @pytest.mark.nep49
    def test_np(self, q, kx):
        assert q('0b').np() is False
        assert q('1b').np()
        assert q('1b').t == -1

        assert isinstance(q('0xFF').np(), np.uint8)
        assert q('0xFF').np() == 2 ** 8 - 1
        assert q('0xFF').t == -4

        try:
            q('0Wh').np()
        except kx.PyKXException:
            pass
        assert q('0Wh').t == -5

        try:
            q('0Wi').np()
        except kx.PyKXException:
            pass
        assert q('0Wi').t == -6

        try:
            q('0Wj').np()
        except kx.PyKXException:
            pass
        assert q('0Wj').t == -7

        assert isinstance(q('0We').np(), np.float32)
        assert q('0We').np() == float('inf')
        assert q('0We').t == -8

        assert isinstance(q('0Wf').np(), np.float64)
        assert q('0Wf').np() == float('inf')
        assert q('0Wf').t == -9

        assert q('"a"').np() == b'a'
        assert q('"a"').np(raw=True) == 97
        assert q('"a"').t == -10

    def test_numeric_types(self, q, kx):
        assert isinstance(q('1.2f'), kx.NumericAtom)
        assert isinstance(q('1.2e'), kx.NumericAtom)
        assert isinstance(q('123j'), kx.NumericAtom)
        assert isinstance(q('123i'), kx.NumericAtom)
        assert isinstance(q('123h'), kx.NumericAtom)
        assert isinstance(q('0x7b'), kx.NumericAtom)
        assert isinstance(q('1b  '), kx.NumericAtom)
        assert isinstance(q('123j'), kx.IntegralNumericAtom)
        assert isinstance(q('123i'), kx.IntegralNumericAtom)
        assert isinstance(q('123h'), kx.IntegralNumericAtom)
        assert isinstance(q('0x7b'), kx.IntegralNumericAtom)
        assert isinstance(q('1b  '), kx.IntegralNumericAtom)

    @pytest.mark.nep49
    def test_pd(self, q, pd):
        assert q('0b').pd() is False
        assert q('1b').pd() is True

        assert (q('123').pd() == pd.Series([123])).all()
        assert (q('123i').pd() == pd.Series([123], dtype="int32")).all()
        assert (q('123h').pd() == pd.Series([123], dtype="int16")).all()
        assert (q('1.5e').pd() == pd.Series([1.5], dtype="float32")).all()
        assert (q('1.5').pd() == pd.Series([1.5])).all()

        assert q('"a"').pd() == b'a'
        assert q('"a"').pd(raw=True) == 97

    @pytest.mark.nep49
    def test_pa(self, q, pa):
        assert q('0b').pa() is False
        assert q('1b').pa()

        assert q('123').pa() == pa.array([123])
        assert q('123i').pa() == pa.array([123], type="int32")
        assert q('123h').pa() == pa.array([123], type="int16")
        assert q('1.5e').pa() == pa.array([1.5], type="float32")
        assert q('1.5').pa() == pa.array([1.5])

        assert q('"a"').pa() == b'a'
        assert q('"a"').pa(raw=True) == 97

    def test_numeric_methods(self, kx):
        k_numeric_symbol = kx.K('1729')
        k_long = kx.K(22)
        k_float = kx.K(math.e)
        k_neg_float = kx.K(-math.pi)
        assert int(k_long) == 22
        assert int(k_numeric_symbol) == 1729
        assert int(k_float) == math.trunc(k_float) == 2
        assert float(k_long) == 22.0
        assert float(k_numeric_symbol) == 1729.0
        assert complex(k_float) == complex(math.e)
        assert complex(k_numeric_symbol) == complex(1729)
        assert round(k_float, ndigits=2) == 2.72
        assert round(k_long, ndigits=-1) == 20
        assert math.floor(k_float) == 2
        assert math.floor(k_neg_float) == -4
        assert math.floor(k_long) == 22
        assert math.ceil(k_float) == 3
        assert math.ceil(k_long) == 22
        assert math.trunc(k_long) == 22
        assert math.trunc(k_float) == 2

    def test_integral_methods(self, kx):
        k_long = kx.K(828)
        assert index(k_long) == 828

    def test_real_numeric_methods(self, kx):
        x = 5.1
        y = 9.3
        a = kx.K(x)
        b = kx.K(y)

        assert x ** 2 == a ** 2
        assert x ** y == a ** b
        assert 2 ** x == 2 ** a
        assert y ** x == b ** a

    @pytest.mark.filterwarnings('ignore:The q datetime type is deprecated')
    def test_bool(self, q):
        assert not q('0b')
        assert not q('0x00')
        assert not q('0h')
        assert not q('0i')
        assert not q('0j')
        assert not q('0e')
        assert not q('0f')
        assert not q('" "')
        assert not q('`')
        assert not q('0p')
        assert not q('2000.01m')
        assert not q('2000.01.01')
        assert not q('00:00:00.000000000')
        assert not q('00:00')
        assert not q('00:00:00')
        assert not q('00:00:00.000')

        assert q('1b')
        assert q('0x01')
        assert q('1h')
        assert q('1i')
        assert q('1j')
        assert q('1e')
        assert q('1f')
        assert q('"a"')
        assert q('`a')
        assert q('1p')
        assert q('2000.02m')
        assert q('2000.01.02')
        assert q('00:00:00.000000001')
        assert q('00:01')
        assert q('00:00:01')
        assert q('00:00:00.001')

        # Nulls and infinities:
        assert not q('0Ng')
        for type_char in 'hijefpmdznuvt':
            assert not q(f'0N{type_char}')
            assert q(f'0W{type_char}')


class Test_EnumAtom:
    def test_py(self, q):
        # pytest.set_trace()
        e = q('u:`abc`xyz`hmm;`u$`xyz')
        assert e.py() == 'xyz'
        assert e.py(raw=True) == 1


class Test_TemporalSpanAtom:
    @pytest.mark.nep49
    def test_pd(self, q, pd):
        a = q('41927D02:22:17.297584128')
        b = q('55200D12:09:33.595746304')
        c = q('21:02')
        d = q('40:09')
        e = q('00:48:18')
        f = q('01:46:00')
        g = q('16:36:29.214')
        h = q('08:31:52.958')

        assert a.pd() == pd.Timedelta('41927D02:22:17.297584128')
        assert b.pd() == pd.Timedelta('55200D12:09:33.595746304')
        assert c.pd() == pd.Timedelta('21:02:00')
        assert d.pd() == pd.Timedelta('40:09:00')
        assert e.pd() == pd.Timedelta('00:48:18')
        assert f.pd() == pd.Timedelta('01:46:00')
        assert g.pd() == pd.Timedelta('16:36:29.214')
        assert h.pd() == pd.Timedelta('08:31:52.958')

        assert a.pd(raw=True) == 3622501337297584128
        assert b.pd(raw=True) == 4769323773595746304
        assert c.pd(raw=True) == 1262
        assert d.pd(raw=True) == 2409
        assert e.pd(raw=True) == 2898
        assert f.pd(raw=True) == 6360
        assert g.pd(raw=True) == 59789214
        assert h.pd(raw=True) == 30712958

    def test_null_py_conversion(self, q, pd):
        for query in ('0Nv', '0Nu', '0Nt', '0Nn'):
            x = q(query).py()
            assert pd.isnull(x)
            assert isinstance(x, type(pd.NaT))


class Test_TemporalFixedAtom:
    @pytest.mark.nep49
    def test_pd(self, q, pd):
        a = q('2133.04.29D23:12:27.985231872')
        b = q('1761.01.20D15:06:00.175740928')
        c = q('2000.01m')
        d = q('2000.02m')
        e = q('2000.01.01')
        f = q('2000.01.02')

        assert a.pd() == pd.Timestamp('2133.04.29T23:12:27.985231872')
        assert b.pd() == pd.Timestamp('1761.01.20T15:06:00.175740928')
        assert c.pd() == pd.Timestamp('2000.01')
        assert d.pd() == pd.Timestamp('2000.02')
        assert e.pd() == pd.Timestamp('2000.01.01')
        assert f.pd() == pd.Timestamp('2000.01.02')

        assert a.pd(raw=True) == 4207417947985231872
        assert b.pd(raw=True) == -7540332839824259072
        assert c.pd(raw=True) == 0
        assert d.pd(raw=True) == 1
        assert e.pd(raw=True) == 0
        assert f.pd(raw=True) == 1


class Test_TemporalAtom:
    @pytest.mark.nep49
    def test_timestamp(self, q, kx):
        timestamp = q('2150.10.22D20:31:15.070713856')
        assert isinstance(timestamp, kx.TimestampAtom)
        assert timestamp.np() == np.datetime64(
            '2150-10-22T20:31:15.070713856', 'ns')
        assert timestamp.py() == datetime(2150, 10, 22, 20, 31, 15, 70713)
        assert timestamp.np(raw=True) == 4759072275070713856
        assert timestamp.py(raw=True) == 4759072275070713856

    @pytest.mark.nep49
    def test_timestamp_timezone(self, kx):
        kx.config._set_keep_local_times(False)
        la = pytz.timezone("America/Los_Angeles")
        tz = datetime(2022, 7, 20, 10, 0, 0, 0, la)
        assert not tz == kx.toq(tz).py()
        assert tz == kx.toq(tz).py(tzinfo=pytz.UTC)
        assert tz == kx.toq(tz).py(tzinfo=la)
        kx.config._set_keep_local_times(True)
        assert datetime(2022, 7, 20, 10, 0, 0, 0) == kx.toq(tz).py()
        assert tz == kx.toq(tz).py(tzinfo=la, tzshift=False)

    @pytest.mark.nep49
    def test_month(self, q, kx):
        month = q('1972.05m')
        assert isinstance(month, kx.MonthAtom)
        assert month.np() == np.datetime64('1972-05', 'M')
        assert month.py() == date(1972, 5, 1)
        assert month.np(raw=True) == -332
        assert month.py(raw=True) == -332

    @pytest.mark.nep49
    def test_date(self, q, kx):
        q_date = q('1972.05.31')
        assert isinstance(q_date, kx.DateAtom)
        assert q_date.np() == np.datetime64('1972-05-31', 'D')
        assert q_date.py() == date(1972, 5, 31)
        assert q_date.np(raw=True) == -10076
        assert q_date.py(raw=True) == -10076

    @pytest.mark.nep49
    def test_datetime(self, q, kx):
        with pytest.warns(DeprecationWarning):
            datetime = q('0001.02.03T04:05:06.007')
        assert isinstance(datetime, kx.DatetimeAtom)
        with pytest.raises(TypeError):
            datetime.np()
        with pytest.raises(TypeError):
            datetime.py()
        assert datetime.np(raw=True) == -730085.8297915857
        assert datetime.py(raw=True) == -730085.8297915857

    @pytest.mark.nep49
    def test_timespan(self, q, kx):
        timespan = q('43938D19:07:31.664551936')
        assert isinstance(timespan, kx.TimespanAtom)
        assert timespan.np() == np.timedelta64(3796312051664551936, 'ns')
        assert timespan.py() == timedelta(
            days=43938, seconds=68851, microseconds=664551)
        assert timespan.np(raw=True) == 3796312051664551936
        assert timespan.py(raw=True) == 3796312051664551936

    @pytest.mark.nep49
    def test_minute(self, q, kx):
        minute = q('03:36')
        assert isinstance(minute, kx.MinuteAtom)
        assert minute.np() == np.timedelta64(216, 'm')
        assert minute.py() == timedelta(minutes=216)
        assert minute.np(raw=True) == 216
        assert minute.py(raw=True) == 216

    @pytest.mark.nep49
    def test_second(self, q, kx):
        second = q('03:36:59')
        assert isinstance(second, kx.SecondAtom)
        assert second.np() == np.timedelta64(13019, 's')
        assert second.py() == timedelta(seconds=13019)
        assert second.np(raw=True) == 13019
        assert second.py(raw=True) == 13019

    @pytest.mark.nep49
    def test_time(self, q, kx):
        time = q('16:36:29.214')
        assert isinstance(time, kx.TimeAtom)
        assert time.np() == np.timedelta64(59789214, 'ms')
        assert time.py() == timedelta(seconds=59789, microseconds=214000)
        assert time.np(raw=True) == 59789214
        assert time.py(raw=True) == 59789214


class Test_SymbolAtom:
    def test_str(self, q):
        assert 'symsymsym' == str(q('`symsymsym'))
        assert 'symsymsym' == q('`symsymsym').py()
        assert '🙃' == str(q('`$"🙃"'))
        assert '🙃' == q('`$"🙃"').py()

    def test_bytes(self, q):
        assert b'symsymsym' == bytes(q('`symsymsym'))
        assert b'symsymsym' == q('`symsymsym').py(raw=True)
        assert b'\xf0\x9f\x99\x83' == bytes(q('`$"🙃"'))
        assert b'\xf0\x9f\x99\x83' == q('`$"🙃"').py(raw=True)

    def test_dunders(self, q):
        assert q('`dun') + 'der' == 'dunder'
        assert 'der' + q('`dun') == 'derdun'
        assert int(q('`102201')) == 102201
        assert float(q('`102.201')) == 102.201
        assert complex(q('`102.201')) == 102.201+0j


class Test_GUIDAtom:
    def test_py(self, q):
        assert UUID(int=0) == q('0Ng')
        g = q('first -1?0Ng')
        assert g.py() == UUID(str(g))
        assert isinstance(g.py(raw=True), complex)


class Test_Vector:
    def test_len(self, q):
        assert len(q('til 3')) == 3

    def test_getting(self, q):
        v = q('2 + til 3')
        with pytest.raises(IndexError):
            v[-4]
        assert v[-3] == 2
        assert v[-2] == 3
        assert v[-1] == 4
        assert v[0] == 2
        assert v[1] == 3
        assert v[2] == 4
        with pytest.raises(IndexError):
            v[3]
        assert v[q('1 2')].py() == [3, 4]
        f = q('1 2 3f')
        assert f[q('1 2')].py() == [2.0, 3.0]
        assert f._unlicensed_getitem(-1) == 3.0

    def test_setting(self, q, kx):
        v = q.til(10)
        with pytest.raises(IndexError):
            v[10] = 10
        with pytest.raises(IndexError):
            v[-11] = 10
        with pytest.raises(kx.QError) as err:
            v[2] = 'a'
        assert "Failed to assign value of type: <class 'str'>" in str(err)
        for i in range(3):
            v[0]+=i
        assert v[0]>0
        v[1] = 2
        assert v[1] == 2
        v[-1] = 20
        assert v[9] == 20
        vlist = kx.List([1, 0.1, 3])
        vlist[2] = 'a'
        assert vlist[2] == 'a'
        vlist[:2] = 0.1
        assert vlist[0] == 0.1
        assert vlist[1] == 0.1

    def test_append(self, q, kx):
        p0 = [1, 2, 3]
        q0 = kx.toq(p0)
        p0.append(1)
        q0.append(1)
        assert all(p0 == q0)
        with pytest.raises(kx.QError) as err:
            q0.append(2.0)
        assert "Appending data of type: <class 'pykx.wrappers.FloatAtom'" in str(err)

        p1 = [1, 2.0, 3]
        q1 = kx.toq(p1)
        p1.append('a')
        p1.append([1, 2, 3])
        q1.append('a')
        q1.append([1, 2, 3])
        assert q('{x~y}', p1, q1)
        assert 5 == len(q1)

    def test_extend(self, q, kx):
        p0 = [1, 2, 3]
        q0 = kx.toq(p0)
        p0.extend([1])
        q0.extend([1])
        assert all(p0 == q0)
        with pytest.raises(kx.QError) as err:
            q0.extend([1, 2.0])
        assert "Extending data of type: <class 'pykx.wrappers.List'" in str(err)

        p1 = [1, 2.0, 3]
        q1 = kx.toq(p1)
        p1.extend(['a'])
        p1.extend([1, 2, 3])
        q1.extend(['a'])
        q1.extend([1, 2, 3])
        assert q('{x~y}', p1, q1)
        assert 7 == len(q1)

    def test_count(self, q):
        v = q('1 2 1 2 3 2 2 1')
        assert v.count(1) == 3
        assert v.count(2) == 4
        assert v.count(3) == 1

    # def test_setting(self, q):
    #     v = q('2 + til 3')
    #     with pytest.raises(IndexError):
    #         v[-4]
    #     v[-3] = -1000
    #     assert v[-3] == -1000
    #     v[-2] = -1001
    #     assert v[-2] == -1001
    #     v[-1] = -1002
    #     assert v[-1] == -1002
    #     v[0] = 1000
    #     assert v[0] == 1000
    #     v[1] = 1001
    #     assert v[1] == 1001
    #     v[2] = 1002
    #     assert v[2] == 1002
    #     with pytest.raises(IndexError):
    #         v[3]

    def test_reversed(self, q):
        v = q('til 5')
        assert list(reversed(list(reversed(v)))) == list(v)

    def test_comparisons(self, q, kx):
        v = q('til 3')

        assert 0 in v
        assert 2 in v
        assert 1 in v
        assert 3 not in v
        assert -1 not in v
        assert q('0') in v
        assert q('2') in v
        assert q('1') in v
        assert q('3') not in v
        assert q('-1') not in v

        assert (v == v).all()
        assert all(v == v)
        assert isinstance(v == v, kx.BooleanVector)
        assert all(v == range(3))
        assert np.array_equal((v == range(0, 6, 2)), [True, False, False])

        assert not all(v != v)
        assert not all(v != range(3))
        assert np.array_equal((v != range(0, 6, 2)), [False, True, True])

        assert not any(v > v)
        assert np.array_equal(v > 1, [False, False, True])

        assert not any(v < v)
        assert np.array_equal(v < 1, [True, False, False])

        assert all(v >= v)
        assert np.array_equal(v >= 1, [False, True, True])

        assert all(v <= v)
        assert np.array_equal(v <= 1, [True, True, False])

        assert q('"b"$til 3').np().dtype == bool
        assert q('"x"$til 3').np().dtype == np.uint8
        assert q('"h"$til 3').np().dtype == np.int16
        assert q('"i"$til 3').np().dtype == np.int32
        assert q('"j"$til 3').np().dtype == np.int64
        assert q('"e"$til 3').np().dtype == np.dtype('float32')
        assert q('"f"$til 3').np().dtype == np.dtype('float64')
        assert q('"c"$til 3').np().dtype == np.dtype('S1')

        assert v.py() == [0, 1, 2]
        assert (q('"abc"') == b'abc').all()

    def test_empty_vector(self, q):
        assert q('()') == q('()')
        assert q('()') != q('"j"$()')

        assert q('"b"$()').np().dtype == bool
        assert q('"b"$()').np(raw=True).dtype == bool

        assert q('"x"$()').np().dtype == np.uint8
        assert q('"x"$()').np(raw=True).dtype == np.uint8

        assert q('"h"$()').np().dtype == np.int16
        assert q('"h"$()').np(raw=True).dtype == np.int16

        assert q('"i"$()').np().dtype == np.int32
        assert q('"i"$()').np(raw=True).dtype == np.int32

        assert q('"j"$()').np().dtype == np.int64
        assert q('"j"$()').np(raw=True).dtype == np.int64

        assert q('"e"$()').np().dtype == np.dtype('float32')
        assert q('"e"$()').np(raw=True).dtype == np.dtype('float32')

        assert q('"f"$()').np().dtype == np.dtype('float64')
        assert q('"f"$()').np(raw=True).dtype == np.dtype('float64')

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q('1 0N 3').np().base, q('1 0N 3').np(raw=True))

    @pytest.mark.nep49
    def test_pd(self, q, pd):
        assert isinstance(q('til 10000').pd(), pd.Series)
        assert all(q('til 10000').pd() == np.arange(10000))

    def test_array(self, q):
        assert all(np.array(q('til 100')) == q('til 100').np())

    def test_index(self, q):
        v = q('til 100')
        assert v.index(22) == 22
        assert v.index(88) == 88
        with pytest.raises(ValueError):
            v.index(-1)
        with pytest.raises(ValueError):
            v.index(100)

    def test_dunder_ops(self, q):
        assert all(q('1 10 100') + [1, 2, 3] == [2, 12, 103])
        assert all(q('1 10 100') - [1, 2, 3] == [0, 8, 97])
        assert all(q('1 10 100') * [1, 2, 3] == [1, 20, 300])
        assert all(q('1 10 150') / [1, 2, 3] == [1, 5, 50])
        assert all(q('1 10 100') // [1, 2, 3] == [1, 5, 33])
        assert all(q('1 10 100') // [1, 2, 3] == [1, 5, 33])

    def test_pow(self, q):
        assert all(pow(q('2 7 10'), q('3 4 5'), q('5 2 1000000000'))
               == q('3 1 100000'))
        assert all(pow(q('2 7 10'), q('3 4 5'), 6) == q('2 1 4'))
        assert all(pow(q('2 7 10'), q('3 4 5')) == q('8 2401 100000'))
        assert all(pow(q('2 7 10'), 2) == q('4 49 100'))
        assert all(pow(2, q('2 7 10')) == q('4 128 1024'))

    def test_real_pow(self, q):
        v = q('1.2 3.4 5.6')
        assert np.allclose(pow(v, 2), [1.44, 11.56, 31.36])
        assert np.allclose(pow(2, v), [2.297397, 10.55606, 48.50293])
        with pytest.raises(TypeError):
            pow(v, 2, 3)
        with pytest.raises(TypeError):
            pow(v, 2, v)
        with pytest.raises(TypeError):
            pow(2, v, 3)

    @pytest.mark.nep49
    def test_pa(self, q, pa):
        v = q('100?0Wt')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.DurationArray)

        v = q('100?0Wv')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.DurationArray)

        with pytest.raises(pa.lib.ArrowNotImplementedError):
            q('100?0Wu').pa()

        v = q('100?0Wn')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.DurationArray)

        v = q('"d"$100?0Wi')
        assert all(v.pa().to_numpy(zero_copy_only=False) == v.np())
        assert isinstance(v.pa(), pa.lib.Date32Array)

        with pytest.raises(pa.lib.ArrowNotImplementedError):
            q('"m"$100?0Wi').pa()

        v = q('100?0Wp')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.TimestampArray)

        v = q('100?`3')
        assert all(v.pa().to_numpy(zero_copy_only=False) == v.np())
        assert isinstance(v.pa(), pa.lib.StringArray)

        v = q('"c"$100?256')
        assert all(v.pa().to_numpy(zero_copy_only=False) == v.np())
        assert isinstance(v.pa(), pa.lib.BinaryArray)

        v = q('100?100.0')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.DoubleArray)

        v = q('"e"$100?100.0')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.FloatArray)

        v = q('100?0Wj')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.Int64Array)

        v = q('100?0Wi')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.Int32Array)

        v = q('100?0Wh')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.Int16Array)

        v = q('"x"$100?256')
        assert all(pa.array(v).to_numpy() == v.np())
        assert all(v.pa().to_numpy() == v.np())
        assert isinstance(v.pa(), pa.lib.UInt8Array)

        v = q('"b"$100?2')
        assert all(v.pa().to_numpy(zero_copy_only=False) == v.np())
        assert isinstance(v.pa(), pa.lib.BooleanArray)

    def test_has_null_and_has_inf(self, q):
        assert not q('(`o;7;{x*y+z})').has_nulls
        assert q('(`;7;{x*y+z})').has_nulls
        assert q('(`hmmm;0N;{x*y+z})').has_nulls

        assert not q('(`hmmm;0N;"not inf";123456789;{x*y+z})').has_infs
        assert q('(`hmmm;0N;"not inf";0W;{x*y+z})').has_infs
        assert q('(`hmmm;0N;"not inf";-0W;{x*y+z})').has_infs

        assert not q('011011100b').has_nulls
        assert not q('011011100b').has_infs

        assert not q('v where not null v:10?0Ng').has_nulls
        assert q('0Ng,3?0Ng').has_nulls
        assert not q('v where not null v:10?0Ng').has_infs

        assert not q('0xdeadbeef').has_nulls
        assert not q('0xdeadbeef').has_infs

        def f(type_code, zero):
            v = q(f'v:v where (not &[;]. v=/:(neg z;z:0W{type_code})) & not null v:100?{zero};v')
            assert not v.has_nulls
            assert not v.has_infs
            assert q(f'@[v;-3?50;:;0N{type_code}]').has_nulls
            assert q(f'@[v;-3?50;:;0W{type_code}]').has_infs

        types = (
            ('h', '0h'), ('i', '0i'), ('j', '0j'), ('e', '10000e'), ('f', '10000f'), ('p', '0p'),
            ('m', '2000.01m'), ('d', '2000.01.01d'), ('n', '00:00:00.000000000'), ('u', '00:00'),
            ('v', '00:00:00'), ('t', '00:00:00.000'),
        )
        for type_code, zero in types:
            f(type_code, zero)


class Test_List:
    v = '(0b;"G"$"00000000-0000-0000-0000-000000000001";0x02;3h;4i;5j;6e;7f)'

    def test_type(self, q, kx):
        assert q(self.v).t == kx.List.t
        assert isinstance(q(self.v), kx.List)

    def test_getting(self, q, kx):
        qv = q(self.v)
        with pytest.raises(IndexError):
            qv[8]
        with pytest.raises(IndexError):
            qv[-9]
        assert qv[0].py() is False
        assert qv[1].t == -2
        assert len(qv) == 8
        assert qv[-1] == 7.0
        assert isinstance(qv[-1].py(), float)

        x = qv[2:5]
        assert isinstance(x[0], kx.ByteAtom)
        assert isinstance(x[1], kx.ShortAtom)
        assert isinstance(x[2], kx.IntAtom)

        assert qv[q('0 7')].py() == [False, 7.0]

    # def test_setting(self, q):
    #     x = q(self.v)
    #     x[0] = q('1b')
    #     assert x[0]
    #     x[-1] = 'no longer 7f'
    #     assert x[-1].py() == 'no longer 7f'
    #     x[2:4] = [q('(0x20; -1234)'), 'qwerty']
    #     assert all(x[2:4] == [q('(0x20; -1234)'), 'qwerty'])
    #     with pytest.raises(kx.QError):
    #         x[4:8] = [1, 2]
    #     x[4:7] = 0
    #     assert all(a == 0 for a in x[4:7])

    def test_py(self, q, kx):
        assert q('1 0N 3h').py() == [1, q('0Nh'), 3]
        assert isinstance(q('1 0N 3h').py()[1], kx.ShortAtom)

    @pytest.mark.nep49
    def test_np(self, q, kx):
        qv = q(self.v)
        assert qv.np(raw=True).dtype == np.uintp
        assert qv.np().dtype == object
        assert qv.np()[1] == UUID(int=1)
        assert isinstance(qv.np()[-1], float)
        assert qv.np()[0] is False

        nested = q('(((1 2);(3 4));((5 6);(7 8));((9 10);(11 12)))')
        npnested = nested.np()
        pynested = np.array([
            np.array([np.array([1, 2], dtype="int64"), np.array([3, 4], dtype="int64")], dtype=object), # noqa
            np.array([np.array([5, 6], dtype="int64"), np.array([7, 8], dtype="int64")], dtype=object), # noqa
            np.array([np.array([9, 10], dtype="int64"), np.array([11, 12], dtype="int64")], dtype=object)], # noqa
            dtype=object
        )
        for x in range(3):
            for y in range(2):
                assert (npnested[x][y]==pynested[x][y]).all()

        ndarray = np.arange(3 * 1 * 2).reshape(3, 1, 2)
        qndarray = kx.K(ndarray).np()
        for x in range(3):
            assert(qndarray[x][0] == ndarray[x][0]).all()

    def test_contains(self, q):
        qv = q(self.v)
        assert object() not in qv
        assert q('0b') in qv
        assert False in qv
        assert UUID(int=1) in qv
        assert 7 in qv
        assert 8 not in qv
        assert (1,) not in q('til 3')
        assert () not in q('til 0')
        assert any(((2, 3) == x).all() for x in q('((0;1);(2;3))'))
        assert q('5') not in ()
        assert not any((q('5') == x).all() for x in ((7, 8), (9, 5)))
        assert [(q('5') == x).all() for x in ((7, 8), (5, 5))] == [False, True]

    def test_empty_vector(self, q):
        assert q('0h$()').np().dtype == object
        assert q('0h$()').np(raw=True).dtype == np.uint64


# NaN is tricky to compare, so we generate GUID vectors until we get one whose complex form has no
# NaNs in it.
def _guids_without_NaNs(q):
    guids = q('-8?0Ng')
    s = sum(guids.np(raw=True))
    while math.isnan(s.real + s.imag):
        guids = q('-8?0Ng')
        s = sum(guids.np(raw=True))
    return guids


@pytest.mark.nep49
def test_PandasUUIDArray(q, kx):
    guids = _guids_without_NaNs(q)
    raw = guids.np(raw=True)
    v = kx.wrappers.PandasUUIDArray(raw)
    assert all(v == kx.wrappers.PandasUUIDArray._from_sequence(raw))
    with pytest.raises(NotImplementedError):
        kx.wrappers.PandasUUIDArray._from_factorized(raw)
    assert v[0] == raw[0]
    assert len(v) == len(guids)
    assert all(np.array(v) == raw)
    assert v.nbytes == raw.nbytes
    assert all(v.isna() == (raw == 0))
    assert all(v.take((1, 3, 1, 2) == raw.take((1, 3, 1, 2))))
    v_copy = v.copy()
    assert v_copy is not v
    assert all(v_copy == v)
    with pytest.raises(NotImplementedError):
        v._concat_same_type()


@pytest.mark.nep49
def test_ArrowUUIDType(q, kx, pa):
    guids = _guids_without_NaNs(q)
    v_pd = kx.wrappers.PandasUUIDArray(guids.np(raw=True))
    v = pa.array(v_pd)
    assert isinstance(v, pa.lib.ExtensionArray)
    assert isinstance(v.type, kx.wrappers.ArrowUUIDType)
    assert all(v.to_pandas() == v_pd)
    assert v.type.extension_name == 'pykx.uuid'
    assert isinstance(v.type.to_pandas_dtype(), kx.wrappers.PandasUUIDType)
    assert v.type.__arrow_ext_serialize__() == b''
    assert isinstance(v.type.__arrow_ext_deserialize__(v.type.storage_type, ''),
                      kx.wrappers.ArrowUUIDType)
    assert isinstance(kx.wrappers.PandasUUIDArray(v), kx.wrappers.PandasUUIDArray)

    with pytest.raises(ValueError, match=r'(?i)Cannot convert multiple chunks'):
        pa.Table.from_arrays(
            [pa.chunked_array([q('enlist 0Ng'), q('enlist 0Ng')])],
            names=['a']
        ).to_pandas()


def test_deserialize(kx):
    with pytest.raises(kx.QError) as err:
        kx._wrappers.deserialize(b'\x01\x02\x00\x00\x10\x00\x00\x00\x80accesss\x00')
        assert 'access' in str(err.value)


def test_deserialize_unsupported_message(kx):
    with pytest.raises(kx.QError) as err:
        kx.deserialize(b'unsupported message format')
    assert 'Failed to deserialize supplied non PyKX IPC' in str(err.value)


class Test_GUIDVector:
    guid_strings = [
        '7a964b5c-0185-6160-a4ff-d94bc7df5f62',
        '669ec0f3-56a1-3d31-611a-850ba0f054a5',
        '3ef9f315-7619-ee0d-557d-00a38e6fdbe2',
        'ccd2c597-63c6-c58f-d54a-6a1d0dd79401',
        '9dd64cf6-203a-0fc6-f224-163797c7c067',
        '55b3d3ac-617f-11cd-9185-760d4c861459',
        '6fe4ccd0-92e1-eb95-5da8-2b23044935f1',
        '7e1793d2-6edf-5b96-d3ec-1c524df14dc2',
    ]
    q_vec_str = '(' + ';'.join(('"G"$"' + x + '"' for x in guid_strings)) + ')'
    uuids = [UUID(hex=x) for x in guid_strings]
    as_complex = [
        +1.87919031e+156+7.34202801e+165j,
        +1.67701176e-071-7.55226709e-129j,
        +1.41062692e-241-1.61783697e+168j,
        -1.09574796e-232+4.86231369e-301j,
        -3.09257877e+029+5.98095014e+191j,
        -1.79952215e+063+1.32499102e+121j,
        -4.44634916e-203-2.16568222e+237j,
        -5.68962006e-201-2.57204856e+011j,
    ]

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.GUIDVector.t
        assert isinstance(q(self.q_vec_str), kx.GUIDVector)

    def test_creation(self, q, kx):
        guids = q(self.q_vec_str)
        assert guids.t == kx.GUIDVector.t

    def test_py(self, q):
        assert q(self.q_vec_str).py() == self.uuids

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), np.asarray(self.uuids))

    def test_raw_py(self, q):
        assert np.allclose(q(self.q_vec_str).py(raw=True), self.as_complex)

    @pytest.mark.nep49
    def test_pd(self, q):
        assert all(q(self.q_vec_str).pd() == np.asarray(self.uuids))

    @pytest.mark.nep49
    def test_raw_pd(self, q):
        assert np.allclose(q(self.q_vec_str).pd(raw=True), self.as_complex)

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.allclose(
            q(self.q_vec_str).np(raw=True),
            np.asarray(self.as_complex))

    def test_getting(self, q):
        guids = q(self.q_vec_str)
        assert guids[2:5].py() == self.uuids[2:5]
        assert guids[2] == self.uuids[2]
        assert guids[q('1')] == self.uuids[1]

    # def test_setting(self, q):
    #     x = UUID(int=1234567890)
    #     v = q(self.q_vec_str)
    #     v[-1] = x
    #     assert v[-1] == x

    # def test_setting_slice(self, q):
    #     v = q(self.q_vec_str)
    #     us = [UUID(int=58913), UUID(int=123456789), UUID(int=1221)]
    #     x = self.uuids[:]
    #     x[2:5] = us
    #     v[2:5] = us
    #     assert v.py() == x

    def test_empty_vector(self, q):
        assert q('"g"$()').np().dtype == object
        assert q('"g"$()').np(raw=True).dtype == complex

    @pytest.mark.nep49
    def test_pa(self, q, pa):
        assert q(self.q_vec_str).pa(raw=False).to_string() == '[\n  7A964B5C01856160A4FFD94BC7DF5F62,\n  669EC0F356A13D31611A850BA0F054A5,\n  3EF9F3157619EE0D557D00A38E6FDBE2,\n  CCD2C59763C6C58FD54A6A1D0DD79401,\n  9DD64CF6203A0FC6F224163797C7C067,\n  55B3D3AC617F11CD9185760D4C861459,\n  6FE4CCD092E1EB955DA82B23044935F1,\n  7E1793D26EDF5B96D3EC1C524DF14DC2\n]' # noqa
        assert q(self.q_vec_str).pa(raw=True).to_string()  == '[\n  7A964B5C01856160A4FFD94BC7DF5F62,\n  669EC0F356A13D31611A850BA0F054A5,\n  3EF9F3157619EE0D557D00A38E6FDBE2,\n  CCD2C59763C6C58FD54A6A1D0DD79401,\n  9DD64CF6203A0FC6F224163797C7C067,\n  55B3D3AC617F11CD9185760D4C861459,\n  6FE4CCD092E1EB955DA82B23044935F1,\n  7E1793D26EDF5B96D3EC1C524DF14DC2\n]' # noqa
        q(self.q_vec_str).pa().to_pandas()
        assert True


class Test_CharVector:
    an = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'
    shrug = '¯\\_(ツ)_/¯'
    emoji = '🙃🙃🙃🙃🙃'
    japanese = 'テスト目的の日本語文'

    def test_str(self, q):
        assert str(q('""')) == ''
        assert str(q('.Q.an')) == self.an
        assert str(q(f'"{repr(self.shrug)[1:-1]}"')) == self.shrug
        assert str(q(f'"{self.emoji}"')) == self.emoji
        assert str(q(f'"{self.japanese}"')) == self.japanese

    def test_bytes(self, q):
        assert bytes(q('""')) == b''
        assert bytes(q('.Q.an')) == self.an.encode()
        assert bytes(q(f'"{repr(self.shrug)[1:-1]}"')) == self.shrug.encode()
        assert bytes(q(f'"{self.emoji}"')) == self.emoji.encode()
        assert bytes(q(f'"{self.japanese}"')) == self.japanese.encode()

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(
            q('.Q.an').np(),
            np.frombuffer(self.an.encode(), dtype='|S1')
        )
        assert np.array_equal(
            q(f'"{repr(self.shrug)[1:-1]}"').np(),
            np.frombuffer(self.shrug.encode(), dtype='|S1')
        )
        assert np.array_equal(
            q(f'"{self.emoji}"').np(),
            np.frombuffer(self.emoji.encode(), dtype='|S1')
        )
        assert np.array_equal(
            q(f'"{self.japanese}"').np(),
            np.frombuffer(self.japanese.encode(), dtype='|S1')
        )

    def test_py(self, q):
        assert q('.Q.an').py() == self.an.encode()
        assert q(f'"{repr(self.shrug)[1:-1]}"').py() == self.shrug.encode()
        assert q(f'"{self.emoji}"').py() == self.emoji.encode()
        assert q(f'"{self.japanese}"').py() == self.japanese.encode()

    def test_empty_vector(self, q):
        assert q('"c"$()').np().dtype == np.dtype('|S1')
        assert q('"c"$()').np(raw=True).dtype == np.dtype('|S1')


class Test_SymbolVector:
    q_vec_str = '`abcd`efg`hijk`lmnop`qrs`tuv`wx`y`z'

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.SymbolVector.t
        assert isinstance(q(self.q_vec_str), kx.SymbolVector)

    def test_getting(self, q):
        vec = q(self.q_vec_str)
        assert vec[0] == 'abcd'
        assert vec[-1] == 'z'
        assert np.array_equal(
            vec[2:4],
            np.asarray(['hijk', 'lmnop'])
        )
        with pytest.raises(IndexError):
            vec[-10]
        with pytest.raises(IndexError):
            vec[9]
        assert vec[q('2')] == 'hijk'

    # def test_setting(self, q):
    #     v = q(self.q_vec_str)
    #     v[2] = 'xyzzy'
    #     assert v[2] == 'xyzzy'

    #     v[1:4] = 'afapha'
    #     assert v == q('`abcd`afapha`afapha`afapha`qrs`tuv`wx`y`z')

    #     v[2:5] = ('q', 'w', 'e')
    #     assert v == q('`abcd`afapha`q`w`e`tuv`wx`y`z')

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(),
            np.asarray(self.q_vec_str.split('`')[1:])
        )

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True),
            np.asarray([x.encode() for x in self.q_vec_str.split('`')[1:]])
        )

    def test_empty_vector(self, q):
        assert q('`$()').np().dtype == np.dtype('O')
        assert q('`$()').np(raw=True).dtype == np.dtype('O')


# class Test_TemporalVector:
#     def test_setting(self, q):
#         class sub1(kx.MonthVector):
#             pass

#         class sub2:
#             pass

#         class sub3(sub1, sub2):
#             pass

#         v = sub3(q('2000.01 2000.02m'))
#         v[1] = np.datetime64('2020-08', 'M')
#         assert v[1] == q('2020.08m')


class Test_TemporalFixedVector:
    # def test_setting(self, q):
    #     v = q('2000.01 2000.02m')
    #     v[1] = np.datetime64('2020-08', 'M')
    #     assert v[1] == q('2020.08m')

    def test_py(self, q):
        x = q('2133.04.29D23:12:27.985231872 1761.01.20D15:06:00.175740928')
        y = q('2000.01 2000.02m')
        z = q('2000.01.01 2000.01.02')

        assert x.py() == [datetime(2133, 4, 29, 23, 12, 27, 985231),
                          datetime(1761, 1, 20, 15, 6, 0, 175740)]
        assert y.py() == [date(2000, 1, 1), date(2000, 2, 1)]
        assert z.py() == [date(2000, 1, 1), date(2000, 1, 2)]

        assert x.py(raw=True) == [4207417947985231872, -7540332839824259072]
        assert y.py(raw=True) == [0, 1]
        assert z.py(raw=True) == [0, 1]


class Test_TemporalSpanVector:
    # def test_setting(self, q):
    #     v = q('00:00:00.000000000 01:02:03.040506070')
    #     v[0] = np.timedelta64(37230405060708, 'ns')
    #     assert v[0] == q('10:20:30.405060708')

    def test_py(self, q):
        w = q('41927D02:22:17.297584128 55200D12:09:33.595746304')
        x = q('21:02 40:09')
        y = q('00:48:18 01:46:00')
        z = q('16:36:29.214 08:31:52.958')

        assert w.py() == [
            timedelta(days=41927, seconds=8537, microseconds=297584),
            timedelta(days=55200, seconds=43773, microseconds=595746)]
        assert x.py() == [
            timedelta(seconds=75720),
            timedelta(days=1, seconds=58140)]
        assert y.py() == [
            timedelta(seconds=2898),
            timedelta(seconds=6360)]
        assert z.py() == [
            timedelta(seconds=59789, microseconds=214000),
            timedelta(seconds=30712, microseconds=958000)]

        assert w.py(raw=True) == [3622501337297584128, 4769323773595746304]
        assert x.py(raw=True) == [1262, 2409]
        assert y.py(raw=True) == [2898, 6360]
        assert z.py(raw=True) == [59789214, 30712958]


class Test_TimestampVector:
    q_vec_str = ' '.join((
        '2133.04.29D23:12:27.985231872',
        '1761.01.20D15:06:00.175740928',
        '1970.11.24D07:14:56.031040000',
        '2002.05.08D12:13:17.444856160',
        '2030.04.20D01:58:00.379527808',
        '1810.04.19D20:57:38.095740928',
        '2285.08.16D21:33:55.448855552',
        '1761.04.19D01:57:20.951794688',
    ))
    np_ts = np.array((
        '2133-04-29T23:12:27.985231872',
        '1761-01-20T15:06:00.175740928',
        '1970-11-24T07:14:56.031040000',
        '2002-05-08T12:13:17.444856160',
        '2030-04-20T01:58:00.379527808',
        '1810-04-19T20:57:38.095740928',
        '2285-08-16T21:33:55.448855552',
        '1761-04-19T01:57:20.951794688'),
        dtype='datetime64[ns]'
    )

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.TimestampVector.t
        assert isinstance(q(self.q_vec_str), kx.TimestampVector)

    def test_getting(self, q):
        assert q(self.q_vec_str)[2] == self.np_ts[2]
        assert q(self.q_vec_str)[-1] == self.np_ts[-1]

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), self.np_ts)

    @pytest.mark.nep49
    def test_timestamp_timezone_vector(self, kx):
        kx.config._set_keep_local_times(False)
        la = pytz.timezone("America/Los_Angeles")
        tz = [datetime(2022, 7, 20, 10, 0, 0, 0, la), datetime(2021, 6, 5, 7, 0, 0, 0, la)]

        assert not tz == kx.toq(tz).py()
        assert tz == kx.TimestampVector(tz).py(tzinfo=pytz.UTC)
        assert tz == kx.toq(tz).cast(kx.TimestampVector).py(tzinfo=la)
        kx.config._set_keep_local_times(True)
        assert kx.toq(tz).py() == \
            [datetime(2022, 7, 20, 10, 0, 0, 0), datetime(2021, 6, 5, 7, 0, 0, 0)]
        assert kx.toq(tz).cast(kx.TimestampVector).py(tzinfo=la, tzshift=False) == tz
        assert kx.TimestampVector(tz).py(tzinfo=la, tzshift=False) == tz

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True) + 946684800000000000,
            self.np_ts.astype(np.int64)
        )

    def test_empty_vector(self, q):
        assert q('"p"$()').np().dtype == np.dtype('datetime64[ns]')
        assert q('"p"$()').np(raw=True).dtype == np.int64

    def test_extracting_date_and_time(self, kx):
        ts = kx.q('2023.10.25D16:42:01.292070013')
        assert ts.date == kx.DateAtom(kx.q('2023.10.25'))
        assert ts.time == kx.TimeAtom(kx.q('16:42:01.292'))
        assert ts.year == kx.IntAtom(kx.q('2023i'))
        assert ts.month == kx.IntAtom(kx.q('10i'))
        assert ts.day == kx.IntAtom(kx.q('25i'))
        assert ts.hour == kx.IntAtom(kx.q('16i'))
        assert ts.minute == kx.IntAtom(kx.q('42i'))
        assert ts.second == kx.IntAtom(kx.q('1i'))

        ts_2 = kx.q('2018.11.09D12:21:08.456123789')
        tsv = kx.q('enlist', ts, ts_2)
        assert (tsv.date == kx.DateVector(kx.q('2023.10.25 2018.11.09'))).all()
        assert (tsv.time == kx.TimeVector(kx.q('16:42:01.292 12:21:08.456'))).all()
        assert (tsv.year == kx.IntVector(kx.q('2023 2018i'))).all()
        assert (tsv.month == kx.IntVector(kx.q('10 11i'))).all()
        assert (tsv.day == kx.IntVector(kx.q('25 9i'))).all()
        assert (tsv.hour == kx.IntVector(kx.q('16 12i'))).all()
        assert (tsv.minute == kx.IntVector(kx.q('42 21i'))).all()
        assert (tsv.second == kx.IntVector(kx.q('1 8i'))).all()


class Test_MonthVector:
    q_vec_str = '2006.04 1947.10 1876.04 2170.01m'
    np_months = np.array(['2006-04', '1947-10', '1876-04', '2170-01'],
                         dtype='datetime64[M]')

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.MonthVector.t
        assert isinstance(q(self.q_vec_str), kx.MonthVector)

    def test_getting(self, q):
        assert q(self.q_vec_str)[2] == self.np_months[2]
        assert q(self.q_vec_str)[-1] == self.np_months[-1]

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), self.np_months)

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True) + 360,
            self.np_months.astype(np.int32)
        )

    def test_empty_vector(self, q):
        assert q('"m"$()').np().dtype == np.dtype('datetime64[M]')
        assert q('"m"$()').np(raw=True).dtype == np.int32


class Test_DateVector:
    q_vec_str = '2006.04.01 1947.10.22 1876.04.07 2170.01.24'
    np_dates = np.array(
        ['2006-04-01', '1947-10-22', '1876-04-07', '2170-01-24'],
        dtype='datetime64[D]')

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.DateVector.t
        assert isinstance(q(self.q_vec_str), kx.DateVector)

    def test_getting(self, q):
        assert q(self.q_vec_str)[2] == self.np_dates[2]
        assert q(self.q_vec_str)[-1] == self.np_dates[-1]

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), self.np_dates)

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True) + 10957,
            self.np_dates.astype(np.int32)
        )

    def test_empty_vector(self, q):
        assert q('"d"$()').np().dtype == np.dtype('datetime64[D]')
        assert q('"d"$()').np(raw=True).dtype == np.int32


class Test_DatetimeVector:
    q_vec_str = ' '.join((
        '2133.04.29T23:12:27.985231872',
        '1761.01.20T15:06:00.175740928',
        '1970.11.24T07:14:56.031040000',
        '2002.05.08T12:13:17.444856160',
        '2030.04.20T01:58:00.379527808',
        '1810.04.19T20:57:38.095740928',
        '2285.08.16T21:33:55.448855552',
        '1761.04.19T01:57:20.951794688',
    ))
    np_dt = np.array((
        48696.96699057,
        -87272.37083131,
        -10629.6979626,
        858.50922968,
        11067.08194883,
        -69287.12664242,
        104322.89855843,
        -87183.91850751
    ))

    def test_type(self, q, kx):
        with pytest.warns(DeprecationWarning):
            assert q(self.q_vec_str).t == kx.DatetimeVector.t
            assert isinstance(q(self.q_vec_str), kx.DatetimeVector)

    def test_getting(self, q):
        with pytest.warns(DeprecationWarning):
            assert math.isclose(
                q(self.q_vec_str)[0].np(raw=True), self.np_dt[0])

    def test_py(self, q):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError):
                q(self.q_vec_str).py()

    def test_raw_py(self, q):
        with pytest.warns(DeprecationWarning):
            for i, x in enumerate(q(self.q_vec_str).py(raw=True)):
                assert math.isclose(x, self.np_dt[i])

    @pytest.mark.nep49
    def test_np(self, q):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError):
                q(self.q_vec_str).np()

    @pytest.mark.nep49
    def test_raw_np(self, q):
        with pytest.warns(DeprecationWarning):
            assert np.isclose(q(self.q_vec_str).np(raw=True), self.np_dt).all()

    def test_empty_vector(self, q):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError):
                q('"z"$()').np()
            assert q('"z"$()').np(raw=True).dtype == np.dtype('float64')


class Test_TimespanVector:
    q_vec_str = ' '.join((
        '20:44:06.070010215',
        '20:15:19.936021417',
        '13:01:23.842661976',
        '01:51:42.334698289',
    ))
    np_timespans = np.array(
        [74646070010215, 72919936021417, 46883842661976, 6702334698289],
        dtype='timedelta64[ns]')

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.TimespanVector.t
        assert isinstance(q(self.q_vec_str), kx.TimespanVector)

    def test_getting(self, q):
        assert q(self.q_vec_str)[2] == self.np_timespans[2]
        assert q(self.q_vec_str)[-1] == self.np_timespans[-1]

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), self.np_timespans)

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True),
            self.np_timespans.astype(np.int64)
        )

    def test_empty_vector(self, q):
        assert q('"n"$()').np().dtype == np.dtype('timedelta64[ns]')
        assert q('"n"$()').np(raw=True).dtype == np.int64


class Test_MinuteVector:
    q_vec_str = '20:44 20:15 13:01 01:51'
    np_minutes = np.array([1244, 1215, 781, 111], dtype='timedelta64[m]')

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.MinuteVector.t
        assert isinstance(q(self.q_vec_str), kx.MinuteVector)

    def test_getting(self, q):
        assert q(self.q_vec_str)[2] == self.np_minutes[2]
        assert q(self.q_vec_str)[-1] == self.np_minutes[-1]

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), self.np_minutes)

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True),
            self.np_minutes.astype(np.int32)
        )

    def test_empty_vector(self, q):
        assert q('"u"$()').np().dtype == np.dtype('timedelta64[m]')
        assert q('"u"$()').np(raw=True).dtype == np.int32


class Test_SecondVector:
    q_vec_str = '20:44:06 20:15:19 13:01:23 01:51:42'
    np_seconds = np.array([74646, 72919, 46883,  6702],
                          dtype='timedelta64[s]')

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.SecondVector.t
        assert isinstance(q(self.q_vec_str), kx.SecondVector)

    def test_getting(self, q):
        assert q(self.q_vec_str)[2] == self.np_seconds[2]
        assert q(self.q_vec_str)[-1] == self.np_seconds[-1]

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), self.np_seconds)

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True),
            self.np_seconds.astype(np.int32)
        )

    def test_empty_vector(self, q):
        assert q('"v"$()').np().dtype == np.dtype('timedelta64[s]')
        assert q('"v"$()').np(raw=True).dtype == np.int32


class Test_TimeVector:
    q_vec_str = '20:44:06.070 20:15:19.936 13:01:23.842 01:51:42.334'
    np_times = np.array([74646070, 72919936, 46883842, 6702334],
                        dtype='timedelta64[ms]')

    def test_type(self, q, kx):
        assert q(self.q_vec_str).t == kx.TimeVector.t
        assert isinstance(q(self.q_vec_str), kx.TimeVector)

    def test_getting(self, q):
        assert q(self.q_vec_str)[2].np() == self.np_times[2]
        assert q(self.q_vec_str)[-1].np() == self.np_times[-1]

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(q(self.q_vec_str).np(), self.np_times)

    @pytest.mark.nep49
    def test_raw_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True),
            self.np_times.astype(np.int32)
        )

    def test_empty_vector(self, q):
        assert q('"t"$()').np().dtype == np.dtype('timedelta64[ms]')
        assert q('"t"$()').np(raw=True).dtype == np.int32


class Test_EnumVector:
    q_vec_str = '`u$v:6#u:`abc`xyz`hmm'

    def test_getting(self, q):
        v = q(self.q_vec_str)
        assert v[0].t == -20

    @pytest.mark.nep49
    def test_np(self, q):
        assert np.array_equal(
            q(self.q_vec_str).np(),
            np.array(['abc', 'xyz', 'hmm', 'abc', 'xyz', 'hmm'])
        )
        assert np.array_equal(
            q(self.q_vec_str).np(raw=True),
            np.array([0, 1, 2, 0, 1, 2])
        )

    def test_py(self, q):
        assert q(self.q_vec_str).py() == ['abc', 'xyz', 'hmm',
                                          'abc', 'xyz', 'hmm']
        assert q(self.q_vec_str).py(raw=True) == [0, 1, 2, 0, 1, 2]

    def test_empty_vector(self, q):
        q(self.q_vec_str)
        assert q('`u$()').np().dtype == np.dtype('O')
        assert q('`u$()').np(raw=True).dtype == np.int64

    def test_bool(self, q):
        with pytest.raises(TypeError):
            bool(q(self.q_vec_str))
        with pytest.raises(TypeError):
            bool(q('0#', q(self.q_vec_str)))


class Test_Anymap:
    def test_anymap(self, kx, q, tmp_path):
        q(f'\\cd {tmp_path}') # Creating anymaps will pollute the local directory
        a = q('get`:a set ('
              '(1 2;3 4);'
              '`time`price`vol!(2022.03.29D16:45:14.880819;1.;100i);'
              '([]a:1 2;b:("ab";"cd")))')
        assert isinstance(a, kx.Anymap)
        assert isinstance(a, kx.List)
        assert a.t == 77

        apy = [
            [[1, 2], [3, 4]],
            {'time': datetime(2022, 3, 29, 16, 45, 14, 880819), 'price': 1.0, 'vol': 100},
            {'a': [1, 2], 'b': [b'ab', b'cd']}
        ]
        assert a.py() == apy
        assert a[0].py() == apy[0]
        assert a[0, 0].py() == [apy[0], apy[0]]
        assert a[1:3].py() == apy[1:3]
        assert a[:3:2].py() == apy[:3:2]
        assert a[-1].py() == apy[-1]
        assert a[:].py() == apy[:]

        b = q('get`:b set ((1 2;3 4))')
        assert isinstance(b, kx.Anymap)
        assert b.py() == [[1, 2], [3, 4]]
        assert [x.tolist() for x in b.np()] == [[1, 2], [3, 4]]
        assert [x.tolist() for x in b.pd()] == [[1, 2], [3, 4]]

    def test_anymap_pyarrow(self, q, pa, tmp_path):
        q(f'\\cd {tmp_path}') # Creating anymaps will pollute the local directory
        assert q('get`:b set ((1 2;3 4))').pa().tolist() == [[1, 2], [3, 4]]


class Test_Mapping:
    def test_get(self, q):
        d = q('`a`b!`x`y')
        assert d.get('a') == 'x'
        assert d.get('a', default='hmmm') == 'x'
        assert d.get('c') is None
        assert d.get('c', default='hmmm') == 'hmmm'


class Test_Table:
    q_table_str = '([] a:til 3; b:"xyz"; c:-3?0Ng)'

    def test_bool(self, q):
        assert q(self.q_table_str).any().any()
        assert not q(self.q_table_str).all().all()
        assert q('([] a:1 2; b:"uv")').any().any()
        assert q('([] a:1 2; b:"uv")').all().all()
        assert q('([]())').all().all()
        assert q('([]();())').all().all()
        assert not q('([]())').any().any()
        assert not q('([]();())').any().any()

    def test_py(self, q):
        t = q(self.q_table_str).py()
        x = list(t.values())
        assert isinstance(x[0], list)
        assert isinstance(x[1], bytes)
        assert isinstance(x[2], list)
        assert isinstance(t, dict)
        assert t['a'] == [0, 1, 2]
        assert t['b'] == b'xyz'
        assert all(isinstance(x, UUID) for x in t['c'])

    def test_getting(self, q, kx):
        t = q(self.q_table_str)
        assert isinstance(t['a'], kx.Vector)
        assert t['a'][1] == 1
        assert isinstance(t['b'], kx.CharVector)
        assert t['b'][2] == b'z'
        assert all(isinstance(x, kx.GUIDAtom) for x in t['c'])
        assert t['b'][0] == b'x'
        with pytest.raises(TypeError):
            t[object()]
        assert (t[0]['c'] == t['c'][0]).all()

    def test_row_getting(self, q, kx):
        t = q(self.q_table_str)
        assert isinstance(t[1], kx.Table)
        assert {k: v for k, v in t[1].py().items() if k in 'ab'} == {'a': [1], 'b': b'y'}
        assert isinstance(t[0:2], kx.Table)
        assert {k: v for k, v in t[:2].py().items() if k in 'ab'} == {'a': [0, 1], 'b': b'xy'}

    # def test_setting(self, q):
    #     t = q(self.q_table_str)
    #     t['a'][-1] = 9001
    #     t['b'][0] = b'^'
    #     t['b'][1] = b'_'
    #     t['b'][2] = b'^'
    #     assert dict((str(k), v.py()) for k, v in t.items() if str(k) in 'ab') \
    #         == {'a': [0, 1, 9001], 'b': b'^_^'}
    #     t['a'] = q('6 2 1')
    #     assert t['a'].py() == [6, 2, 1]
    #     t[1] = q('(12; "w"; 0Ng)')
    #     assert t[1].py() == {'a': 12, 'b': b'w', 'c': UUID(int=0)}
    #     t[0:2] = [q('(21; "z"; 0Ng)'), q('(1212; "a"; 0Ng)')]
    #     assert t[0].py() == {'a': 21, 'b': b'z', 'c': UUID(int=0)}
    #     assert t[1].py() == {'a': 1212, 'b': b'a', 'c': UUID(int=0)}

    def test_pd(self, q, kx, pd):
        df = q(self.q_table_str).pd()
        assert isinstance(df, pd.DataFrame)
        assert list(df.keys()) == ['a', 'b', 'c']
        assert df.dtypes[0] == np.dtype('int64')
        assert df.dtypes[1] == np.dtype('|S1')
        assert df.dtypes[2] == np.dtype('object')
        assert list(df['a']) == [0, 1, 2]
        assert list(df['b']) == [b'x', b'y', b'z']
        assert all(isinstance(x, UUID) for x in df['c'])
        df = pd.DataFrame({
            'first_score': [100, 90, np.nan, 95],
            'second_score': [30, 45, 56, np.nan],
            'third_score': [np.nan, 40, 80, 90]})
        assert (kx.K(df) == df).all().all()

    def test_pd_null_time_conversion(self, q, pd):
        w = q('([]a:(03:14:15.900000000;0Nn))').pd()
        x = q('([]a:(.z.t;0Nt))').pd()
        y = q('([]a:(.z.D;0Nd))').pd()
        z = q('([]a:(2000.01;0Nm))').pd()
        pandas_2 = pd.__version__.split('.')[0] == 2
        assert w.dtypes['a'] == np.dtype('<m8[ns]')
        assert x.dtypes['a'] == np.dtype('<m8[ms]') if pandas_2 else np.dtype('<m8[ns]')
        assert y.dtypes['a'] == np.dtype('datetime64[s]') if pandas_2 else np.dtype('<M8[ns]')
        assert z.dtypes['a'] == np.dtype('O')
        assert pd.isnull(w['a'][1])
        assert pd.isnull(x['a'][1])
        assert pd.isnull(y['a'][1])
        assert pd.isnull(z['a'][1])

    def test_np(self, q):
        recarray = q(self.q_table_str).np()
        assert isinstance(recarray, np.recarray)
        assert recarray.dtype == np.dtype(
            (np.record, [('a', '<i8'), ('b', 'S1'), ('c', 'O')]))
        assert recarray[0][0] == 0
        assert recarray[0][1] == b'x'
        assert recarray[1][0] == 1
        assert recarray[1][1] == b'y'
        assert recarray[2][0] == 2
        assert recarray[2][1] == b'z'

    def test_flip(self, q, kx):
        d = q(self.q_table_str).flip
        assert isinstance(d, kx.Dictionary)
        assert dict((str(k), v.py()) for k, v in d.items() if str(k) in 'ab') \
            == {'a': [0, 1, 2], 'b': b'xyz'}

    def test_len(self, q):
        assert len(q(self.q_table_str)) == 3
        assert len(q('([] a:til 99; b:99#"xyz"; c:-99?0Ng)')) == 99

    def test_pa(self, q, pa):
        t = q(self.q_table_str)
        assert all(t.pa().to_pandas() == t.pd(raw_guids=True))

    def test_has_null_and_has_inf(self, q):
        table = q('([]0w,9?1f;0n,9?1f)')
        assert table.has_nulls
        assert table.has_infs

    def test_null_to_pandas(self, q, pd):
        q('ty:2 5 6 7 8 9 10 11 12 13 14 16 17 18 19h')
        q('nulls:(0Ng;0Nh;0Ni;0Nj;0Ne;0n;" ";`;0Np;0Nm;0Nd;0Nn;0Nu;0Nv;0Nt)')
        t = q('flip ({`$.Q.t x} each ty)!{enlist nulls[x]} each til count ty')
        df = t.pd()

        assert df['g'].iloc[0] == UUID('00000000-0000-0000-0000-000000000000')

        for c, t, v in [('h', np.int16, -2**15),
                        ('i', np.int32, -2**31),
                        ('j', np.int64, -2**63)]:
            assert df[c].dtype == t
            assert type(df[c].values) is np.ma.MaskedArray
            assert type(df[c].iloc[0]) is np.ma.core.MaskedConstant
            assert df[c].values.fill_value == v

        for c, t in [('e', np.float32),
                     ('f', np.float64)]:
            assert df[c].dtype == t
            assert np.isnan(df[c].iloc[0])

        assert df['c'].iloc[0] == b' '
        assert df['s'].iloc[0] == ''

        pandas_2 = pd.__version__.split('.')[0] == '2'
        for c, t in [('p', np.dtype('datetime64[ns]')),
                     ('m', np.dtype('datetime64[s]') if pandas_2 else np.dtype('datetime64[ns]')),
                     ('d', np.dtype('datetime64[s]') if pandas_2 else np.dtype('datetime64[ns]')),
                     ('n', np.dtype('timedelta64[ns]')),
                     ('u', np.dtype('timedelta64[s]') if pandas_2 else np.dtype('timedelta64[ns]')),
                     ('v', np.dtype('timedelta64[s]') if pandas_2 else np.dtype('timedelta64[ns]')),
                     ('t', np.dtype('timedelta64[ms]') if pandas_2 else np.dtype('timedelta64[ns]'))
        ]:
            assert df[c].dtype == t
            assert pd.isna(df[c].iloc[0])

    def test_table_constructor(self, kx, q):
        assert kx.Table(data={'x': list(range(10)), 'y': list(10 - x for x in range(10))}).py() \
            == q('([] x: til 10; y: 10 - til 10)').py()
        assert kx.Table(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['x', 'y']
        ).py() == q('([] x: til 10; y: 10 - til 10)').py()
        assert kx.Table(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['x']
        ).py() == q('([] x: til 10; x1: 10 - til 10)').py()
        assert kx.Table(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['y']
        ).py() == q('([] y: til 10; x: 10 - til 10)').py()
        assert kx.Table(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ]
        ).py() == q('([] x: til 10; x1: 10 - til 10)').py()

    def test_table_negative_indexing(self, q):
        # KXI-31898
        tab = q('([]til 5)')
        assert q('(all/)', tab[-1] == q('([]enlist 4)'))
        with pytest.raises(IndexError):
            tab[10]
        with pytest.raises(IndexError):
            tab[-6]


@pytest.mark.filterwarnings('ignore:Splayed tables are not yet implemented')
class Test_SplayedTable:
    @staticmethod
    def create_splayed_table(q, tmp_path):
        q(f'\\cd {tmp_path}')
        q('`:db/t/ set ([] a:til 3; b:"xyz"; c:-3?0Ng)')
        q(r'\l db')
        return q('t')

    def test_key_related_methods(self, q, tmp_path, kx):
        t = self.create_splayed_table(q, tmp_path)
        assert isinstance(t, kx.SplayedTable)
        assert list(t.keys()) == ['a', 'b', 'c']
        assert list(t) == ['a', 'b', 'c']
        assert len(t) == 3

    def test_not_implemented_methods(self, q, tmp_path):
        t = self.create_splayed_table(q, tmp_path)
        assert t._values is None
        with pytest.raises(NotImplementedError):
            t.any()
        with pytest.raises(NotImplementedError):
            t.all()
        with pytest.raises(NotImplementedError):
            t['a']
        with pytest.raises(NotImplementedError):
            t.items()
        with pytest.raises(NotImplementedError):
            t.values()
        with pytest.raises(NotImplementedError):
            t.flip
        with pytest.raises(NotImplementedError):
            t.pd()
        with pytest.raises(NotImplementedError):
            t.py()


@pytest.mark.filterwarnings('ignore:(Splayed|Partitioned) tables are not yet implemented')
class Test_ParitionedTable:
    def create_partitioned_table(self, q, tmp_path):
        q(f'\\cd {tmp_path}')
        q('`:db/2020.01/t/ set ([] a:til 3; b:"xyz"; c:-3?0Ng)')
        q('`:db/2020.02/t/ set ([] a:1+til 3; b:"cat"; c:-3?0Ng)')
        q('`:db/2020.03/t/ set ([] a:2+til 3; b:"bat"; c:-3?0Ng)')
        q(r'\l db')
        return q('t')

    def test_key_related_methods(self, q, tmp_path, kx):
        t = self.create_partitioned_table(q, tmp_path)
        assert isinstance(t, kx.PartitionedTable)
        assert list(t.keys()) == ['a', 'b', 'c']
        assert list(t) == ['a', 'b', 'c']
        assert len(t) == 9

    def test_not_implemented_methods(self, q, tmp_path):
        t = self.create_partitioned_table(q, tmp_path)
        assert t._values is None
        with pytest.raises(NotImplementedError):
            t.any()
        with pytest.raises(NotImplementedError):
            t.all()
        with pytest.raises(NotImplementedError):
            t['a']
        with pytest.raises(NotImplementedError):
            t.items()
        with pytest.raises(NotImplementedError):
            t.values()
        with pytest.raises(NotImplementedError):
            t.flip
        with pytest.raises(NotImplementedError):
            t.pd()
        with pytest.raises(NotImplementedError):
            t.py()


class Test_Dictionary:
    def test_bool(self, q):
        assert not q('()!()').any()
        assert q('()!()').all()
        assert q('(`x`y)!(`a`b)').any()
        assert q('(`x`y)!(`a`b)').all()

    def test_getting(self, q):
        assert q('`a`b`c!til 3')['b'] == 1
        with pytest.raises(KeyError):
            q('`a`b`c!til 3')['z']
        assert (q('`a`b`c!til 3')[q('`b`z')] == q('1 0N')).all() # q index -> q behavior
        with pytest.raises(KeyError):
            q('`a`b`c!til 3')['b', 'z'] # Python index -> Python behavior
        assert q('`a`b`c!til 3')[q('`a`b')].py() == [0, 1]

    def test_py(self, q):
        assert {'a': 0, 'b': 1, 'c': 2} == q('`a`b`c!til 3').py()
        d1 = {
            'a': 'x',
            'b': np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            'c': np.array([250, 175], dtype=np.uint8)
        }
        d2 = q('`a`b`c!(`x;til 8; 0xfaaf)').py()
        for (a, b), (x, y) in zip(d1.items(), d2.items()):
            assert a == x
            assert list(b) == list(y)

    def test_attributes(self, q, kx):
        d = q('(`a; "b"; `c)!(`x; 1; 3.141592)')
        for x, y in d.items():
            assert isinstance(x, kx.K)
            assert isinstance(y, kx.K)
        assert isinstance(d.values(), kx.List)
        assert ['x', 1, 3.141592] == d.values().py()
        assert isinstance(d.keys(), kx.List)
        assert ['a', b'b', 'c'] == d.keys().py()
        assert 3 == len(d)

    def test_has_null_and_has_inf(self, q):
        dic = q('flip ([]0w,9?1f;0n,9?1f)')
        assert dic.has_nulls
        assert dic.has_infs

    def test_nested_dict(self, q):
        single_nested = {
            'a': [b'', {}, 1],
            'b': [b'', {}, 2]
        }
        double_nested = {
            'a': [
                b'',
                {
                    'c': [9, 4],
                    'd': [7, 6]
                },
                1
            ],
            'b': [
                b'',
                {
                    'c': [19, 14],
                    'd': [17, 16]
                },
                2
            ]
        }
        assert single_nested == q('`a`b!(("";()!();1);("";()!();2))').py()
        assert double_nested == q('`a`b!((""; (`c`d)!((9; 4); (7; 6)); 1);'
                                  '(""; (`c`d)!((19; 14); (17; 16)); 2))').py()

    def test_dict_setting(self, kx, q):
        pykx_dict = kx.toq({'x': 1})
        assert all('x' == pykx_dict.keys())
        assert pykx_dict['x'] == 1
        pykx_dict['x'] = 2
        assert pykx_dict['x'] != 1
        assert pykx_dict['x'] == 2
        for i in range(3):
            pykx_dict['x'] += i
        assert pykx_dict['x'] == 5
        pykx_dict['x1'] = 10
        assert all(['x', 'x1'] == pykx_dict.keys())
        assert 10 == pykx_dict['x1']
        isinstance(pykx_dict.values(), kx.LongVector)
        pykx_dict['x2'] = 'a'
        assert all(['x', 'x1', 'x2'] == pykx_dict.keys())
        isinstance(pykx_dict.values(), kx.List)

    def test_nested_keyed_dict(self, q):
        single_nested = {
            'a': {'error': b'', 'metadata': {}, 'data': 1},
            'b': {'error': b'', 'metadata': {}, 'data': 2}
        }
        double_nested = {
            'a': {
                'error': b'',
                'metadata': {
                    'c': {'metaa': 9, 'metab': 7},
                    'd': {'metaa': 4, 'metab': 6}
                },
                'data': 1
            },
            'b': {
                'error': b'',
                'metadata': {
                    'c': {'metaa': 19, 'metab': 17},
                    'd': {'metaa': 14, 'metab': 16}
                },
                'data': 2
            }
        }

        assert q('`a`b!([] error: ("";""); metadata: (()!();()!()); data: (1; 2))').py()\
            == single_nested
        assert q('`a`b!([] error: ("";"");'
                 'metadata: ((`c`d)!([] metaa: (9; 4); metab: (7; 6));'
                 '(`c`d)!([] metaa: (19; 14); metab: (17; 16)));'
                 'data: (1; 2))').py() == double_nested


class Test_KeyedTable:
    kt = '([k1:100+til 3] x:til 3; y:`singly`keyed`table)'
    mkt = '([k1:`a`b`a;k2:100+til 3] x:til 3; y:`multi`keyed`table)'

    def test_bool(self, q):
        assert q(self.kt).any()
        assert not q(self.kt).all()
        assert q('([()]())').all()
        assert not q('([()]())').any()
        assert q(self.mkt).any()
        assert not q(self.mkt).all()
        assert q('([();()]())').all()
        assert not q('([();()]())').any()

    def test_type(self, q, kx):
        kt = q(self.kt)
        assert isinstance(kt, kx.KeyedTable)
        assert kt.t == 99

    @pytest.mark.nep49
    def test_pa(self, q, pa):
        with pytest.raises(NotImplementedError):
            q(self.kt).pa()

    def test_queries(self, q):
        # test_query does more intensive testsing of the query features. This just helps ensure
        # it works for keyed tables too, as one would expect.
        kt = q(self.kt)
        q['kt'] = kt
        assert q('~[select x from kt]')(q.qsql.select(kt, 'x'))
        assert q('~[exec x from kt]')(q.qsql.exec(kt, 'x'))
        assert q('~[update x:101b from kt]')(q.qsql.update(kt, {'x': [True, False, True]}))
        assert q('~[delete x from kt]')(q.qsql.delete(kt, 'x'))
        mkt = q(self.mkt)
        q['mkt'] = mkt
        assert q('~[select x from mkt]')(q.qsql.select(mkt, 'x'))
        assert q('~[exec x from mkt]')(q.qsql.exec(mkt, 'x'))
        assert q('~[update x:101b from mkt]')(q.qsql.update(mkt, {'x': [True, False, True]}))
        assert q('~[delete x from mkt]')(q.qsql.delete(mkt, 'x'))

    def test_pd(self, q):
        kt_pd = q(self.kt).pd()
        assert kt_pd['x'][100] == 0
        assert kt_pd['x'][101] == 1
        assert kt_pd['x'][102] == 2
        assert kt_pd['y'][100] == 'singly'
        assert kt_pd['y'][101] == 'keyed'
        assert kt_pd['y'][102] == 'table'
        assert b'pykx' not in pickle.dumps(kt_pd)

    def test_multi_keyed_pd(self, q):
        mkt_pd = q(self.mkt).pd()
        assert mkt_pd['x'][('a', 100)] == 0
        assert mkt_pd['x'][('b', 101)] == 1
        assert mkt_pd['x'][('a', 102)] == 2
        assert mkt_pd['y'][('a', 100)] == 'multi'
        assert mkt_pd['y'][('b', 101)] == 'keyed'
        assert mkt_pd['y'][('a', 102)] == 'table'

    def test_empty_keyed_table(self, q, kx):
        q_mkt_empty = q('0#`a xkey ([] a:1 2 3;b:3 4 5)')
        q_mkt_multi_empty = q('0#`a`b xkey ([] a:1 2 3;b:3 4 5;c:6 7 8)')
        mkt_empty = q_mkt_empty.pd()
        mkt_multi_empty = q_mkt_multi_empty.pd()
        assert len(mkt_empty) == 0
        assert len(mkt_multi_empty) == 0
        assert mkt_empty.index.name == 'a'
        assert mkt_multi_empty.index.names == ['a', 'b']
        assert list(mkt_empty.columns) == ['b']
        assert list(mkt_multi_empty.columns) == ['c']
        assert type(kx.toq(mkt_empty)) == kx.KeyedTable
        assert type(kx.toq(mkt_multi_empty)) == kx.KeyedTable
        assert len(kx.toq(mkt_empty)) == 0
        assert len(kx.toq(mkt_multi_empty)) == 0

    def test_py(self, q):
        assert q(self.kt).py() == {
            (100,): {'x': 0, 'y': 'singly'},
            (101,): {'x': 1, 'y': 'keyed'},
            (102,): {'x': 2, 'y': 'table'}
        }

    def test_multi_keyed_py(self, q):
        # PyPy doesn't compare dictionaries properly, so this is a workaround
        mkt_py = q(self.mkt).py()
        mkt_keys = list(mkt_py.keys())
        mkt_values = list(mkt_py.values())
        d = {
            ('a', 100): {'x': 0, 'y': 'multi'},
            ('b', 101): {'x': 1, 'y': 'keyed'},
            ('a', 102): {'x': 2, 'y': 'table'}
        }
        d_keys = list(d.keys())
        d_values = list(d.values())
        assert all(mkt_keys[x] == d_keys[x] for x in range(3))
        assert all(mkt_values[x] == d_values[x] for x in range(3))

    def test_getting(self, kx, q):
        kt = q(self.kt)
        assert kt[q('404')].py() == {'x': q('0N'), 'y': ''}
        assert kt[q('100')].py() == {'x': 0, 'y': 'singly'}
        assert kt[q('enlist 100')].py() == {'x': [0], 'y': ['singly']}
        assert kt[(100,)].py() == {'x': [0], 'y': ['singly']}
        assert kt[[100]].py() == {'x': [0], 'y': ['singly']}
        assert kt[(101,)].py() == {'x': [1], 'y': ['keyed']}
        assert kt[(102,)].py() == {'x': [2], 'y': ['table']}

    def test_multi_keyed_getting(self, kx, q):
        mkt = q(self.mkt)
        assert mkt[('z', 404)].py() == {'x': [], 'y': []}
        assert mkt[('a', 100)].py() == {'x': [0], 'y': ['multi']}
        assert mkt[('b', 101)].py() == {'x': [1], 'y': ['keyed']}
        assert mkt[('a', 102)].py() == {'x': [2], 'y': ['table']}

    def test_attributes(self, q, kx):
        mkt = q(self.mkt)
        assert list(mkt) == [('a', 100), ('b', 101), ('a', 102)]
        assert mkt.keys() == [('a', 100), ('b', 101), ('a', 102)]
        v1 = mkt.values().py()
        v2 = {
            'x': [0, 1, 2],
            'y': ['multi', 'keyed', 'table']
        }
        for (a, b), (x, y) in zip(v1.items(), v2.items()):
            assert a == x
            assert list(b) == list(y)
        assert len(mkt) == 3

    def test_keyed_table_constructor(self, kx, q):
        assert kx.KeyedTable(
            data={'x': list(range(10)), 'y': list(10 - x for x in range(10))}
        ).py() == q('([idx: til 10] x: til 10; y: 10 - til 10)').py()
        assert kx.KeyedTable(
            data={'x': list(range(10)), 'y': list(10 - x for x in range(10))},
            index=[2 * x for x in range(10)]
        ).py() == q('([idx: 2 * til 10] x: til 10; y: 10 - til 10)').py()
        assert kx.KeyedTable(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['x', 'y']
        ).py() == q('([idx: til 10] x: til 10; y: 10 - til 10)').py()
        assert kx.KeyedTable(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['x']
        ).py() == q('([idx: til 10] x: til 10; x1: 10 - til 10)').py()
        assert kx.KeyedTable(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['y']
        ).py() == q('([idx: til 10] y: til 10; x: 10 - til 10)').py()
        assert kx.KeyedTable(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ]
        ).py() == q('([idx: til 10] x: til 10; x1: 10 - til 10)').py()
        assert kx.KeyedTable(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            index=[2 * x for x in range(10)]
        ).py() == q('([idx: 2 * til 10] x: til 10; x1: 10 - til 10)').py()
        assert kx.KeyedTable(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['idx']
        ).py() == q('([index: til 10] idx: til 10; x: 10 - til 10)').py()
        assert kx.KeyedTable(
            [
                [0, 10],
                [1, 9],
                [2, 8],
                [3, 7],
                [4, 6],
                [5, 5],
                [6, 4],
                [7, 3],
                [8, 2],
                [9, 1]
            ],
            columns=['idx', 'index']
        ).py() == q('([idx1: til 10] idx: til 10; index: 10 - til 10)').py()


class Test_Function:
    def test_bool(self, q):
        assert q('{}')
        assert q('flip')
        assert q('-')
        assert q("'")
        assert q('2+')
        assert q('(flip reverse @)')
        assert q('{}\'')
        assert q('{}/')
        assert q('{}\\')
        assert q('{}\':')
        assert q('{}/:')
        assert q('{}\\:')
        assert q('.pykx.i.pyfunc')
        assert q.pykx.i.pyfunc

    def test_call(self, q):
        assert list(range(8)) == q('til')(8).py()
        assert 36 == q('{x*y+z}')(12, 2, 1).py()
        assert 999 == q('neg')(-999)
        assert [2, 3, 4, 5] == q("'")(q('2+'))(q('til 4')).py()
        assert 4 == q('*[;2]')(2)
        assert -12 == q('neg abs neg @')(12)
        assert 1024 == q.pykx.modpow(2, 10, None)

    @pytest.mark.unlicensed(unlicensed_only=True)
    def test_call_unlicensed(self, kx, q_port):
        q = kx.QConnection(port=q_port)
        funcs = (
            q('flip'), q('-'), q("'"), q('2+'), q('(flip reverse @)'), q('{}\''), q('{}/'),
            q('{}\\'), q('{}\':'), q('{}/:'), q('{}\\:'),
        )
        for x in funcs:
            with pytest.raises(kx.LicenseException, match='call a q function in a Python process'):
                x()

    def test_kwargs(self, q):
        assert 36 == q('{x*y+z}')(z=1, x=12, y=2).py()
        with pytest.raises(TypeError):
            q('{x*x} neg @')(x=10)
        with pytest.raises(TypeError):
            q('value')(x=q('{}'))
        with pytest.raises(TypeError):
            q("'")(y='hmmm')
        with pytest.raises(TypeError):
            q('*[;2]')(x=10)
        with pytest.raises(TypeError):
            q('+')(x=2, y=2)
        with pytest.raises(TypeError):
            q('+').scan(x=2, y=2)

    def test_mixed_args(self, q):
        assert 36 == q('{x*y+z}')(12, 1, z=2).py()
        assert 36 == q('{x*y+z}')(12, z=1, y=2).py()

    def test_non_lambda(self, q):
        assert 231 == q('*')(21, 11).py()
        assert 5 == q('@')(q('k)|!8'), 2)
        with pytest.raises(TypeError):
            q('*')(21, y=11).py()

    def test_kwarg_non_lambda(self, q):
        with pytest.raises(TypeError):
            q('*')(21, y=11).py()
        with pytest.raises(TypeError):
            q('@')(x=q('k)|!8'), y=2)

    def test_unexpected_kwarg(self, q):
        with pytest.raises(TypeError):
            q('{x*y+z}')(12, z=1, fake=1234, y=2)

    def test_specify_twice(self, q):
        with pytest.raises(TypeError):
            q('{x*y+z}')(12, z=1, y=2, x=12)

    def test_unsupported_conversion(self, q):
        with pytest.raises(TypeError):
            q('{x}')(object())

    def test_empty_args(self, q):
        assert q('{[]}')() == None # noqa: E711

    def test_identity(self, q, kx):
        identity = q('::')
        assert isinstance(identity, kx.Identity)
        assert identity.py() is None
        assert bool(identity) is False
        assert str(identity) == '::'
        assert repr(identity) == "pykx.Identity(pykx.q('::'))"
        assert identity.is_null
        assert not identity.is_inf

    def test_pnull(self, q, kx):
        pnull = q('value[(;)]1')
        for x in (pnull, q('value[(;)]1'), q('value[(;)]2'), ...):
            assert pnull == x
        for x in (q('::'), None, object()):
            assert pnull != x
        for x in (None, 0, -1, 1, 2, 2**63-1, -2**63+1, -2**63):
            assert not pnull > x
            assert not pnull < x
            assert not pnull >= x
            assert not pnull <= x
        assert not pnull > pnull
        assert not pnull < pnull
        assert pnull >= pnull
        assert pnull <= pnull
        assert isinstance(pnull, kx.ProjectionNull)
        assert repr(pnull) == "pykx.ProjectionNull(pykx.q('::'))"
        assert pnull.py() is Ellipsis
        assert q('104h ~ type {1b}@', pnull)
        assert pnull.is_null
        assert not pnull.is_inf

    def test_all_function_types(self, q, kx):
        assert isinstance(q('{}'), kx.Lambda)
        assert isinstance(q('flip'), kx.UnaryPrimitive)
        assert isinstance(q('flip'), kx.UnaryPrimative)
        assert isinstance(q('-'), kx.Operator)
        assert isinstance(q("'"), kx.Iterator)
        assert isinstance(q('2+'), kx.Projection)
        assert isinstance(q('(flip reverse @)'), kx.Composition)
        assert isinstance(q('{}\''), kx.Each)
        assert isinstance(q('{}/'), kx.Over)
        assert isinstance(q('{}\\'), kx.Scan)
        assert isinstance(q('{}\':'), kx.EachPrior)
        assert isinstance(q('{}/:'), kx.EachRight)
        assert isinstance(q('{}\\:'), kx.EachLeft)
        assert isinstance(q('.pykx.i.pyfunc'), kx.Foreign)
        assert isinstance(q.Q.ajf0, kx.SymbolicFunction)

    def test_args_property(self, q):
        assert q('{x+y*z}').args == ()
        assert q('{x+y*z}[;12]').args == (..., 12)
        assert q('flip reverse @').args == (q('flip'), q('reverse'))
        assert q('flip reverse each').args == (q('flip'), q('reverse each'))
        assert q('{[a;b] a+b}').scan.args == ()
        f = q('{x+y*z}[;12]')
        ff = f.func(*f.args)
        assert str(f) == str(ff)
        assert f == ff

    def test_params_property(self, q, kx):
        assert q('*').params == ()
        assert q('*[;2]').params == ()
        assert q('{[x;y;thethirdarg] x+y*thethirdarg}').params \
            == ('x', 'y', 'thethirdarg')
        assert q('{x+y*z}[;12]').params == ('x', 'z')
        assert q('{x*x} neg {[argarg] argarg} @').params == ('argarg',)
        assert q('{[a;b] a+b}').scan.params == ('a', 'b')
        assert q.Q.ajf0.params == ('f', 'g', 'x', 'y', 'z')
        # Ensure the param names of converted functions don't begin with the "PyKXParam" prefix
        assert kx.K(sorted).params == ('iterable', 'key', 'reverse')

    def test_func_property(self, kx, q):
        f1 = q('{x*x}')
        assert f1 == f1.func
        assert q('mod[;10]').func == q('mod')
        assert q('flip reverse @').func == q('reverse')
        f2 = q('{[a;b] a+b}')
        assert f2.scan.func == f2

        f3 = q.pykx.modpow
        assert f3(10, 2, 19) == f3(10, 2, 19) == 5

        f4 = q.pykx.i.pyfunc
        assert isinstance(f4, kx.SymbolicFunction)
        assert not isinstance(f4, kx.Foreign)
        assert isinstance(f4.func, kx.Foreign)

    def test_dotted_adverbs(self, q):
        assert all(q('in').each(np.array([1, 2, 3]), q('(1 0 1;til 10;5 6 7)'))
                   == q('110b'))
        assert q('+').over(np.array([1, 2, 3, 4])).py() == 10
        assert q('+').scan(np.array([1, 2, 3, 4])).py() == [1, 3, 6, 10]
        assert list(q('-').each_prior(np.array([1, 1, 2, 3, 5, 8, 13]))) \
            == list(q('-').prior(np.array([1, 1, 2, 3, 5, 8, 13]))) \
            == [1, 0, 1, 1, 2, 3, 5]
        assert all(all(x) for x in q(',').each_right(q('"ab"'), q('"XY"'))
                   == q(',').sv(q('"ab"'), q('"XY"')))
        assert q(',').sv(q('"ab"'), q('"XY"')).py() == [b'abX', b'abY']
        assert all(all(x) for x in q(',').each_left(q('"ab"'), q('"XY"'))
                   == q(',').vs(q('"ab"'), q('"XY"')))
        assert q(',').vs(q('"ab"'), q('"XY"')).py() == [b'aXY', b'bXY']

    def test_nested_error(self, kx, q):
        try:
            q('{x[y;z]}', lambda x, y: x.py() + y.py(), 'sym', 2)
        except Exception as e:
            ex = e
        else:
            raise AssertionError
        assert isinstance(ex, kx.QError)
        assert isinstance(ex.__cause__, TypeError)

    def test_symbolic_function(self, kx, q, q_port):
        f1 = q.Q.dpft

        assert f1.t == -11
        assert isinstance(f1, kx.SymbolAtom)
        assert isinstance(f1, kx.Function)
        assert isinstance(f1, kx.SymbolicFunction)

        assert isinstance(f1.sym, kx.SymbolAtom)
        assert f1.sym == '.Q.dpft'
        assert f1 != '.Q.dpft'
        assert f1 == f1
        assert f1 == f1.with_execution_ctx(q)
        assert f1 >= f1
        assert f1 <= f1
        assert not f1 > f1
        assert not f1 < f1
        assert not f1 != f1
        assert f1
        assert f1 == f1.py() == f1.np() == f1.pd() == f1.pd()
        assert str(f1) == '.Q.dpft'
        assert bytes(f1) == b'.Q.dpft'

        f2 = kx.SymbolicFunction('.vvv.f')
        q('.vvv.f:{testAlias::1b}')

        assert f1 != f2
        assert f2 == f2

        # By default, the execution context for the symbolic function is embedded q:
        q('testAlias::0b')
        assert not q('testAlias')
        f2()
        assert q('testAlias')

        # We can get a symbolic function for the same symbol with a different execution context:
        with kx.QConnection(port=q_port) as conn:
            f3 = f2.with_execution_ctx(conn)
            assert f2 != f3
            assert f2.sym == f3.sym == '.vvv.f'
            with pytest.raises(kx.QError, match='.vvv.f'):
                f3()
            conn('.vvv.f:{testAlias::1b}')
            conn('testAlias::0b')
            assert not conn('testAlias')
            f3()
            assert conn('testAlias')


# Conversions of nested K lists requires a license, We need to be able to call
# __getitem__ on the list to get correctly typed Numpy arrays to use.
@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs(kx, q):
    a = kx.toq(range(-10, 11))
    b = a.np()[:]
    c = np.multiply(a, 2)
    assert type(c) == type(a)
    assert all(c.np() == np.multiply(b, 2))

    a = kx.toq([float(x) / 10 for x in range(-10, 11)], ktype=kx.FloatVector)
    b = a.np()[:]
    c = np.sin(a)
    assert type(c) == type(a)
    assert all(c.np() == np.sin(b))

    a = q('3 0N#999?1f')
    b = q('3 0N#999?1f')
    anp = a.__typed_array__()
    bnp = b.__typed_array__()
    assert np.all(np.add(a, b) == np.add(anp, bnp))

    a = q('3 0N#999?1f')
    b = q('0N 3#999?1f')
    anp = a.__typed_array__()
    bnp = b.__typed_array__()
    assert np.all(np.matmul(a, b) == np.matmul(anp, bnp))

    a = [[1, 2], [3, 4], [5, 6]]
    b = [[1, 2, 3], [4, 5, 6]]
    assert np.all(np.matmul(kx.toq(a), kx.toq(b), dtype=kx.FloatVector).__typed_array__()
                  == np.matmul(a, b).astype(np.float32))

    a = q('((1 2); (3 4); (5))')
    b = q('((1 2 3); (4 5 6))')
    with pytest.raises(kx.QError):
        np.matmul(a, b)

    a = q('((1 2); (3 4); (5; (6; 7)))')
    b = q('((1 2 3); (4 5 6))')
    with pytest.raises(kx.QError):
        np.matmul(a, b)

    a = q('((1 2); (3 4); (5; 6))')
    b = q('((1 2 3); (4; (5 6)))')
    with pytest.raises(kx.QError):
        np.matmul(a, b)

    a = q('((1 2); (3 4); (5 6f))')
    b = q('((1 2 3); (4 5 6))')
    with pytest.raises(kx.QError):
        np.matmul(a, b)


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs_dtype(kx, q):
    a = q('1 2 3')
    b = q('4 5 6')

    assert np.add(a, b, dtype=np.int32).dtype == np.int32
    assert np.add(a, b, dtype=np.int64).dtype == np.int64
    assert np.add(a, b, dtype=np.float32).dtype == np.float32
    assert np.add(a, b, dtype=np.float64).dtype == np.float64

    assert isinstance(np.add(a, b, dtype=kx.FloatVector), kx.FloatVector)
    assert isinstance(np.add(a, b, dtype=kx.LongVector), kx.LongVector)
    assert isinstance(np.add(a, b, dtype=kx.IntVector), kx.IntVector)
    assert isinstance(np.add(a, b, dtype=kx.ShortVector), kx.ShortVector)

    assert np.add.reduce(a, dtype=np.int32).dtype == np.int32
    assert np.add.reduce(a, dtype=np.int64).dtype == np.int64
    assert np.add.reduce(a, dtype=np.float32).dtype == np.float32
    assert np.add.reduce(a, dtype=np.float64).dtype == np.float64

    assert isinstance(np.add.reduce(a, dtype=kx.FloatAtom), kx.FloatAtom)
    assert isinstance(np.add.reduce(a, dtype=kx.LongAtom), kx.LongAtom)
    assert isinstance(np.add.reduce(a, dtype=kx.IntAtom), kx.IntAtom)
    assert isinstance(np.add.reduce(a, dtype=kx.ShortAtom), kx.ShortAtom)


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs_out(kx, q):
    a = q('1 2 3')
    b = q('4 5 6')
    c = q('0 0 0')

    np.add(a, b, out=c)
    assert all(c.__typed_array__() == np.add(a.__typed_array__(), b.__typed_array__()))

    c = q('0 0 0').np()
    np.add(a, b, out=c)
    assert all(c == np.add(a.__typed_array__(), b.__typed_array__()))

    np.add(a, b, out=c, dtype=np.float32)
    assert all(c == np.add(a.__typed_array__(), b.__typed_array__()).astype(np.float32))

    a = [[1, 2], [3, 4], [5, 6]]
    b = [[1, 2, 3], [4, 5, 6]]
    c = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    ak = kx.toq(a)
    bk = kx.toq(b)
    np.matmul(ak, bk, out=c)
    assert np.all(c == np.matmul(a, b))

    c = q('0 0 0')
    d = c.np()[:]
    with pytest.raises(ValueError):
        np.add(a, b, out=(c, d))

    a = q('3 0N#999?1f')
    b = q('0N 3#999?1f')
    with pytest.raises(ValueError):
        np.matmul(a, b, out=a)


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs_accumulate(kx, q):
    a = kx.toq(range(10))
    b = a.np()[:]
    c = np.add.accumulate(a)
    assert type(c) == type(a)
    assert all(c.np() == np.add.accumulate(b))

    a = q('((1; 0); (0; 1))')
    b = np.eye(2)
    assert isinstance(a, kx.K)
    assert np.all(np.add.accumulate(a, 0).__typed_array__() == np.add.accumulate(b, 0))
    assert np.all(np.add.accumulate(a, 1).__typed_array__() == np.add.accumulate(b, 1))


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs_at(kx, q):
    a = kx.toq(range(1, 5))
    b = a.np()[:]
    np.negative.at(a, kx.toq(range(2)))
    np.negative.at(b, [0, 1])
    assert isinstance(a, kx.K)
    assert all(a.np() == b)

    a = kx.toq(range(1, 5))
    b = a.np()[:]
    np.add.at(a, q('0 1 2 2'), 1)
    np.add.at(b, [0, 1, 2, 2], 1)
    assert isinstance(a, kx.K)
    assert all(a.np() == b)

    a = kx.toq(range(1, 5))
    a_2 = kx.toq(range(1, 3))
    b = a.np()[:]
    b_2 = a_2.np()[:]
    np.add.at(a, q('0 1'), a_2)
    np.add.at(b, [0, 1], b_2)
    assert isinstance(a, kx.K)
    assert all(a.np() == b)

    with pytest.raises(TypeError):
        np.add.at(a)


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs_outer(kx, q):
    a = kx.toq(range(1, 4))
    a_2 = kx.toq(range(4, 7))
    b = a.np()[:]
    b_2 = a_2.np()[:]
    c = np.multiply.outer(a, a_2)
    assert isinstance(c, kx.K)
    assert np.all(c.__typed_array__() == np.multiply.outer(b, b_2))

    with pytest.raises(TypeError):
        np.multiply.outer(a)


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs_reduce(kx, q):
    a = kx.toq(range(1, 10))
    b = a.np()[:]
    c = np.multiply.reduce(a)
    assert isinstance(c, kx.K)
    assert c.np() == np.multiply.reduce(b)

    a = np.arange(8).reshape((2, 2, 2))
    b = q('(((0 1); (2 3)); ((4 5); (6 7)))')
    c = np.add.reduce(b, 0)
    assert isinstance(c, kx.K)
    assert np.all(c.__typed_array__() == np.add.reduce(a, 0))
    c = np.add.reduce(b, 1)
    assert isinstance(c, kx.K)
    assert np.all(c.__typed_array__() == np.add.reduce(a, 1))
    c = np.add.reduce(b, 2)
    assert isinstance(c, kx.K)
    assert np.all(c.__typed_array__() == np.add.reduce(a, 2))

    assert np.add.reduce([10], initial=5) == np.add.reduce(q('enlist 10'), initial=q('5'))

    assert np.all(np.minimum.reduce([[1, 2], [3, 4]], initial=10, where=[True, False])
                  == np.minimum.reduce(q('((1 2f); (3 4f))'),
                                       initial=q('10f'),
                                       where=q('(1b; 0b)')))


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_ufuncs_reduceat(kx, q):
    a = kx.toq(np.linspace(0, 15, 16).reshape(4, 4))
    a_2 = q('0 3 1 2 0')
    b = np.linspace(0, 15, 16).reshape(4, 4)
    b_2 = a_2.np()[:]
    c = np.add.reduceat(a, a_2)
    assert type(c) == type(a)
    assert np.all(c.__typed_array__() == np.add.reduceat(b, b_2))

    a = kx.toq(np.linspace(0, 15, 16).reshape(4, 4))
    a_2 = q('0 3')
    b = np.linspace(0, 15, 16).reshape(4, 4)
    b_2 = a_2.np()[:]
    c = np.multiply.reduceat(a, a_2, q('1'))
    assert type(c) == type(a)
    assert np.all(c.__typed_array__() == np.multiply.reduceat(b, b_2, 1))

    a = q('til 8')
    b = np.arange(8)
    c = np.add.reduceat(a, [0, 4, 1, 5, 2, 6, 3, 7])
    assert all(c.np()[::2] == np.add.reduceat(b, [0, 4, 1, 5, 2, 6, 3, 7])[::2])

    with pytest.raises(TypeError):
        np.add.reduceat(a)


# Conversions of nested K lists requires a license, We need to be able to call
# __getitem__ on the list to get correctly typed Numpy arrays to use.
@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_functions(kx, q):
    a = q('100?100')
    b = a.np()[:]
    c = np.sum(a)
    assert c.dtype == np.int64
    assert c == np.sum(b)

    a = q('1 - 100?2f')
    b = a.np()[:]
    c = np.lib.scimath.arcsin(a)
    assert np.all(c == np.lib.scimath.arcsin(b))

    a = [[1, 2], [3, 4], [5, 6]]
    b = [[1, 2, 3], [4, 5, 6]]
    assert np.all(np.dot(a, b) == np.dot(kx.toq(a), kx.toq(b)))


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.wrappers), list)
    assert sorted(dir(kx.wrappers)) == dir(kx.wrappers)


@pytest.mark.licensed
@pytest.mark.nep49
def test_numpy_equals(kx, q):
    if version.parse(np.__version__) < version.parse("1.25.0"):
        assert False == (np.array([[1, 2, 3], [4, 5, 6]]) == q('(1 1 1;2 2 5 2)')).py()
        assert False == (np.array([[1, 2, 3], [4, 5, 6]]) == q('(1 1 1 7;2 2 5 2)')).py()
    else:
        with pytest.raises(kx.QError):
            assert np.array([[1, 2, 3], [4, 5, 6]]) == q('(1 1 1;2 2 5 2)')
        with pytest.raises(ValueError):
            assert np.array([[1, 2, 3], [4, 5, 6]]) == q('(1 1 1 7;2 2 5 2)')


@pytest.mark.licensed
def test_attributes_vector(kx, q):
    assert '`s#' in str(q('til 10').sorted())
    assert '`u#' in str(q('til 10').unique())
    assert '`p#' in str(q('til 10').parted())
    assert '`g#' in str(q('til 10').grouped())

    with pytest.raises(kx.QError):
        q('2 45 1 3 9 8 2').sorted()

    with pytest.raises(kx.QError):
        q('2 45 1 3 9 8 2').unique()

    with pytest.raises(kx.QError):
        q('2 2 3 3 4 4 2').parted()


@pytest.mark.licensed
def test_attributes_table(kx, q):
    tab = q('([] til 10; 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.sorted()
    assert q.meta(tab).py()[('x',)]['a'] == 's'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.sorted('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 's'
    assert q.meta(tab).py()[('x1',)]['a'] == 's'
    tab = q('([] til 10; 10 + til 10)')
    tab.sorted(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 's'
    assert q.meta(tab).py()[('x1',)]['a'] == 's'

    tab = q('([] til 10; 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.unique()
    assert q.meta(tab).py()[('x',)]['a'] == 'u'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.unique('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 'u'
    assert q.meta(tab).py()[('x1',)]['a'] == 'u'
    tab = q('([] til 10; 10 + til 10)')
    tab.unique(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 'u'
    assert q.meta(tab).py()[('x1',)]['a'] == 'u'

    tab = q('([] til 10; 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.parted()
    assert q.meta(tab).py()[('x',)]['a'] == 'p'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.parted('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 'p'
    assert q.meta(tab).py()[('x1',)]['a'] == 'p'
    tab = q('([] til 10; 10 + til 10)')
    tab.parted(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 'p'
    assert q.meta(tab).py()[('x1',)]['a'] == 'p'

    tab = q('([] til 10; 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.grouped()
    assert q.meta(tab).py()[('x',)]['a'] == 'g'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.grouped('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 'g'
    assert q.meta(tab).py()[('x1',)]['a'] == 'g'
    tab = q('([] til 10; 10 + til 10)')
    tab.grouped(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 'g'
    assert q.meta(tab).py()[('x1',)]['a'] == 'g'

    tab = q('([] 1 2 5 3 2 6 4; 8 8 7 8 6 5 7)')

    with pytest.raises(kx.QError):
        tab.sorted(['x', 'x1'])

    with pytest.raises(kx.QError):
        tab.unique(['x', 'x1'])

    with pytest.raises(kx.QError):
        tab.parted(['x', 'x1'])


@pytest.mark.licensed
def test_attributes_keyed_table(kx, q):
    tab = q('([til 10] x1: 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.sorted()
    assert q.meta(tab).py()[('x',)]['a'] == 's'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.sorted('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 's'
    assert q.meta(tab).py()[('x1',)]['a'] == 's'
    tab = q('([til 10] x1: 10 + til 10)')
    tab.sorted(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 's'
    assert q.meta(tab).py()[('x1',)]['a'] == 's'

    tab = q('([til 10] x1: 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.unique()
    assert q.meta(tab).py()[('x',)]['a'] == 'u'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.unique('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 'u'
    assert q.meta(tab).py()[('x1',)]['a'] == 'u'
    tab = q('([til 10] x1: 10 + til 10)')
    tab.unique(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 'u'
    assert q.meta(tab).py()[('x1',)]['a'] == 'u'

    tab = q('([til 10] x1: 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.parted()
    assert q.meta(tab).py()[('x',)]['a'] == 'p'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.parted('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 'p'
    assert q.meta(tab).py()[('x1',)]['a'] == 'p'
    tab = q('([til 10] x1: 10 + til 10)')
    tab.parted(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 'p'
    assert q.meta(tab).py()[('x1',)]['a'] == 'p'

    tab = q('([til 10] x1: 10 + til 10)')
    assert q.meta(tab).py()[('x',)]['a'] == ''
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.grouped()
    assert q.meta(tab).py()[('x',)]['a'] == 'g'
    assert q.meta(tab).py()[('x1',)]['a'] == ''
    tab.grouped('x1')
    assert q.meta(tab).py()[('x',)]['a'] == 'g'
    assert q.meta(tab).py()[('x1',)]['a'] == 'g'
    tab = q('([til 10] x1: 10 + til 10)')
    tab.grouped(['x', 'x1'])
    assert q.meta(tab).py()[('x',)]['a'] == 'g'
    assert q.meta(tab).py()[('x1',)]['a'] == 'g'

    tab = q('([1 2 5 3 2 6 4] x1: 8 8 7 8 6 5 7)')

    with pytest.raises(kx.QError):
        tab.sorted(['x', 'x1'])

    with pytest.raises(kx.QError):
        tab.unique(['x', 'x1'])

    with pytest.raises(kx.QError):
        tab.parted(['x', 'x1'])


def test_apply_vector(q, kx):
    longvec = q('til 10')
    assert (longvec.apply(lambda x: x+1) == q('1+til 10')).all()
    assert longvec.apply(q.sum) == q('45')

    def func(x):
        return x+1
    assert (longvec.apply(func) == q('1+til 10')).all()
    assert longvec.apply(np.sum) == q('45')

    def func_args(x, y):
        return x+y
    assert (longvec.apply(func_args, 2) == q('2+til 10')).all()
    assert (longvec.apply(q('{x+y}'), 2) == q('2+til 10')).all()

    guidvec = q('-10?0Ng')
    assert 10 == len(guidvec.apply(np.unique))
    assert 10 == len(guidvec.apply(q.distinct))

    def func(x):
        return q.string(x)
    assert isinstance(guidvec.apply(func), kx.List)

    with pytest.raises(RuntimeError):
        longvec.apply(10)

    with pytest.raises(kx.QError):
        longvec.apply(q('{x+y}'), y=1)


def checkHTML(tab):
    html = tab._repr_html_()
    return (html.count('<tr>'), html.count('<th>'), html.count('<td>'))


@pytest.mark.licensed
def test_repr_html(kx, q):
    H = 10
    W = 20
    q.system.console_size = [H, W]

    # Many datatypes
    q('t:flip {(`$/:t)!2#/:(t:{x where not x in " z"}.Q.t)$\\:()}[]')
    q('T:flip {(`$/:upper t)!2#/:enlist each 2#/:(t:{x where not x in " z"}.Q.t)$\\:()}[]')
    q('wideManyDatatypes:t,\'T')
    tab = q('wideManyDatatypes')

    # (rows, headers, details)
    assert (3, 44, 40) == checkHTML(tab)

    # Single column table
    q('singleColTab:([] a:.z.d-til 2000)')
    tab = q('0#singleColTab')
    assert (0, 2, 0) == checkHTML(tab)
    tab = q('1#singleColTab')
    assert (2, 5, 1) == checkHTML(tab)
    tab = q('2#singleColTab')
    assert (3, 6, 2) == checkHTML(tab)
    tab = q('10#singleColTab')
    assert (H, 13, 9) == checkHTML(tab)
    tab = q('11#singleColTab')
    assert (H+1, 14, 10) == checkHTML(tab)

    # Multi column table
    q('multiColTab:([] a:.z.d-til 2000; sym:2000?`7)')
    tab = q('0#multiColTab')
    assert (0, 3, 0) == checkHTML(tab)
    tab = q('1#multiColTab')
    assert (2, 7, 2) == checkHTML(tab)
    tab = q('2#multiColTab')
    assert (3, 8, 4) == checkHTML(tab)
    tab = q('10#multiColTab')
    assert (H, 15, 18) == checkHTML(tab)
    tab = q('11#multiColTab')
    assert (H+1, 16, 20) == checkHTML(tab)

    q('n:-1+last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('0#extraWide')
    assert (0, 20, 0) == checkHTML(tab)
    tab = q('1#extraWide')
    assert (2, 41, 19) == checkHTML(tab)
    tab = q('2#extraWide')
    assert (3, 42, 38) == checkHTML(tab)
    tab = q('10#extraWide')
    assert (H, 49, 171) == checkHTML(tab)
    tab = q('11#extraWide')
    assert (H+1, 50, 190) == checkHTML(tab)

    q('n:last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('0#extraWide')
    assert (0, 21, 0) == checkHTML(tab)
    tab = q('1#extraWide')
    assert (2, 43, 20) == checkHTML(tab)
    tab = q('2#extraWide')
    assert (3, 44, 40) == checkHTML(tab)
    tab = q('10#extraWide')
    assert (H, 51, 180) == checkHTML(tab)
    tab = q('11#extraWide')
    assert (H+1, 52, 200) == checkHTML(tab)

    q('n:1+last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('0#extraWide')
    assert (0, 22, 0) == checkHTML(tab)
    tab = q('1#extraWide')
    assert (2, 45, 21) == checkHTML(tab)
    tab = q('2#extraWide')
    assert (3, 46, 42) == checkHTML(tab)
    tab = q('10#extraWide')
    assert (H, 53, 189) == checkHTML(tab)
    tab = q('11#extraWide')
    assert (H+1, 54, 210) == checkHTML(tab)

    q('n:50+last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('0#extraWide')
    assert (0, 22, 0) == checkHTML(tab)
    tab = q('1#extraWide')
    assert (2, 45, 21) == checkHTML(tab)
    tab = q('2#extraWide')
    assert (3, 46, 42) == checkHTML(tab)
    tab = q('10#extraWide')
    assert (H, 53, 189) == checkHTML(tab)
    tab = q('11#extraWide')
    assert (H+1, 54, 210) == checkHTML(tab)

    # Many keys
    tab = q('(-1+last system"c")!0#extraWide')
    assert (1, 42, 0) == checkHTML(tab)
    tab = q('(-1+last system"c")!1#extraWide')
    assert (2, 61, 2) == checkHTML(tab)
    tab = q('(-1+last system"c")!2#extraWide')
    assert (3, 80, 4) == checkHTML(tab)
    tab = q('(-1+last system"c")!10#extraWide')
    assert (H, 213, 18) == checkHTML(tab)
    tab = q('(-1+last system"c")!11#extraWide')
    assert (H+1, 232, 20) == checkHTML(tab)

    tab = q('(last system"c")!0#extraWide')
    assert (1, 42, 0) == checkHTML(tab)
    tab = q('(last system"c")!1#extraWide')
    assert (2, 61, 2) == checkHTML(tab)
    tab = q('(last system"c")!2#extraWide')
    assert (3, 80, 4) == checkHTML(tab)
    tab = q('(last system"c")!10#extraWide')
    assert (H, 213, 18) == checkHTML(tab)
    tab = q('(last system"c")!11#extraWide')
    assert (H+1, 232, 20) == checkHTML(tab)

    tab = q('(1+last system"c")!0#extraWide')
    assert (1, 42, 0) == checkHTML(tab)
    tab = q('(1+last system"c")!1#extraWide')
    assert (2, 61, 2) == checkHTML(tab)
    tab = q('(1+last system"c")!2#extraWide')
    assert (3, 80, 4) == checkHTML(tab)
    tab = q('(1+last system"c")!10#extraWide')
    assert (H, 213, 18) == checkHTML(tab)
    tab = q('(1+last system"c")!11#extraWide')
    assert (H+1, 232, 20) == checkHTML(tab)

    # Dictionaries
    assert '<p>Empty pykx.Dictionary: ' == q('()!()')._repr_html_()[:26]

    dict = q('(enlist `b)!(enlist 2)')
    assert (2, 5, 1) == checkHTML(dict)
    dict = q('(`a`b)!(1 2)')
    assert (3, 6, 2) == checkHTML(dict)
    dict = q('(10?`6)!(til 10)')
    assert (11, 14, 10) == checkHTML(dict)
    dict = q('(11?`6)!(til 11)')
    assert (11, 14, 10) == checkHTML(dict)

    dict = q('(enlist `b)!([] a:enlist 1)')
    assert (2, 5, 1) == checkHTML(dict)
    dict = q('(enlist `b)!([] a:enlist 1;b:enlist 2)')
    assert (2, 7, 2) == checkHTML(dict)
    dict = q('(enlist `b)!([] a:enlist 1;b:enlist 2)')
    assert (2, 7, 2) == checkHTML(dict)

    dict = q('(9?`6)!flip (`$"col",/:string til 19)!(19#enlist til 9)')
    assert (10, 49, 171) == checkHTML(dict)
    dict = q('(10?`6)!flip (`$"col",/:string til 20)!(20#enlist til 10)')
    assert (11, 52, 200) == checkHTML(dict)
    dict = q('(11?`6)!flip (`$"col",/:string til 21)!(21#enlist til 11)')
    assert (11, 52, 200) == checkHTML(dict)

    # Single Key
    q('singleKeyTab:`sym xkey ([] a:.z.d-til 2000; sym:2000?`7)')
    tab = q('0#singleKeyTab')
    assert (1, 4, 0) == checkHTML(tab)
    tab = q('1#singleKeyTab')
    assert (2, 5, 1) == checkHTML(tab)
    tab = q('2#singleKeyTab')
    assert (3, 6, 2) == checkHTML(tab)
    tab = q('10#singleKeyTab')
    assert (H, 13, 9) == checkHTML(tab)
    tab = q('11#singleKeyTab')
    assert (H+1, 14, 10) == checkHTML(tab)

    # Multi Key
    q('multiKeyTab:`sym`blah xkey ([] a:.z.d-til 2000; sym:2000?`7;blah:-2000?1000000)')
    tab = q('0#multiKeyTab')
    assert (1, 6, 0) == checkHTML(tab)
    tab = q('1#multiKeyTab')
    assert (2, 8, 1) == checkHTML(tab)
    tab = q('2#multiKeyTab')
    assert (3, 10, 2) == checkHTML(tab)
    tab = q('10#multiKeyTab')
    assert (H, 24, 9) == checkHTML(tab)
    tab = q('11#multiKeyTab')
    assert (H+1, 26, 10) == checkHTML(tab)

    # Single column splay table
    tab = q('{x set 0#([] a:.z.d-til 2000);get x}`:singleColSplay/')
    assert (0, 2, 0) == checkHTML(tab)
    tab = q('{x set 1#([] a:.z.d-til 2000);get x}`:singleColSplay/')
    assert (2, 5, 1) == checkHTML(tab)
    tab = q('{x set 2#([] a:.z.d-til 2000);get x}`:singleColSplay/')
    assert (3, 6, 2) == checkHTML(tab)
    tab = q('{x set 10#([] a:.z.d-til 2000);get x}`:singleColSplay/')
    assert (H, 13, 9) == checkHTML(tab)
    tab = q('{x set 11#([] a:.z.d-til 2000);get x}`:singleColSplay/')
    assert (H+1, 14, 10) == checkHTML(tab)

    # Multi column splay
    tab = q('{x set 0#([] a:.z.d-til 2000; b:til 2000);get x}`:multiColSplay/')
    assert (0, 3, 0) == checkHTML(tab)
    tab = q('{x set 1#([] a:.z.d-til 2000; b:til 2000);get x}`:multiColSplay/')
    assert (2, 7, 2) == checkHTML(tab)
    tab = q('{x set 2#([] a:.z.d-til 2000; b:til 2000);get x}`:multiColSplay/')
    assert (3, 8, 4) == checkHTML(tab)
    tab = q('{x set 10#([] a:.z.d-til 2000; b:til 2000);get x}`:multiColSplay/')
    assert (H, 15, 18) == checkHTML(tab)
    tab = q('{x set 11#([] a:.z.d-til 2000; b:til 2000);get x}`:multiColSplay/')
    assert (H+1, 16, 20) == checkHTML(tab)

    q('n:-1+last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('{x set y;get x}[`:multiColSplay/]0#extraWide')
    assert (0, 20, 0) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]1#extraWide')
    assert (2, 41, 19) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]2#extraWide')
    assert (3, 42, 38) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]10#extraWide')
    assert (H, 49, 171) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]11#extraWide')
    assert (H+1, 50, 190) == checkHTML(tab)

    q('n:last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('{x set y;get x}[`:multiColSplay/]0#extraWide')
    assert (0, 21, 0) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]1#extraWide')
    assert (2, 43, 20) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]2#extraWide')
    assert (3, 44, 40) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]10#extraWide')
    assert (H, 51, 180) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]11#extraWide')
    assert (H+1, 52, 200) == checkHTML(tab)

    q('n:1+last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('{x set y;get x}[`:multiColSplay/]0#extraWide')
    assert (0, 22, 0) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]1#extraWide')
    assert (2, 45, 21) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]2#extraWide')
    assert (3, 46, 42) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]10#extraWide')
    assert (H, 53, 189) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]11#extraWide')
    assert (H+1, 54, 210) == checkHTML(tab)

    q('n:50+last system"c";extraWide:flip (`$"col",/:string 1+til n)!n#enlist til 1000')
    tab = q('{x set y;get x}[`:multiColSplay/]0#extraWide')
    assert (0, 22, 0) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]1#extraWide')
    assert (2, 45, 21) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]2#extraWide')
    assert (3, 46, 42) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]10#extraWide')
    assert (H, 53, 189) == checkHTML(tab)
    tab = q('{x set y;get x}[`:multiColSplay/]11#extraWide')
    assert (H+1, 54, 210) == checkHTML(tab)

    # Syms and enums
    q('enums:`sym?`aa`cc`bb')
    q('symsEnums:([] a:(`aa;`aa`bb;enlist `aa`bb;first enums;enums;enlist enums))')
    q('symsEnums')._repr_html_()
    q('30#symsEnums')._repr_html_()
    q('symsEnums (),0')._repr_html_()
    q('symsEnums (),1')._repr_html_()
    q('symsEnums (),2')._repr_html_()
    q('symsEnums (),3')._repr_html_()
    q('symsEnums (),4')._repr_html_()
    q('symsEnums (),5')._repr_html_()
    q('delete sym from `.')
    q('symsEnums')._repr_html_()
    q('30#symsEnums')._repr_html_()
    q('symsEnums (),0')._repr_html_()
    q('symsEnums (),1')._repr_html_()
    q('symsEnums (),2')._repr_html_()
    q('symsEnums (),3')._repr_html_()
    q('symsEnums (),4')._repr_html_()
    q('symsEnums (),5')._repr_html_()

    q('`:symsEnumsSplay/ set .Q.en[`:.] symsEnums;get `:symsEnumsSplay/')._repr_html_()
    q('`:symsEnumsSplay/ set .Q.en[`:.] 30#symsEnums;get `:symsEnumsSplay/')._repr_html_()
    q('`:symsEnumsSplay/ set .Q.en[`:.] symsEnums (),0;get `:symsEnumsSplay/')._repr_html_()
    q('`:symsEnumsSplay/ set .Q.en[`:.] symsEnums (),1;get `:symsEnumsSplay/')._repr_html_()
    q('`:symsEnumsSplay/ set .Q.en[`:.] symsEnums (),2;get `:symsEnumsSplay/')._repr_html_()
    q('`:symsEnumsSplay/ set .Q.en[`:.] symsEnums (),3;get `:symsEnumsSplay/')._repr_html_()
    q('`:symsEnumsSplay/ set .Q.en[`:.] symsEnums (),4;get `:symsEnumsSplay/')._repr_html_()
    q('`:symsEnumsSplay/ set .Q.en[`:.] symsEnums (),5;get `:symsEnumsSplay/')._repr_html_()

    q('keyedSymsEnums:`b xkey update b:i from symsEnums')
    q('keyedSymsEnums')._repr_html_()
    q('30#keyedSymsEnums')._repr_html_()
    q('keyedSymsEnums (),0')._repr_html_()
    q('keyedSymsEnums (),1')._repr_html_()
    q('keyedSymsEnums (),2')._repr_html_()
    q('keyedSymsEnums (),3')._repr_html_()
    q('keyedSymsEnums (),4')._repr_html_()
    q('keyedSymsEnums (),5')._repr_html_()

    import os
    os.makedirs('HDB', exist_ok=True)
    os.chdir('HDB')

    # Partitioned syms and enums
    q('(`$":2001.01.02/partitionedTab/") set .Q.en[`:.] 0#symsEnums')
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] symsEnums;system"l .";partitionedTab'
      )._repr_html_()
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] 30#symsEnums;system"l .";partitionedTab'
      )._repr_html_()
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] symsEnums (),0;system"l .";partitionedTab'
      )._repr_html_()
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] symsEnums (),1;system"l .";partitionedTab'
      )._repr_html_()
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] symsEnums (),2;system"l .";partitionedTab'
      )._repr_html_()
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] symsEnums (),3;system"l .";partitionedTab'
      )._repr_html_()
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] symsEnums (),4;system"l .";partitionedTab'
      )._repr_html_()
    q('(`$":2001.01.01/partitionedTab/") set .Q.en[`:.] symsEnums (),5;system"l .";partitionedTab'
      )._repr_html_()

    # Partitioned
    q('(`$":2001.01.01/partitionedTab/") set 0#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (0, 3, 0) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 1#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (2, 7, 2) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 2#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (3, 8, 4) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 10#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (H+1, 16, 20) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 11#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (H+1, 16, 20) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 0#([] a:.z.d-til 2000;b:til 2000)')
    q('(`$":2001.01.02/partitionedTab/") set 0#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (0, 4, 0) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 0#([] a:.z.d-til 2000;b:til 2000)')
    q('(`$":2001.01.02/partitionedTab/") set 1#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (2, 9, 3) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 10#([] a:.z.d-til 2000;b:til 2000)')
    q('(`$":2001.01.02/partitionedTab/") set 10#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (H+1, 18, 30) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 1#([] a:.z.d-til 2000;b:til 2000)')
    q('(`$":2001.01.02/partitionedTab/") set 0#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (2, 9, 3) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 35#([] a:.z.d-til 2000;b:til 2000)')
    q('(`$":2001.01.02/partitionedTab/") set 0#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (H+1, 18, 30) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 0#([] a:.z.d-til 2000;b:til 2000)')
    q('(`$":2001.01.02/partitionedTab/") set 35#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (H+1, 18, 30) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 11#([] a:.z.d-til 2000;b:til 2000)')
    q('(`$":2001.01.02/partitionedTab/") set 11#([] a:.z.d-til 2000;b:til 2000)')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (H+1, 18, 30) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 0#extraWide')
    q('(`$":2001.01.02/partitionedTab/") set 0#extraWide')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (0, 22, 0) == checkHTML(tab)

    q('(`$":2001.01.01/partitionedTab/") set 5#extraWide')
    q('(`$":2001.01.02/partitionedTab/") set 0#extraWide')
    q('system"l ."')
    tab = q('partitionedTab')
    assert (6, 49, 105) == checkHTML(tab)

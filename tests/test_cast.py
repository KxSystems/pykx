from datetime import date, datetime, time, timedelta
import math
import random

import numpy as np
import pytest


@pytest.mark.unlicensed
def test_cast_to_short(kx):
    x = kx.ShortAtom(99, cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 99

    x = kx.ShortAtom(99.99, cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 99

    x = kx.ShortAtom('99', cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 99

    x = kx.ShortAtom(True, cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 1

    x = kx.ShortAtom(np.int8(99), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 99

    x = kx.ShortAtom(np.uint8(99), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 99

    x = kx.ShortAtom(np.int16(1234), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 1234

    x = kx.ShortAtom(np.uint16(1234), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 1234

    x = kx.ShortAtom(np.int32(12345), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 12345

    x = kx.ShortAtom(np.uint32(12345), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 12345

    x = kx.ShortAtom(np.int64(12345), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 12345

    x = kx.ShortAtom(np.uint64(12345), cast=True)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 12345

    with pytest.raises(OverflowError):
        kx.ShortAtom(np.int32(123456), cast=True)

    with pytest.raises(OverflowError):
        kx.ShortAtom(np.int64(123456), cast=True)

    with pytest.raises(TypeError) as exc:
        kx.ShortAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    x = kx.ShortAtom(3, cast=False)
    assert isinstance(x, kx.ShortAtom)
    assert x.py() == 3

    with pytest.raises(TypeError):
        kx.ShortAtom('3', cast=False)


@pytest.mark.unlicensed
def test_cast_to_int(kx):
    x = kx.IntAtom(99, cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 99

    x = kx.IntAtom(99.99, cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 99

    x = kx.IntAtom('99', cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 99

    x = kx.IntAtom(True, cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 1

    x = kx.IntAtom(np.int8(99), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 99

    x = kx.IntAtom(np.uint8(99), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 99

    x = kx.IntAtom(np.int16(1234), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 1234

    x = kx.IntAtom(np.uint16(1234), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 1234

    x = kx.IntAtom(np.int32(12345), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 12345

    x = kx.IntAtom(np.uint32(12345), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 12345

    x = kx.IntAtom(np.int64(12345), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 12345

    x = kx.IntAtom(np.uint64(12345), cast=True)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 12345

    with pytest.raises(OverflowError):
        kx.IntAtom(2**31, cast=True)

    with pytest.raises(OverflowError):
        kx.IntAtom(np.int64(2**31), cast=True)

    with pytest.raises(TypeError) as exc:
        kx.IntAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    x = kx.IntAtom(3, cast=False)
    assert isinstance(x, kx.IntAtom)
    assert x.py() == 3

    with pytest.raises(TypeError):
        kx.IntAtom('3', cast=False)


@pytest.mark.unlicensed
def test_cast_to_long(kx):
    x = kx.LongAtom(99, cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 99

    x = kx.LongAtom(99.99, cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 99

    x = kx.LongAtom('99', cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 99

    x = kx.LongAtom(True, cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 1

    x = kx.LongAtom(np.int8(99), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 99

    x = kx.LongAtom(np.uint8(99), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 99

    x = kx.LongAtom(np.int16(1234), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 1234

    x = kx.LongAtom(np.uint16(1234), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 1234

    x = kx.LongAtom(np.int32(12345), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 12345

    x = kx.LongAtom(np.uint32(12345), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 12345

    x = kx.LongAtom(np.int64(12345), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 12345

    x = kx.LongAtom(np.uint64(12345), cast=True)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 12345

    with pytest.raises(OverflowError):
        kx.LongAtom(2**63, cast=True)

    with pytest.raises(TypeError) as exc:
        kx.LongAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    x = kx.LongAtom(3, cast=False)
    assert isinstance(x, kx.LongAtom)
    assert x.py() == 3

    with pytest.raises(TypeError):
        kx.LongAtom('3', cast=False)


@pytest.mark.unlicensed
def test_cast_to_real(kx):
    x = kx.RealAtom(math.pi, cast=True)
    assert isinstance(x, kx.RealAtom)
    assert math.isclose(x.py(), math.pi, abs_tol=1e-06)

    x = kx.RealAtom(99, cast=True)
    assert isinstance(x, kx.RealAtom)
    assert x.py() == 99

    x = kx.RealAtom('99.99', cast=True)
    assert isinstance(x, kx.RealAtom)
    assert math.isclose(x.py(), 99.99, abs_tol=1e-05)

    x = kx.RealAtom(np.int32(12345), cast=True)
    assert isinstance(x, kx.RealAtom)
    assert x.py() == 12345

    x = kx.RealAtom(np.int64(12345), cast=True)
    assert isinstance(x, kx.RealAtom)
    assert x.py() == 12345

    with pytest.raises(TypeError) as exc:
        kx.RealAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    x = kx.RealAtom(3.14, cast=False)
    assert isinstance(x, kx.RealAtom)
    assert math.isclose(x.py(), 3.14, abs_tol=1e-05)

    with pytest.raises(TypeError):
        kx.RealAtom('3.14', cast=False)


@pytest.mark.unlicensed
def test_cast_to_float(kx):
    x = kx.FloatAtom(math.pi, cast=True)
    assert isinstance(x, kx.FloatAtom)
    assert math.isclose(x.py(), math.pi, abs_tol=1e-15)

    x = kx.FloatAtom(99, cast=True)
    assert isinstance(x, kx.FloatAtom)
    assert x.py() == 99

    x = kx.FloatAtom('99.99', cast=True)
    assert isinstance(x, kx.FloatAtom)
    assert math.isclose(x.py(), 99.99, abs_tol=1e-15)

    x = kx.FloatAtom(np.int32(12345), cast=True)
    assert isinstance(x, kx.FloatAtom)
    assert x.py() == 12345

    x = kx.FloatAtom(np.int64(12345), cast=True)
    assert isinstance(x, kx.FloatAtom)
    assert x.py() == 12345

    with pytest.raises(TypeError) as exc:
        kx.FloatAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    x = kx.FloatAtom(3.14, cast=False)
    assert isinstance(x, kx.FloatAtom)
    assert math.isclose(x.py(), 3.14, abs_tol=1e-15)

    with pytest.raises(TypeError):
        kx.FloatAtom('3.14', cast=False)


@pytest.mark.unlicensed
def test_cast_to_date(kx):
    dt = datetime.now()
    x = kx.DateAtom(dt, cast=True)
    assert isinstance(x, kx.DateAtom)
    assert x.py() == date.today()

    x = kx.MonthAtom(dt, cast=True)
    assert isinstance(x, kx.MonthAtom)
    assert x.py() == date(dt.year, dt.month, 1)

    ndt = np.datetime64(dt)
    x = kx.DateAtom(ndt, cast=True)
    assert isinstance(x, kx.DateAtom)
    assert x.py() == date.today()

    with pytest.raises(TypeError) as exc:
        kx.DateAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    d = date.today()
    x = kx.DateAtom(d, cast=False)
    assert isinstance(x, kx.DateAtom)
    assert x.py() == d

    with pytest.raises(TypeError):
        kx.DateAtom('foo', cast=False)


@pytest.mark.unlicensed
def test_cast_to_datetime(kx):
    d = date.today()
    x = kx.TimestampAtom(d, cast=True)
    assert isinstance(x, kx.TimestampAtom)
    assert x.py() == datetime.combine(d, time.min)

    dt = datetime.now()
    ndt = np.datetime64(dt)
    x = kx.TimestampAtom(ndt, cast=True)
    assert isinstance(x, kx.TimestampAtom)
    assert x.py() == dt

    with pytest.raises(TypeError) as exc:
        kx.TimestampAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    dt = datetime.now()
    x = kx.TimestampAtom(dt, cast=False)
    assert isinstance(x, kx.TimestampAtom)
    assert x.py() == dt

    with pytest.raises(TypeError):
        kx.TimestampAtom('foo', cast=False)


@pytest.mark.unlicensed
def test_cast_to_timedelta(kx):
    td = timedelta(3, 1234, 567890)
    x = kx.MinuteAtom(td, cast=True)
    assert isinstance(x, kx.MinuteAtom)
    assert x.py().total_seconds() == int(td.total_seconds() / 60) * 60.0

    x = kx.SecondAtom(td, cast=True)
    assert isinstance(x, kx.SecondAtom)
    assert x.py().total_seconds() == int(td.total_seconds())

    x = kx.TimeAtom(td, cast=True)
    assert isinstance(x, kx.TimeAtom)
    assert x.py().total_seconds() == int(td.total_seconds() * 1000) / 1000

    ntd = np.timedelta64(td)
    x = kx.TimespanAtom(ntd, cast=True)
    assert isinstance(x, kx.TimespanAtom)
    assert x.py() == td

    with pytest.raises(TypeError) as exc:
        kx.TimespanAtom('foo', cast=True)
        assert exc.value.startswith('Can not cast')

    td = timedelta(3, 1234, 567890)
    x = kx.TimespanAtom(td, cast=False)
    assert isinstance(x, kx.TimespanAtom)
    assert x.py() == td

    with pytest.raises(TypeError):
        kx.TimestampAtom('foo', cast=False)


@pytest.mark.unlicensed
def test_cast_numpy_ndarray_to_short_vector(kx):
    arr = np.array([random.randint(-2**15, 2**15-1) for _ in range(1000)])
    x = kx.ShortVector(arr, cast=True)
    assert isinstance(x, kx.ShortVector)
    assert (x.np() == arr).all()

    arr = arr.astype('int16').astype(str)
    x = kx.ShortVector(arr, cast=True)
    assert isinstance(x, kx.ShortVector)
    assert (x.np() == arr.astype('int16')).all()

    arr = np.array([random.randint(-2**31, 2**31) for _ in range(1000)])
    with pytest.raises(OverflowError):
        kx.ShortVector(arr, cast=True)

    arr = np.array([random.randint(-2**15, 2**15-1) for _ in range(1000)], dtype='int16')
    x = kx.ShortVector(arr, cast=False)
    assert isinstance(x, kx.ShortVector)
    assert (x.np() == arr).all()

    with pytest.raises(TypeError):
        kx.ShortVector(arr.astype(str), cast=False)


@pytest.mark.unlicensed
def test_cast_numpy_ndarray_to_int_vector(kx):
    arr = np.array([random.randint(-2**31, 2**31-1) for _ in range(1000)])
    x = kx.IntVector(arr, cast=True)
    assert isinstance(x, kx.IntVector)
    assert (x.np() == arr).all()

    arr = arr.astype('int32').astype(str)
    x = kx.IntVector(arr, cast=True)
    assert isinstance(x, kx.IntVector)
    assert (x.np() == arr.astype('int32')).all()

    arr = np.array([random.randint(-2**63, 2**63-1) for _ in range(1000)])
    with pytest.raises(OverflowError):
        kx.IntVector(arr, cast=True)

    arr = np.array([random.randint(-2**31, 2*31-1) for _ in range(1000)], dtype='int32')
    x = kx.IntVector(arr, cast=False)
    assert isinstance(x, kx.IntVector)
    assert (x.np() == arr).all()

    with pytest.raises(TypeError):
        kx.IntVector(arr.astype(str), cast=False)


@pytest.mark.unlicensed
def test_cast_numpy_ndarray_to_long_vector(kx):
    arr = np.array([random.randint(-2**63, 2**63-1) for _ in range(1000)])
    x = kx.LongVector(arr, cast=True)
    assert isinstance(x, kx.LongVector)
    assert (x.np() == arr).all()

    arr = arr.astype(str)
    x = kx.LongVector(arr, cast=True)
    assert isinstance(x, kx.LongVector)
    assert (x.np() == arr.astype('int64')).all()

    arr = np.array([random.randint(-2**63, 2**63-1) + random.random() for _ in range(1000)])
    x = kx.LongVector(arr, cast=True)
    assert isinstance(x, kx.LongVector)
    assert (x.np() == arr.astype('int64')).all()

    with pytest.raises(OverflowError):
        arr = np.array([random.randint(-2**127, 2**127-1)])
        x = kx.LongVector(arr, cast=True)

    arr = np.array([random.randint(-2**63, 2**63-1) for _ in range(1000)], dtype='int64')
    x = kx.LongVector(arr, cast=False)
    assert isinstance(x, kx.LongVector)
    assert (x.np() == arr).all()

    with pytest.raises(TypeError):
        kx.LongVector(arr.astype(str), cast=False)


@pytest.mark.unlicensed
def test_cast_numpy_ndarray_to_real_vector(kx):
    arr = np.random.rand(1000)
    x = kx.RealVector(arr, cast=True)
    assert isinstance(x, kx.RealVector)
    assert np.allclose(x.np(), arr, atol=1e-6)

    arr = arr.astype(str)
    x = kx.RealVector(arr, cast=True)
    assert isinstance(x, kx.RealVector)
    assert np.allclose(x.np(), arr.astype(float), atol=1e-6)

    arr = np.random.rand(1000).astype('float32')
    x = kx.RealVector(arr, cast=False)
    assert isinstance(x, kx.RealVector)
    assert np.allclose(x.np(), arr, atol=1e-6)

    with pytest.raises(TypeError):
        kx.RealVector(arr.astype(str), cast=False)


@pytest.mark.unlicensed
def test_cast_numpy_ndarray_to_float_vector(kx):
    arr = np.random.rand(1000).astype('float32')
    x = kx.FloatVector(arr, cast=True)
    assert isinstance(x, kx.FloatVector)
    assert np.allclose(x.np(), arr, atol=1e-15)

    arr = arr.astype(str)
    x = kx.FloatVector(arr, cast=True)
    assert isinstance(x, kx.FloatVector)
    assert np.allclose(x.np(), arr.astype(float), atol=1e-15)

    arr = np.random.rand(1000).astype('float64')
    x = kx.FloatVector(arr, cast=False)
    assert isinstance(x, kx.FloatVector)
    assert np.allclose(x.np(), arr, atol=1e-15)

    with pytest.raises(TypeError):
        kx.FloatVector(arr.astype(str), cast=False)


@pytest.mark.unlicensed
def test_cast_numpy_ndarray_errors(kx):
    with pytest.raises(TypeError) as exc:
        arr = np.random.randint(0, int(datetime.now().timestamp()), size=10) * 1000000
        arr = arr.astype('datetime64[us]') \
                 .astype('datetime64[ns]') \
                 .astype('str')
        kx.TimestampVector(arr, cast=True)

        assert exc.value.startswith('Can not cast')

    with pytest.raises(TypeError) as exc:
        arr = np.array([object() for _ in range(10)])
        kx.LongVector(arr, cast=True)

        assert exc.value.startswith('Can not cast')
        assert exc.value.contains('TypeError("\'<\' not supported between instances of \'object\' and \'int\'"')  # noqa

    with pytest.raises(TypeError) as exc:
        arr = np.array([object() for _ in range(10)])
        kx.ByteVector(arr, cast=True)

        assert exc.value.startswith('Can not cast')
        assert exc.value.contains('TypeError("int() argument must be a string, a bytes-like object or a number, not \'object\'")')  # noqa


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.cast), list)
    assert sorted(dir(kx.cast)) == dir(kx.cast)

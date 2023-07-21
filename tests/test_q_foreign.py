import pytest

import sys
from platform import system

import numpy as np


@pytest.mark.isolate
def test_foreign_ref_count():
    import pykx as kx
    class A: # noqa
        def __init__(self, x, y):
            self.x = x
            self.y = y

    a = A(1, 2)
    start_ref_count = sys.getrefcount(a)
    a_foreign = kx.Foreign(a)
    assert sys.getrefcount(a) == start_ref_count + 1
    b_foreign = kx.toq(a, ktype=kx.Foreign)
    assert sys.getrefcount(a) == start_ref_count + 2
    c = b_foreign.py()
    assert sys.getrefcount(a) == start_ref_count + 3
    # These should increment and then decrement the ref counter as they fall out of scope right away
    b_foreign.py()
    b_foreign.py()
    assert sys.getrefcount(a) == start_ref_count + 3
    del a_foreign
    assert sys.getrefcount(a) == start_ref_count + 2
    del b_foreign
    assert sys.getrefcount(a) == start_ref_count + 1
    del c
    assert sys.getrefcount(a) == start_ref_count


@pytest.mark.isolate
def test_foreign_refcount_under_q():
    if system() != 'Windows':
        import pykx as kx
        class A: # noqa
            def __init__(self, x, y):
                self.x = x
                self.y = y
        a = A(1, 2)
        kx.q('{a:: .pykx.wrap x}', kx.Foreign(a))
        kx.q('b: 5')
        kx.q('refcount: -16!b')
        kx.q('c: b')
        assert kx.q('(refcount + 1i)~-16!b')
        kx.q('.pykx.setattr[a`.; `x; b]')
        assert kx.q('(refcount + 2i)~-16!b')
        kx.q('.pykx.setattr[a`.; `x; .pykx.topy b]')
        assert kx.q('(refcount + 1i)~-16!b')
        kx.q('c: 2')
        assert kx.q('(refcount)~-16!b')


@pytest.mark.unlicensed
def test_foreign_types(kx):
    class A:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def a_method(self):
            return self.x + self.y

    def foo(x):
        return x + 1

    def generator(i):
        x = 1
        for _ in range(i):
            x = x * 2
            yield x

    a = A(1, 2)

    # Numeric Types
    f_int = kx.Foreign(99999)
    f_float = kx.Foreign(3.14)
    f_complex = kx.Foreign(complex(2, 7))
    assert 99999 == f_int.py()
    assert 3.14 == f_float.py()
    assert complex(2, 7) == f_complex.py()

    # Sequence/Iterator Types
    f_generator = kx.Foreign(generator(10))
    f_list = kx.Foreign(list(range(10)))
    f_tuple = kx.Foreign((1, 2))
    f_range = kx.Foreign(range(10))
    gen = f_generator.py()
    assert 2 == next(gen)
    assert 4 == next(gen)
    assert list(range(10)) == f_list.py()
    assert (1, 2) == f_tuple.py()
    assert range(10) == f_range.py()

    # Text/Bytes Types
    f_str = kx.Foreign('hello')
    f_bytes = kx.Foreign(b'there')
    f_bytearray = kx.Foreign(bytearray(b'world'))
    assert 'hello' == f_str.py()
    assert b'there' == f_bytes.py()
    assert bytearray(b'world') == f_bytearray.py()

    # Set/Map Types
    f_set = kx.Foreign({'a', 'b', 'c'})
    f_dict = kx.Foreign({'a': 1, 'b': 2})
    assert {'a', 'b', 'c'} == f_set.py()
    assert {'a': 1, 'b': 2} == f_dict.py()

    # Other First class types
    f_class = kx.Foreign(A(1, 2))
    f_function= kx.Foreign(foo)
    f_method = kx.Foreign(a.a_method)
    f_lambda = kx.Foreign(lambda x: x + 1)
    f_type = kx.Foreign(type(1))
    f_None = kx.Foreign(None)
    f_bool = kx.Foreign(True)
    assert isinstance(f_class.py(), A)
    assert 2 == f_function.py()(1)
    assert 3 == f_method.py()()
    assert 2 == f_lambda.py()(1)
    assert type(1) == f_type.py()
    assert f_None.py() is None
    assert f_bool.py()


@pytest.mark.isolate
def test_foreign_under_q():
    import pykx as kx
    if system() != 'Windows':

        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.z = B(10)

        class B: # noqa
            def __init__(self, x):
                self.x = x

        assert kx.q('(x-1)~'
                    '{last .pykx.eval["lambda x:list(range(x))"][x]`} '
                    'each x:10000 + til 2*1')
        kx.q('{a:: x}', kx.Foreign(A(1, 2)))
        assert kx.q('1 ~ .pykx.toq .pykx.getattr[a; `x]')
        assert kx.q('2 ~ .pykx.toq .pykx.getattr[a; `y]')
        assert kx.q('112h ~ type .pykx.pyeval"1+1"')
        assert kx.q('2 ~ .pykx.toq .pykx.pyeval"1+1"')
        assert kx.q('10 ~ .pykx.toq .pykx.getattr[.pykx.getattr[a; `z]; `x]')
        kx.q('.pykx.setattr[.pykx.getattr[a; `z]; `x; 5]')
        assert kx.q('5 ~ .pykx.toq .pykx.getattr[.pykx.getattr[a; `z]; `x]')


@pytest.mark.isolate
def test_foreign_class_under_q():
    if system() != 'Windows':
        import pykx as kx

        class A:
            def __init__(self, x, y, dx, dy):
                self.x = x
                self.y = y
                self.dx = dx
                self.dy = dy

            def move(self):
                self.x += self.dx
                self.y += self.dy
                return 0

            def __repr__(self):
                return '({}, {})'.format(self.x, self.y)

        a = A(5, 0, 1, 2)
        kx.q('{af:: .pykx.wrap x}', kx.Foreign(a))
        assert kx.q('"(5, 0)"~.pykx.repr af`.')
        assert kx.q('5~af[`:x]`')
        kx.q('.pykx.setattr[af`.; `x; 0]')
        assert a.x == 0
        kx.q('amove: af`:move')
        assert kx.q('112h=type amove`.')
        kx.q('amove[::]')
        assert a.x == 1
        assert a.y == 2
        with pytest.raises(kx.QError):
            kx.q('af[`:r]`')
        kx.q('.pykx.setattr[af`.; `r; 5]')
        kx.q('5~af[`:r]`')
        with pytest.raises(kx.QError):
            kx.q('af[`:q][1; 2]')


@pytest.mark.isolate
def test_foreign_functions_under_q():
    if system() != 'Windows':
        import pykx as kx

        def maximum(x, y, *args):
            maxi = x.py()
            z = [y.py()]
            if args:
                for arg in args:
                    a = arg.py()
                    if isinstance(a, list) or isinstance(a, np.ndarray):
                        z.extend(a)
                    else:
                        z.append(a)
            else:
                maxi = x
                z = [y, *args]
            for item in z:
                if item > maxi:
                    maxi = item
            return maxi
        kx.q('{pymax:: .pykx.wrap x}', kx.Foreign(maximum))

        py_n = kx.q('n: 100?100; n').py()
        assert max(py_n) == kx.q('pymax[n 0; n 1; 2 _ n]`').py()
        assert kx.q('5~pymax[5; 1]`')
        assert kx.q('9~pymax[1; 3; 5; 9; 7; 8]`')
        with pytest.raises(kx.QError):
            kx.q('pymax[5]')

        def args_kwargs(*args, **kwargs):
            return (args, kwargs)
        kx.q('{argskwargs:: .pykx.wrap x}', kx.Foreign(args_kwargs))

        assert [['foo', 1, b'bar', 3], {'apple': 1, 'banana': b'bad', 'orange': 'good'}] == \
            kx.q('argskwargs[`foo; 1; "bar"; 3; `apple pykw 1;'
                 ' `banana pykw "bad"; `orange pykw `good]`').py()

        assert kx.q('argskwargs[`foo; 1; "bar"; 3; `apple pykw 1; `banana pykw "bad"; '
                    '`orange pykw `good]`').py() == \
            kx.q('argskwargs[pyarglist (`foo; 1; "bar"; 3); '
                 'pykwargs (`apple`banana`orange)!(1; "bad"; `good)]`').py()


@pytest.mark.isolate
def test_foreign_setattr_under_q(kx, q):
    if system() != 'Windows':
        import pykx as kx
        class A: # noqa
            def __init__(self, x):
                self.x = x
        a = A(0)
        kx.q('{af:: .pykx.wrap x}', kx.Foreign(a))
        assert a.x == 0
        kx.q('.pykx.setattr[af`.; `x; 5]')
        assert a.x == 5
        kx.q('.pykx.setattr[af`.; `x;.pykx.topy (1 2 3 4 5)]')
        assert a.x == [1, 2, 3, 4, 5]
        kx.q('.pykx.setattr[af`.;`x; .pykx.tonp (1 2 3 4 5)]')
        assert (a.x == kx.q('1 2 3 4 5').np()).all()
        kx.q('tab: ([] a:10?10; b:10?10)')
        kx.q('.pykx.setattr[af`.; `x; .pykx.topd tab]')
        assert a.x.equals(kx.q('tab').pd())


@pytest.mark.isolate
def test_foreign_setattr_arrow_under_q(q, kx, pa):
    if system() != 'Windows':
        import pykx as kx
        class A: # noqa
            def __init__(self, x):
                self.x = x
        a = A(0)
        kx.q('{af:: .pykx.wrap x}', kx.Foreign(a))
        assert a.x == 0
        kx.q('tab: ([] a:10?10; b:10?10)')
        kx.q('.pykx.setattr[af`.; `x;.pykx.topa tab]')
        assert a.x.to_pandas().equals(kx.q('tab').pa().to_pandas())

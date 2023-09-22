# Do not import pykx here - use the `kx` fixture instead!

import pytest


def test_float_vector_shape(q, kx):
    qrand = kx.random.random(10, 1.0)
    assert isinstance(qrand, kx.FloatVector)


def test_long_vector_shape(q, kx):
    qrand = kx.random.random(10, 10)
    assert isinstance(qrand, kx.LongVector)


def test_List_shape(q, kx):
    qrand = kx.random.random((5, 5), 1.0)
    assert isinstance(qrand, kx.List)


def test_without_seed(q, kx):
    rand1 = kx.random.random(25, 100.0)
    rand2 = kx.random.random(25, 100.0)
    assert len(rand1) == len(rand2)
    assert all([a != b for a, b in zip(rand1, rand2)])


def test_with_seed(q, kx):
    kx.random.seed(12345)
    rand1 = kx.random.random(25, 100.0)
    kx.random.seed(12345)
    rand2 = kx.random.random(25, 100.0)
    assert len(rand1) == len(rand2)
    assert all([a == b for a, b in zip(rand1, rand2)])


def test_with_seed_kwarg(q, kx):
    rand1 = kx.random.random(25, 100.0, seed=54321)
    rand2 = kx.random.random(25, 100.0, seed=54321)
    assert len(rand1) == len(rand2)
    assert all([a == b for a, b in zip(rand1, rand2)])


def test_random_from_list(q, kx):
    data = (2, 4, 6, 8, 10)
    rand = kx.random.random(25, data)
    assert all(a in data for a in rand)


def test_deal_uniqueness(q, kx):
    rand = kx.random.random(-10, 10)
    assert len(rand) == len(set(rand))


def test_deal_from_list(q, kx):
    data = (2, 4, 6, 8, 10)
    rand = kx.random.random(-3, data)
    assert len(rand) == len(set(rand))
    assert all(a in data for a in rand)


def test_nested_deal(q, kx):
    data = [1, 2, 3, 4, 5]
    rand = kx.random.random([-2, 2], data)
    rand_linear = rand.py()
    rand_linear = rand_linear[0]+rand_linear[1]
    assert all(a in data for a in rand_linear)
    assert len(rand_linear) == len(set(rand_linear))


def test_q_object_param(q, kx):
    rand1 = kx.random.random(kx.LongAtom(4), kx.FloatAtom(1.0), seed=kx.IntAtom(246))
    rand2 = kx.random.random(4, 1.0, seed=246)
    assert len(rand1) == len(rand2)
    assert all([a == b for a, b in zip(rand1, rand2)])


def test_seed_with_bad_input(q, kx):
    user_seed = 100
    kx.random.seed(user_seed)
    with pytest.raises(Exception) as err_info:
        kx.random.random(-10, 1, seed=10)
    assert str(err_info.value) == "length"
    assert q('system"S "') == user_seed

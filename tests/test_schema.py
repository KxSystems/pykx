# Do not import pykx here - use the `kx` fixture instead!

import pytest


def test_simple_schema(q, kx):
    qtab = kx.schema.builder({
        'col1': kx.GUIDAtom,
        'col2': kx.TimeAtom,
        'col3': kx.BooleanAtom,
        'col4': kx.FloatAtom})
    assert isinstance(qtab, kx.Table)
    assert kx.q.cols(qtab).py() == ['col1', 'col2', 'col3', 'col4']
    assert kx.q('{exec t from 0!meta x}', qtab).py() == b'gtbf'


def test_single_key_schema(q, kx):
    qtab = kx.schema.builder({'col1': kx.TimestampAtom,
                              'col2': kx.FloatAtom,
                              'col3': kx.IntAtom},
                             key='col1')
    assert isinstance(qtab, kx.KeyedTable)
    assert kx.q.cols(kx.q.key(qtab)).py() == ['col1']
    assert kx.q.cols(qtab).py() == ['col1', 'col2', 'col3']
    assert kx.q('{exec t from 0!meta x}', qtab).py() == b'pfi'


def test_multi_key_schema(q, kx):
    qtab = kx.schema.builder({'col1': kx.TimestampAtom,
                              'col2': kx.SymbolAtom,
                              'col3': kx.IntAtom,
                              'col4': kx.List},
                             key=['col1', 'col2'])
    assert isinstance(qtab, kx.KeyedTable)
    assert kx.q.cols(kx.q.key(qtab)).py() == ['col1', 'col2']
    assert kx.q.cols(qtab).py()==['col1', 'col2', 'col3', 'col4']
    assert kx.q('{exec t from 0!meta x}', qtab).py() == b'psi '


def test_builder_error(q, kx):
    with pytest.raises(Exception) as err_info:
        kx.schema.builder(1)
    assert str(err_info.value) == "'schema' argument should be a dictionary"

    with pytest.raises(Exception) as err_info:
        kx.schema.builder({'a': kx.FloatAtom, 1: kx.IntAtom})
    assert str(err_info.value) == "'schema' keys must be of type 'str'"

    with pytest.raises(Exception) as err_info:
        kx.schema.builder({'a': kx.FloatAtom, 'b': kx.IntAtom}, key=['a', 1])
    assert str(err_info.value) == "Supplied 'key' must be a list of 'str' types only"

    with pytest.raises(Exception) as err_info:
        kx.schema.builder({'a': 1, 'b': kx.IntAtom})
    assert str(err_info.value) == "Error: <class 'KeyError'> raised for column a"

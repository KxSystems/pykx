# Do not import pykx here - use the `kx` fixture instead!
from abc import ABCMeta

import pytest


def has_sql_fucntionality(_q):
    try:
        _q('.s')
        return True
    except BaseException:
        return False


@pytest.mark.embedded
def test_sql(q):
    if has_sql_fucntionality(q):
        # python q object based on memory location
        qtab = q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')

        # assign python q object to named entity
        q['qtab'] = q['stab'] = qtab

        assert q('.s.e"select * from qtab"').py() == q.sql("select * from $1", q('stab')).py()

        assert q('.s.e"select col1 from qtab where col2>0.5"').py() \
            == q.sql("select col1 from $1 where col2 > $2", q('stab'), q('0.5f')).py()


@pytest.mark.embedded
def test_prepared_sql(q, kx):
    if has_sql_fucntionality(q):
        # python q object based on memory location
        qtab = q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')

        # assign python q object to named entity
        q['qtab'] = q['stab'] = qtab

        p1 = q.sql.prepare(
            "select * from stab where col2 > $1 and col3 < $2",
            kx.FloatAtom(0.0),
            kx.LongAtom(0)
        )
        p11 = q.sql.prepare(
            "select * from stab where col2 > $1 and col3 < $2",
            kx.FloatAtom,
            kx.LongAtom
        )
        q('p:.s.sq["select * from qtab where col2 > $1 and col3 < $2"](0n; 0N)')
        p2 = q.sql.prepare(
            "select * from stab where col1 in $1 and col2 > $2",
            kx.SymbolVector(['', '']),
            kx.FloatAtom(0.0),
        )
        p21 = q.sql.prepare(
            "select * from stab where col1 in $1 and col2 > $2",
            kx.SymbolVector,
            kx.FloatAtom,
        )
        q('p2:.s.sq["select * from qtab where col1 in $1 and col2 > $2"](``; 0n)')

        assert q('.s.sx[p](0.5f; 2)').py() \
            == q.sql.execute(p1, kx.FloatAtom(0.5), kx.LongAtom(2)).py()
        assert q.sql.execute(p1, kx.FloatAtom(0.5), kx.LongAtom(2)).py() \
            == q.sql.execute(p11, kx.FloatAtom(0.5), kx.LongAtom(2)).py()

        assert q('.s.sx[p2](`a`b; 0.5)').py() \
            == q.sql.execute(p2, kx.SymbolVector(['a', 'b']), kx.FloatAtom(0.5)).py()
        assert q.sql.execute(p2, kx.SymbolVector(['a', 'b']), kx.FloatAtom(0.5)).py() \
            == q.sql.execute(p21, kx.SymbolVector(['a', 'b']), kx.FloatAtom(0.5)).py()

        p3 = q.sql.prepare(
            "select * from $1",
            kx.Table.prototype({
                'col1': kx.SymbolVector,
                'col2': kx.FloatVector,
                'col3': kx.LongVector
            })
        )
        q('p3:.s.sq["select * from $1"; (([] col1:``; col2:0.0 0.0; col3:0 0); 0)]')

        assert q('.s.sx[p3](qtab; 0)').py() == q.sql.execute(p3, q('stab')).py()


@pytest.mark.unlicensed
def test_vec_atom_prototypes(kx):
    types = [
        kx.GUIDAtom,
        kx.CharAtom,
        kx.SymbolAtom,
        kx.ByteAtom,
        kx.ShortAtom,
        kx.IntAtom,
        kx.LongAtom,
        kx.RealAtom,
        kx.FloatAtom,
        kx.TimestampAtom,
        kx.MonthAtom,
        kx.DateAtom,
        kx.TimespanAtom,
        kx.MinuteAtom,
        kx.SecondAtom,
        kx.TimeAtom,

        kx.GUIDVector,
        kx.CharVector,
        kx.SymbolVector,
        kx.ByteVector,
        kx.ShortVector,
        kx.IntVector,
        kx.LongVector,
        kx.RealVector,
        kx.FloatVector,
        kx.TimestampVector,
        kx.MonthVector,
        kx.DateVector,
        kx.TimespanVector,
        kx.MinuteVector,
        kx.SecondVector,
        kx.TimeVector,
    ]
    for t in types:
        proto = t._prototype()
        assert t == type(proto)
        # this is the check we use internally, to avoid testing every type check in an actual
        # sql call we just check that they are converted properly and that the check we use will
        # match against them but not a constructed type
        assert type(t) is ABCMeta or type(t) is type
        assert not (type(proto) is ABCMeta or type(proto) is type)


@pytest.mark.embedded
def test_table_prototype(q, kx):
    proto = kx.Table.prototype({'a': kx.LongVector, 'b': kx.RealVector, 'c': kx.SymbolVector})
    q['proto'] = proto
    assert type(proto) == kx.Table
    assert type(q('proto `a')) == kx.LongVector
    assert type(q('proto `b')) == kx.RealVector
    assert type(q('proto `c')) == kx.SymbolVector


@pytest.mark.embedded
def test_sql_get_input_values(q, kx):
    if has_sql_fucntionality(q):
        # python q object based on memory location
        qtab = q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')

        # assign python q object to named entity
        q['stab'] = qtab

        p1 = q.sql.prepare(
            "select * from stab where col2 > $1 and col3 < $2",
            kx.FloatAtom,
            kx.LongAtom
        )
        p2 = q.sql.prepare(
            "select * from stab where col1 in $1 and col2 > $2",
            kx.SymbolVector,
            kx.FloatAtom,
        )

        assert q.sql.get_input_types(p1) == ['FloatAtom/FloatVector', 'LongAtom/LongVector']
        assert q.sql.get_input_types(p2) == ['SymbolAtom/SymbolVector', 'FloatAtom/FloatVector']

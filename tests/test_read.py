from contextlib import nullcontext

import pytest


@pytest.fixture
def tmp_csv_path_1(tmp_path):
    p = tmp_path/'tmp1.csv'
    p.write_text('\n'.join((
        'col,a,b,c',
        *(f'{"abc"[i % 3]},{i},{10 - i},{3 * i}' for i in range(10)),
    )))
    yield p
    p.unlink()


@pytest.fixture
def tmp_csv_path_2(tmp_path):
    p = tmp_path/'tmp2.csv'
    p.write_text('\n'.join(('a', *(str(x) for x in range(10)))))
    yield p
    p.unlink()


@pytest.mark.ipc
def test_read_csv(kx, q, tmp_csv_path_1, tmp_csv_path_2):
    assert isinstance(
        q.read.csv(str(tmp_csv_path_1), kx.CharVector('JJJ'), kx.CharAtom(','), True),
        kx.Table,
    )
    assert isinstance(
        q.read.csv(str(tmp_csv_path_1), kx.CharVector('JJJ'), kx.CharAtom(','), True),
        kx.Table,
    )
    assert isinstance(q.read.csv(kx.SymbolAtom(str(tmp_csv_path_1)), 'JJJ'), kx.Table)
    assert isinstance(q.read.csv(tmp_csv_path_1, b'JJJ', ',', False), kx.List)
    assert isinstance(q.read.csv(tmp_csv_path_2, 'J'), kx.Table)
    assert isinstance(q.read.csv(tmp_csv_path_2, kx.CharAtom('J')), kx.Table)
    assert isinstance(q.read.csv(tmp_csv_path_2, b'J', b','), kx.Table)
    assert isinstance(q.read.csv(tmp_csv_path_2, kx.CharVector('J'), b',', False), kx.List)
    assert isinstance(q.read.csv(tmp_csv_path_1, [kx.LongAtom, kx.LongAtom, kx.LongAtom]), kx.Table)
    if not kx.licensed:
        ctx = pytest.raises(kx.LicenseException)
    elif isinstance(q, kx.QConnection):
        ctx = pytest.raises(ValueError)
    else:
        ctx = nullcontext()
    with ctx:
        assert isinstance(
            q.read.csv(tmp_csv_path_1, {'a': kx.LongAtom, 'b': kx.LongAtom, 'c': kx.LongAtom}),
            kx.Table
        )
        tab = q.read.csv(tmp_csv_path_1, [kx.LongAtom, ' ', kx.LongAtom])
        assert isinstance(tab, kx.Table)
        assert len(tab.columns) == 2


@pytest.mark.ipc
def test_read_csv_with_type_guessing(kx, q, tmp_csv_path_1, tmp_csv_path_2):
    if not kx.licensed:
        ctx = pytest.raises(kx.LicenseException)
    elif isinstance(q, kx.QConnection):
        ctx = pytest.raises(ValueError)
    else:
        ctx = nullcontext()

    with ctx:
        assert isinstance(
            q.read.csv(str(tmp_csv_path_1), None, kx.CharAtom(','), True),
            kx.Table,
        )
    with ctx:
        assert isinstance(
            q.read.csv(str(tmp_csv_path_1), None, kx.CharAtom(','), True),
            kx.Table,
        )
    with ctx:
        assert isinstance(q.read.csv(kx.SymbolAtom(str(tmp_csv_path_1))), kx.Table)
    with ctx:
        assert isinstance(q.read.csv(tmp_csv_path_1, None, ',', False), kx.List)
    with ctx:
        assert isinstance(q.read.csv(tmp_csv_path_2), kx.Table)
    with ctx:
        assert isinstance(q.read.csv(tmp_csv_path_2, None, b','), kx.Table)
    with ctx:
        assert isinstance(q.read.csv(tmp_csv_path_2, None, b',', False), kx.List)


@pytest.fixture
def tmp_fixed_path(tmp_path):
    p = tmp_path/'tmp.fixed'
    p.write_text('023054045067063072081093101155122174141202')
    yield p
    p.unlink()


@pytest.mark.ipc
def test_read_fixed(kx, q, tmp_fixed_path):
    assert isinstance(q.read.fixed(str(tmp_fixed_path), b'JJ', [1, 2]), kx.List)
    assert isinstance(
        q.read.fixed(kx.SymbolAtom(str(tmp_fixed_path)), [b'J', b'J'], [1, 2]),
        kx.List,
    )
    assert isinstance(q.read.fixed(tmp_fixed_path, ['J', 'J'], [1, 2]), kx.List)

    assert q.read.fixed(tmp_fixed_path, ['J', 'J'], [1, 2]).py() == \
        q.read.fixed(tmp_fixed_path, kx.CharVector(['J', 'J']), kx.LongVector([1, 2])).py()


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_read_fixed_async(kx, tmp_path, q_port, tmp_fixed_path):
    async with kx.AsyncQConnection(port=q_port) as q:
        assert isinstance(await q.read.fixed(str(tmp_fixed_path), b'JJ', [1, 2]), kx.List)
        assert isinstance(
            await q.read.fixed(kx.SymbolAtom(str(tmp_fixed_path)), [b'J', b'J'], [1, 2]),
            kx.List,
        )
        assert isinstance(await q.read.fixed(tmp_fixed_path, ['J', 'J'], [1, 2]), kx.List)

        assert (await q.read.fixed(tmp_fixed_path, ['J', 'J'], [1, 2])).py() == \
               (await q.read.fixed(tmp_fixed_path,
                kx.CharVector(['J', 'J']),
                kx.LongVector([1, 2]))
               ).py()


@pytest.fixture
def tmp_json_path(tmp_path):
    p = tmp_path/'tmp.json'
    p.write_text(
        '[{"t":0,"a":0},{"t":1,"a":2},{"t":2,"a":4},{"t":3,"a":6},{"t":4,"a":8},{"t":5,"a":10},'
        '{"t":6,"a":12},{"t":7,"a":14},{"t":8,"a":16},{"t":9,"a":18}]'
    )
    yield p
    p.unlink()


@pytest.mark.ipc
def test_read_json(kx, q, tmp_json_path):
    assert isinstance(q.read.json(str(tmp_json_path)), kx.Table)
    assert isinstance(q.read.json(kx.SymbolAtom(str(tmp_json_path))), kx.Table)
    assert isinstance(q.read.json(tmp_json_path), kx.Table)


@pytest.fixture
def tmp_simple_q_table_maker(tmp_path):
    p = tmp_path/'t'

    def table_maker(q):
        q('t:([] a:til 10; b:2*til 10)')
        q('save', p)
        return p
    return table_maker


@pytest.mark.ipc
def test_read_serialized(kx, q, tmp_simple_q_table_maker):
    p = tmp_simple_q_table_maker(q)
    assert isinstance(q.read.serialized(str(p)), kx.Table)
    assert isinstance(q.read.serialized(kx.SymbolAtom(p)), kx.Table)
    assert isinstance(q.read.serialized(p), kx.Table)


# Licensed only is used for because of the type checks in this test
@pytest.mark.ipc(licensed_only=True)
def test_read_splayed(kx, q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    q('{(hsym `$"/" sv string x,y) set .Q.en[hsym x;] z}', tmp_path, 't/', t)
    assert isinstance(q.read.splayed(tmp_path, kx.SymbolAtom('t')), kx.SplayedTable)
    assert isinstance(q.read.splayed(tmp_path, 't'), kx.SplayedTable)


@pytest.mark.unlicensed
def test_dir_read(kx):
    assert isinstance(dir(kx.read), list)
    assert sorted(dir(kx.read)) == dir(kx.read)

import pytest

from pathlib import Path


@pytest.mark.ipc
def test_write_serialized(kx, q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    for x in (str(tmp_path/'str'), kx.SymbolAtom(str(tmp_path/'sym')), tmp_path/'path'):
        q.write.serialized(x, t)
        assert q('{x ~ get hsym y}', t, x)


@pytest.mark.ipc
def test_write_splayed(kx, q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    cases = (
        (str(tmp_path), 'test_splayed_str'),
        (kx.SymbolAtom(str(tmp_path)), kx.SymbolAtom('test_splayed_sym')),
        (tmp_path, 'test_splayed_path'),
        (tmp_path, b'test_splayed_path2'),
    )
    for a, b in cases:
        assert q('{y ~ get hsym x}', Path(str(q.write.splayed(a, b, t)) + '/'), t)


@pytest.mark.ipc
def test_write_csv(kx, q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    comma = tmp_path/'comma.csv'
    tab = tmp_path/'tab.csv'

    def check(x):
        with open(x) as f:
            rows = f.readlines()
            assert len(file_contents) == len(rows)
            for a, b in zip(file_contents, rows):
                assert a == b

    file_contents = ('a,b\n', '0,0\n', '1,2\n', '2,4\n', '3,6\n', '4,8\n')
    q.write.csv(kx.SymbolAtom(str(comma)), t, ',')
    check(comma)

    file_contents = ('a	b\n', '0	0\n', '1	2\n', '2	4\n', '3	6\n', '4	8\n')
    q.write.csv(tab, t, b'	')
    check(tab)


@pytest.mark.ipc
def test_write_json(kx, q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    for x in (str(tmp_path/'str.json'), kx.SymbolAtom(tmp_path/'sym.json'), tmp_path/'path.json'):
        with open(q.write.json(x, t)) as f:
            assert f.readline().strip() == \
                '[{"a":0,"b":0},{"a":1,"b":2},{"a":2,"b":4},{"a":3,"b":6},{"a":4,"b":8}]'


@pytest.mark.ipc
def test_read_write_json(q, tmp_path):
    # Cast to float because read JSON guesses types and defaults to float for numeric values
    t = q('([] a:"f"$til 5; b:2*"f"$til 5)')
    assert q('~', t, q.read.json(q.write.json(tmp_path/'both.json', t)))


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_read_write_json_async(kx, tmp_path, q_port):
    # Cast to float because read JSON guesses types and defaults to float for numeric values
    async with kx.AsyncQConnection(port=q_port) as q:
        t = await q('([] a:"f"$til 5; b:2*"f"$til 5)')
        assert await q('~', t, await q.read.json(q.write.json(tmp_path/'both.json', t)))


@pytest.mark.ipc
def test_read_write_csv(q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    assert q('~', t, q.read.csv(q.write.csv(tmp_path/'both.csv', t), b'JJ'))


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_read_write_csv_async(kx, tmp_path, q_port):
    async with kx.AsyncQConnection(port=q_port) as q:
        t = await q('([] a:til 5; b:2*til 5)')
        assert await q('~', t, await q.read.csv(q.write.csv(tmp_path/'both.csv', t), b'JJ'))


@pytest.mark.ipc
def test_read_write_serialized(q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    assert q('~', t, q.read.serialized(q.write.serialized(tmp_path/'both', t)))
    assert q('~', 145, q.read.serialized(q.write.serialized(tmp_path/'both', 145)))


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_read_write_serialized_async(kx, tmp_path, q_port):
    async with kx.AsyncQConnection(port=q_port) as q:
        t = await q('([] a:"f"$til 5; b:2*"f"$til 5)')
        assert await q('~', t, await q.read.serialized(q.write.serialized(tmp_path/'both', t)))
        assert await q('~', 145, await q.read.serialized(q.write.serialized(tmp_path/'both', 145)))


@pytest.mark.ipc
def test_read_write_splayed(kx, q, tmp_path):
    t = q('([] a:til 5; b:2*til 5)')
    q.write.splayed(tmp_path, 'splayed', t)
    q['splayed_table'] = q.read.splayed(tmp_path, 'splayed')
    assert q('~', t, q('select from splayed_table'))


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_read_write_splayed_async(kx, tmp_path, q_port):
    async with kx.AsyncQConnection(port=q_port) as q:
        t = await q('([] a:"f"$til 5; b:2*"f"$til 5)')
        q.write.splayed(tmp_path, 'splayed', t)
        q['splayed_table'] = await q.read.splayed(tmp_path, 'splayed')
        assert await q('~', t, await q('select from splayed_table'))


@pytest.mark.unlicensed
def test_dir_write(kx):
    assert isinstance(dir(kx.write), list)
    assert sorted(dir(kx.write)) == dir(kx.write)

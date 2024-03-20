from tempfile import TemporaryDirectory

# Do not import pykx here - use the `kx` fixture instead!
import pytest


@pytest.mark.ipc
def test_select(kx, q):
    # python q object based on memory location
    qtab = q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')
    qbool = q('100?0b')
    # assign python q object to named entity
    q['qtab'] = q['q-tab'] = qtab
    q['bool'] = qbool
    assert q.qsql.select(qtab).t == 98
    assert q('select from qtab').py() == q.qsql.select('q-tab').py()
    assert q('select from qtab').py() == q.qsql.select(qtab).py()
    assert q('select from qtab').py() == q.qsql.select(qtab.pd()).py()
    assert q('select from qtab where col1=`a').py() == q.qsql.select(qtab, where='col1=`a').py()
    assert q('select from qtab where bool').py() == q.qsql.select(qtab, where=qbool).py()
    assert q('select from qtab where col1=`a,col2>0.5').py() ==\
        q.qsql.select(qtab, where=['col1=`a', 'col2>0.5']).py()
    assert q('select col1 from qtab').py() == q.qsql.select(qtab, columns={'col1': 'col1'}).py()
    assert q('select col1 from qtab').py() == q.qsql.select(qtab, columns='col1').py()
    assert q('select col1,col2 from qtab').py() == \
        q.qsql.select(qtab, columns=['col1', 'col2']).py()
    assert q('select maxCol2:max col2 from qtab').py() ==\
        q.qsql.select(qtab, columns={'maxCol2': 'max col2'}).py()
    assert q('select sumcols:col2+col3 from qtab').py() ==\
        q.qsql.select(qtab, columns={'sumcols': 'col2+col3'}).py()
    assert q('select maxCol2:max col2 by groupcol:col1 from qtab').py() ==\
        q.qsql.select(qtab, columns={'maxCol2': 'max col2'}, by={'groupcol': 'col1'}).py()
    assert q('select minCol2:min col2,max col3 by col1 from qtab where col3<0.5,col2>0.7').py() ==\
        q.qsql.select(qtab,
                      columns={'minCol2': 'min col2', 'col3': 'max col3'},
                      by={'col1': 'col1'},
                      where=['col3<0.5', 'col2>0.7']
        ).py()
    with pytest.raises(TypeError):
        q.qsql.select([1, 2, 3]).py()
    with pytest.raises(kx.QError) as err:
        q.qsql.select(qtab,
                      columns={'col2': 'col2', 'col3': 'col3'},
                      where=['col3<0.5', 'col2>0.7'],
                      by={'col1': 'col1'},
                      inplace=True
        ).py()
    assert 'Returned data format does not match' in str(err)
    q.qsql.select(qtab,
                  columns={'col2': 'col2', 'col3': 'col3'},
                  where=['col3<0.5', 'col2>0.7'],
                  inplace=True
    )
    assert q('select col2, col3 from qtab where col3<0.5,col2>0.7').py() ==\
        qtab.py()


def test_partitioned_query(kx, q):
    with TemporaryDirectory() as tmp_dir:
        db = kx.DB(path=tmp_dir)
        N = 1000
        qtab = kx.Table(data={
            'date': kx.q.asc(kx.random.random(N, kx.q('2020.01.01 2020.01.02 2020.01.03'))),
            'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
            'price': kx.random.random(N, 10.0),
            'size': kx.random.random(N, 100)
        })
        db.create(qtab, 'qtable', 'date')
        with pytest.raises(kx.QError) as err:
            kx.q.qsql.select(db.qtable, where=['sym=`AAPL'], inplace=True)
        assert "Application of 'inplace' updates not supported" in str(err)

        with pytest.raises(kx.QError) as err:
            kx.q.qsql.delete(db.qtable, where=['sym=`AAPL'], inplace=True)
        assert "Application of 'inplace' updates not supported" in str(err)

        with pytest.raises(kx.QError) as err:
            kx.q.qsql.update(db.qtable, where=['sym=`AAPL'], inplace=True)
        assert "Application of 'inplace' updates not supported" in str(err)


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_select_async(kx, q_port):
    async with kx.AsyncQConnection(port=q_port) as q:
        # python q object based on memory location
        qtab = await q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')
        qbool = await q('100?0b')
        # assign python q object to named entity
        q['qtab'] = q['q-tab'] = qtab
        q['bool'] = qbool
        assert (await q.qsql.select(qtab)).t == 98
        assert (await q('select from qtab')).py() == (await q.qsql.select('q-tab')).py()
        assert (await q('select from qtab')).py() == (await q.qsql.select(qtab)).py()
        assert (await q('select from qtab where col1=`a')).py() ==\
               (await q.qsql.select(qtab, where='col1=`a')).py()
        assert (await q('select from qtab where bool')).py() ==\
               (await q.qsql.select(qtab, where=qbool)).py()
        assert (await q('select from qtab where col1=`a,col2>0.5')).py() ==\
               (await q.qsql.select(qtab, where=['col1=`a', 'col2>0.5'])).py()
        assert (await q('select col1 from qtab')).py() ==\
               (await q.qsql.select(qtab, columns={'col1': 'col1'})).py()
        assert (await q('select col1 from qtab')).py() ==\
               (await q.qsql.select(qtab, columns='col1')).py()
        assert (await q('select col1,col2 from qtab')).py() == \
               (await q.qsql.select(qtab, columns=['col1', 'col2'])).py()
        assert (await q('select maxCol2:max col2 from qtab')).py() ==\
               (await q.qsql.select(qtab, columns={'maxCol2': 'max col2'})).py()
        assert (await q('select sumcols:col2+col3 from qtab')).py() ==\
               (await q.qsql.select(qtab, columns={'sumcols': 'col2+col3'})).py()
        assert (await q('select maxCol2:max col2 by groupcol:col1 from qtab')).py() ==\
               (await q.qsql.select(qtab,
                                    columns={'maxCol2': 'max col2'},
                                    by={'groupcol': 'col1'})
               ).py()
        assert (await
                q('select minCol2:min col2,max col3 by col1 from qtab where col3<0.5,col2>0.7')
               ).py()\
            == (await q.qsql.select(qtab,
                                    columns={'minCol2': 'min col2', 'col3': 'max col3'},
                                    by={'col1': 'col1'},
                                    where=['col3<0.5', 'col2>0.7']
            )).py()

        with pytest.raises(kx.QError) as err:
            await (q.qsql.select(qtab,
                                 columns={'col2': 'col2', 'col3': 'col3'},
                                 where=['col3<0.5', 'col2>0.7'],
                                 inplace=True
                  ))
        assert "'inplace' not supported" in str(err)


@pytest.mark.ipc
def test_exec(q):
    qtab = q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')
    q['qtab'] = q['q-tab'] = qtab
    assert q.qsql.exec(qtab).t == 99
    assert q('exec from qtab').py() == q.qsql.exec('q-tab').py()
    assert q('exec from qtab').py() == q.qsql.exec(qtab).py()
    assert q('exec from qtab').py() == q.qsql.exec(qtab.pd()).py()
    assert q('exec from qtab where col1=`a').py() == q.qsql.exec(qtab, where='col1=`a').py()
    assert q('exec col1 from qtab').py() == q.qsql.exec(qtab, 'col1').py()
    assert q('exec col1, col2 from qtab').py() == q.qsql.exec(qtab, ['col1', 'col2']).py()
    assert q('exec col2,col3 by col1 from qtab').py() ==\
        q.qsql.exec(qtab, columns={'col2': 'col2', 'col3': 'col3'}, by='col1').py()
    assert q('exec col2,col3 by col1 from qtab').py() ==\
        q.qsql.exec(qtab, ['col2', 'col3'], by='col1').py()
    assert q('exec col2 by col1 from qtab').py() == q.qsql.exec(qtab, 'col2', by='col1').py()
    assert q('exec avgCol3:avg col3 by col1 from qtab').py() ==\
        q.qsql.exec(qtab, {'avgCol3': 'avg col3'}, by='col1').py()
    assert q('exec sumcol2:sum col2 by aggcol:col1 from qtab').py() ==\
        q.qsql.exec(qtab, columns={'sumcol2': 'sum col2'}, by={'aggcol': 'col1'}).py()
    with pytest.raises(TypeError):
        q.qsql.exec([1, 2, 3]).py()


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_exec_async(kx, q_port):
    async with kx.AsyncQConnection(port=q_port) as q:
        # python q object based on memory location
        qtab = await q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')
        q['qtab'] = q['q-tab'] = qtab
        assert (await q.qsql.exec(qtab)).t == 99
        assert (await q('exec from qtab')).py() == (await q.qsql.exec('q-tab')).py()
        assert (await q('exec from qtab')).py() == (await q.qsql.exec(qtab)).py()
        assert (await q('exec from qtab where col1=`a')).py() ==\
               (await q.qsql.exec(qtab, where='col1=`a')).py()
        assert (await q('exec col1 from qtab')).py() == (await q.qsql.exec(qtab, 'col1')).py()
        assert (await q('exec col1, col2 from qtab')).py() ==\
               (await q.qsql.exec(qtab, ['col1', 'col2'])).py()
        assert (await q('exec col2,col3 by col1 from qtab')).py() ==\
               (await q.qsql.exec(qtab, columns={'col2': 'col2', 'col3': 'col3'}, by='col1')).py()
        assert (await q('exec col2,col3 by col1 from qtab')).py() ==\
               (await q.qsql.exec(qtab, ['col2', 'col3'], by='col1')).py()
        assert (await q('exec col2 by col1 from qtab')).py() ==\
               (await q.qsql.exec(qtab, 'col2', by='col1')).py()
        assert (await q('exec avgCol3:avg col3 by col1 from qtab')).py() ==\
               (await q.qsql.exec(qtab, {'avgCol3': 'avg col3'}, by='col1')).py()
        assert (await q('exec sumcol2:sum col2 by aggcol:col1 from qtab')).py() ==\
               (await q.qsql.exec(qtab,
                                  columns={'sumcol2': 'sum col2'},
                                  by={'aggcol': 'col1'})
               ).py()


@pytest.mark.ipc
def test_update(q):
    qtab = q('([]name:`tom`dick`harry;age:28 29 35;hair:`fair`dark`fair;eye:`green`brown`gray)')
    q['qtab'] = q['q-tab'] = qtab
    assert q('update eye:`blue`brown`green from qtab').py() ==\
        q.qsql.update('q-tab', {'eye': '`blue`brown`green'}).py()
    assert q('update eye:`blue`brown`green from qtab').py() ==\
        q.qsql.update(qtab, {'eye': '`blue`brown`green'}).py()
    assert q('update eye:`blue`brown`green from qtab').py() ==\
        q.qsql.update(qtab.pd(), {'eye': '`blue`brown`green'}).py()
    assert q('update eye:`blue from qtab where hair=`fair').py() ==\
        q.qsql.update(qtab, {'eye': ['blue']}, where='hair=`fair').py()
    assert q('update age:25 30 31 from qtab').py() == \
        q.qsql.update(qtab, {'age': [25, 30, 31]}).py()
    assert q('update age:25 30 31 from qtab').py() == \
        q.qsql.update(qtab, {'age': q('25 30 31')}).py()

    with pytest.raises(ValueError):
        q.qsql.update(qtab, {'hair': ''})

    byqtab = q('([]p:100?`a`b`c;name:100?`nut`bolt`screw;'
               'color:100?`red`green`blue;weight:0.5*100?20;'
               'city:100?`london`paris`rome)')
    q['byqtab'] = byqtab
    assert q('update avg weight by city from byqtab').py() ==\
        q.qsql.update(byqtab, {'weight': 'avg weight'}, by={'city': 'city'}).py()

    with pytest.raises(TypeError):
        q.qsql.update([1, 2, 3], columns={'weight': 'avg weight'}, by={'city': 'city'}, inplace=True) # noqa: E501

    q.qsql.update('byqtab', columns={'weight': 'max weight'}, by={'city': 'city'}, inplace=True)
    q.qsql.update(byqtab, columns={'weight': 'max weight'}, by={'city': 'city'}, inplace=True)
    assert q['byqtab'].py() == byqtab.py()

    with pytest.raises(RuntimeError) as err:
        q.qsql.update(qtab, columns={'newcol': 'weight'}, modify=True, inplace=True)
    assert 'Attempting to use both' in str(err)
    assert 0 == q('count .pykx.i.updateCache').py()


def test_update_sym_leak(q):
    qtab = q('([]name:`tom`dick`harry;age:28 29 35;hair:`fair`dark`fair;eye:`green`brown`gray)')
    q.qsql.update(qtab, {'eye': '`blue`brown`green'})
    syms = q.Q.w()['syms']
    q.qsql.update(qtab, {'eye': '`blue`brown`green'})
    q.qsql.update(qtab, {'eye': '`blue`brown`green'})
    assert syms == q.Q.w()['syms']


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_update_async(kx, q_port):
    async with kx.AsyncQConnection(port=q_port) as q:
        qtab = await q(('([]name:`tom`dick`harry;age:28 29 35;'
                        'hair:`fair`dark`fair;eye:`green`brown`gray)'))
        q['qtab'] = q['q-tab'] = qtab
        assert (await q('update eye:`blue`brown`green from qtab')).py() ==\
               (await q.qsql.update('q-tab', {'eye': '`blue`brown`green'})).py()
        assert (await q('update eye:`blue`brown`green from qtab')).py() ==\
               (await q.qsql.update(qtab, {'eye': '`blue`brown`green'})).py()
        assert (await q('update eye:`blue from qtab where hair=`fair')).py() ==\
               (await q.qsql.update(qtab, {'eye': ['blue']}, where='hair=`fair')).py()
        assert (await q('update age:25 30 31 from qtab')).py() == \
               (await q.qsql.update(qtab, {'age': [25, 30, 31]})).py()
        assert (await q('update age:25 30 31 from qtab')).py() == \
               (await q.qsql.update(qtab, {'age': await q('25 30 31')})).py()
        byqtab = await q('([]p:100?`a`b`c;name:100?`nut`bolt`screw;'
                         'color:100?`red`green`blue;weight:0.5*100?20;'
                         'city:100?`london`paris`rome)')
        q['byqtab'] = byqtab
        assert (await q('update max weight by city from byqtab')).py() ==\
            (await q.qsql.update(byqtab, {'weight': 'max weight'}, by={'city': 'city'})).py()
        await q.qsql.update('byqtab',
                            columns={'weight': 'max weight'},
                            by={'city': 'city'},
                            inplace=True)
        with pytest.raises(kx.QError):
            await q.qsql.update(byqtab,
                                columns={'weight': 'max weight'},
                                by={'city': 'city'},
                                inplace=True)
        assert q['byqtab'].py() == \
            (await q.qsql.update(byqtab, {'weight': 'max weight'}, by={'city': 'city'})).py()


@pytest.mark.ipc
def test_delete(q):
    qtab = q('([]name:`tom`dick`harry;age:28 29 35;hair:`fair`dark`fair;eye:`green`brown`gray)')
    qbool = q('3?0b')
    q['qtab'] = q['q-tab'] = qtab
    q['qbool'] = qbool
    assert 0 == len(q.qsql.delete(qtab))
    assert 0 == len(q.qsql.delete('q-tab'))
    assert q('delete name from qtab').py() == q.qsql.delete(qtab, 'name').py()
    assert q('delete name from qtab').py() == q.qsql.delete(qtab.pd(), 'name').py()
    assert q('delete name,hair from qtab').py() == q.qsql.delete(qtab, ['name', 'hair']).py()
    assert q('delete from qtab where hair=`fair').py() == \
        q.qsql.delete(qtab, where='hair=`fair').py()
    assert q('delete from qtab where qbool').py()
    q.qsql.delete('q-tab', where='hair=`fair', inplace=True)
    q.qsql.delete(qtab, where='hair=`fair', inplace=True)
    assert q['q-tab'].py() == qtab.py()
    with pytest.raises(TypeError):
        q.qsql.delete('q-tab', where='hair=`fair', columns=['age'])
    with pytest.raises(TypeError):
        q.qsql.delete([1, 2, 3], where='hair=`fair', columns=['age'])


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_delete_async(kx, q_port):
    async with kx.AsyncQConnection(port=q_port) as q:
        qtab = await q(('([]name:`tom`dick`harry;age:28 29 35;'
                        'hair:`fair`dark`fair;eye:`green`brown`gray)'))
        qbool = await q('3?0b')
        q['qtab'] = q['q-tab'] = qtab
        q['qbool'] = qbool
        assert 0 == len(await q.qsql.delete(qtab))
        assert 0 == len(await q.qsql.delete('q-tab'))
        assert (await q('delete name from qtab')).py() ==\
               (await q.qsql.delete(qtab, 'name')).py()
        assert (await q('delete name,hair from qtab')).py() ==\
               (await q.qsql.delete(qtab, ['name', 'hair'])).py()
        assert (await q('delete from qtab where hair=`fair')).py() == \
               (await q.qsql.delete(qtab, where='hair=`fair')).py()
        assert (await q('delete from qtab where qbool')).py()
        await q.qsql.delete('q-tab', where='hair=`fair', inplace=True)
        with pytest.raises(kx.QError):
            await q.qsql.delete(qtab, where='hair=`fair', inplace=True)
        with pytest.raises(TypeError):
            await q.qsql.delete('q-tab', where='hair=`fair', columns=['age'])
        assert q['q-tab'].py() == (await q('delete from qtab where hair=`fair')).py()


@pytest.mark.ipc
def test_query_pa(q, pa):
    qtab = q('([]name:`tom`dick`harry;age:28 29 35;hair:`fair`dark`fair;eye:`green`brown`gray)')
    q['qtab'] = q['q-tab'] = qtab
    assert q('delete name from qtab').py() == q.qsql.delete(qtab.pa(), 'name').py()

    qtab = q('([]name:`tom`dick`harry;age:28 29 35;hair:`fair`dark`fair;eye:`green`brown`gray)')
    q['qtab'] = q['q-tab'] = qtab
    assert q('update eye:`blue`brown`green from qtab').py() ==\
        q.qsql.update(qtab.pa(), {'eye': '`blue`brown`green'}).py()

    qtab = q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')
    q['qtab'] = q['q-tab'] = qtab
    assert q('exec from qtab').py() == q.qsql.exec(qtab.pa()).py()

    qtab = q('([]col1:100?`a`b`c;col2:100?1f;col3:100?5)')
    q['qtab'] = q['q-tab'] = qtab
    assert q('select from qtab').py() == q.qsql.select(qtab.pa()).py()


@pytest.mark.embedded
def test_insert_test_insert(q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')
    q['qtab'] = qtab

    assert q_inserted_tab.py() == q.insert('qtab', [6, 6.0, 'f'], test_insert=True).py()
    assert q('qtab').py() == qtab.py()


@pytest.mark.ipc
def test_insert_match_schema(kx, q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')
    q['qtab'] = qtab

    q.insert('qtab', [6, 6.0, 'f'], match_schema=True)
    assert q_inserted_tab.py() == q('qtab').py()

    q['qtab'] = qtab
    q.insert('qtab', [[6, 7], [6.0, 7.0], ['f', 'g']], match_schema=True)
    assert q('([] a: 1 2 3 4 5 6 7; b: 1.0 2.0 3.0 4.0 5.0 6.0 7.0; c: `a`b`c`d`e`f`g)').py() ==\
        q('qtab').py()

    with pytest.raises(kx.PyKXException):
        q.insert('qtab', [6.0, 6, 'f'], match_schema=True)


@pytest.mark.pandas_api
def test_table_insert_method(q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')

    with pytest.warns(DeprecationWarning,
                      match=r"Keyword 'replace_self' is deprecated please use 'inplace'"):
        assert qtab.insert([6, 6.0, 'f'], replace_self=False).py() == q_inserted_tab.py()
    assert qtab.insert([6, 6.0, 'f'], inplace=False).py() == q_inserted_tab.py()
    assert qtab.py() != q_inserted_tab.py()

    qtab.insert([6, 6.0, 'f'])
    assert qtab.py() == q_inserted_tab.py()


@pytest.mark.pandas_api
def test_table_upsert_method(q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')

    with pytest.warns(DeprecationWarning,
                      match=r"Keyword 'replace_self' is deprecated please use 'inplace'"):
        assert qtab.upsert([6, 6.0, 'f'], replace_self=False).py() == q_inserted_tab.py()
    assert qtab.upsert([6, 6.0, 'f'], inplace=False).py() == q_inserted_tab.py()
    assert qtab.py() != q_inserted_tab.py()

    qtab.upsert([6, 6.0, 'f'])
    assert qtab.py() == q_inserted_tab.py()


@pytest.mark.pandas_api
def test_keyed_table_insert_method(q):
    qtab = q('([a: 1 2 3 4 5] b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([a: 1 2 3 4 5 6] b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')

    with pytest.warns(DeprecationWarning,
                      match=r"Keyword 'replace_self' is deprecated please use 'inplace'"):
        assert qtab.insert([6, 6.0, 'f'], replace_self=False).py() == q_inserted_tab.py()
    assert qtab.insert([6, 6.0, 'f'], inplace=False).py() == q_inserted_tab.py()
    assert qtab.py() != q_inserted_tab.py()

    qtab.insert([6, 6.0, 'f'])
    assert qtab.py() == q_inserted_tab.py()


@pytest.mark.pandas_api
def test_keyed_table_upsert_method(q):
    qtab = q('([a: 1 2 3 4 5] b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([a: 1 2 3 4 5 6] b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')

    with pytest.warns(DeprecationWarning,
                      match=r"Keyword 'replace_self' is deprecated please use 'inplace'"):
        assert qtab.upsert([6, 6.0, 'f'], replace_self=False).py() == q_inserted_tab.py()
    assert qtab.upsert([6, 6.0, 'f'], inplace=False).py() == q_inserted_tab.py()
    assert qtab.py() != q_inserted_tab.py()

    qtab.upsert([6, 6.0, 'f'])
    assert qtab.py() == q_inserted_tab.py()


@pytest.mark.embedded
def test_upsert_test_insert(q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')
    q['qtab'] = qtab

    assert q_inserted_tab.py() == q.upsert(qtab, [6, 6.0, 'f'], test_insert=True).py()
    assert q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)').py() == qtab.py()
    assert q_inserted_tab.py() == q.upsert('qtab', [6, 6.0, 'f'], test_insert=True).py()
    assert q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)').py() == qtab.py()


@pytest.mark.ipc
def test_upsert_match_schema(kx, q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')
    q['qtab'] = qtab

    q.upsert('qtab', [6, 6.0, 'f'], match_schema=True)
    assert q_inserted_tab.py() == q('qtab').py()

    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q['qtab'] = qtab.py()

    with pytest.raises(kx.PyKXException):
        q.upsert('qtab', [6.0, 6, 'f'], match_schema=True)


@pytest.mark.embedded
def test_upsert_match_schema_embedded(kx, q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')
    q['qtab'] = qtab

    q.upsert('qtab', [6, 6.0, 'f'], match_schema=True)
    assert q_inserted_tab.py() == q('qtab').py()

    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q['qtab'] = qtab.py()

    qtab = q.upsert(qtab, q('([] a: 6 7; b: 6.0 7.0; c: `f`g)'), match_schema=True)
    assert q('([] a: 1 2 3 4 5 6 7; b: 1.0 2.0 3.0 4.0 5.0 6.0 7.0; c: `a`b`c`d`e`f`g)').py() ==\
        qtab.py()

    with pytest.raises(kx.PyKXException):
        q.upsert('qtab', [6.0, 6, 'f'], match_schema=True)


@pytest.mark.unlicensed
def test_dir(kx):
    assert isinstance(dir(kx.query), list)
    assert sorted(dir(kx.query)) == dir(kx.query)

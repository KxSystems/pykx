import os
from pathlib import Path
import shutil
import uuid

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
    if kx.licensed:
        assert q('select from qtab where col1=`a').py() ==\
            q.qsql.select(qtab, where=kx.Column('col1') == 'a').py()
    assert q('select from qtab where bool').py() == q.qsql.select(qtab, where=qbool).py()
    assert q('select from qtab where col1=`a,col2>0.5').py() ==\
        q.qsql.select(qtab, where=['col1=`a', 'col2>0.5']).py()
    if kx.licensed:
        assert q('select from qtab where col1=`a,col2>0.5').py() ==\
            q.qsql.select(qtab, where=(kx.Column('col1') == 'a') & (kx.Column('col2') > 0.5)).py()
        assert q('select from qtab where col1=`a,col2>0.5').py() ==\
            q.qsql.select(qtab, where=[kx.Column('col1') == 'a', kx.Column('col2') > 0.5]).py()
    assert q('select col1 from qtab').py() == q.qsql.select(qtab, columns={'col1': 'col1'}).py()
    assert q('select col1 from qtab').py() == q.qsql.select(qtab, columns='col1').py()
    assert q('select col1,col2 from qtab').py() == \
        q.qsql.select(qtab, columns=['col1', 'col2']).py()
    if kx.licensed:
        assert q('select col1,col2 from qtab').py() == \
            q.qsql.select(qtab, columns=[kx.Column('col1'), kx.Column('col2')]).py()
        assert q('select col1,col2 from qtab').py() == \
            q.qsql.select(qtab, columns=kx.Column('col1') & kx.Column('col2')).py()
    assert q('select maxCol2:max col2 from qtab').py() ==\
        q.qsql.select(qtab, columns={'maxCol2': 'max col2'}).py()
    # assert q('select maxCol2:max col2 from qtab').py() ==\
    #    q.qsql.select(qtab, columns={'maxCol2': kx.Column('col2').max()}).py()
    # assert q('select maxCol2:max col2 from qtab').py() ==\
    #    q.qsql.select(qtab, columns=kx.Column('col2', name='maxCol2').max()).py()
    # assert q('select maxCol2:max col2 from qtab').py() ==\
    #   q.qsql.select(qtab, columns=kx.QueryPhrase(kx.Column('col2', name='maxCol2').max())).py()
    assert q('select sumcols:col2+col3 from qtab').py() ==\
        q.qsql.select(qtab, columns={'sumcols': 'col2+col3'}).py()
    if kx.licensed:
        assert q('select sumcols:col2+col3 from qtab').py() ==\
            q.qsql.select(qtab, columns={'sumcols': kx.Column('col2')+kx.Column('col3')}).py()
        assert q('select sumcols:col2+col3 from qtab').py() ==\
            q.qsql.select(qtab, columns=kx.Column('col2', name='sumcols')+kx.Column('col3')).py()
    assert q('select maxCol2:max col2 by groupcol:col1 from qtab').py() ==\
        q.qsql.select(qtab, columns={'maxCol2': 'max col2'}, by={'groupcol': 'col1'}).py()
    if kx.licensed:
        assert q('select maxCol2:max col2 by groupcol:col1 from qtab').py() ==\
            q.qsql.select(qtab, columns=kx.Column('col2', name='maxCol2').max(),
                          by={'groupcol': kx.Column('col1')}).py()
        assert q('select maxCol2:max col2 by groupcol:col1 from qtab').py() ==\
            q.qsql.select(qtab, columns=kx.Column('col2', name='maxCol2').max(),
                          by=kx.Column('col1', name='groupcol')).py()
        assert q('select maxCol2:max col2 by groupcol:col1 from qtab').py() ==\
            q.qsql.select(qtab, columns=kx.Column('col2', name='maxCol2').max(),
                          by=kx.QueryPhrase(kx.Column('col1', name='groupcol'))).py()
    assert q('select minCol2:min col2,max col3 by col1 from qtab where col3<0.5,col2>0.7').py() ==\
        q.qsql.select(qtab,
                      columns={'minCol2': 'min col2', 'col3': 'max col3'},
                      by={'col1': 'col1'},
                      where=['col3<0.5', 'col2>0.7']
        ).py()
    if kx.licensed:
        assert q('select minCol2:min col2,max col3 by col1 from qtab where col3<0.5,col2>0.7').py()\
            == q.qsql.select(qtab,
                             columns=kx.Column('col2', name='minCol2').min()
                             & kx.Column('col3').max(),
                             by=kx.Column('col1'),
                             where=(kx.Column('col3') < 0.5) & (kx.Column('col2') > 0.7)
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
    cwd = Path(os.getcwd())
    tmpdir = cwd / str(uuid.uuid4().hex)[:7]
    try:
        db = kx.DB(path=tmpdir)
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
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)


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

    assert qtab.insert([6, 6.0, 'f'], inplace=False).py() == q_inserted_tab.py()
    assert qtab.py() != q_inserted_tab.py()

    qtab.insert([6, 6.0, 'f'])
    assert qtab.py() == q_inserted_tab.py()


@pytest.mark.pandas_api
def test_table_upsert_method(q):
    qtab = q('([] a: 1 2 3 4 5; b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([] a: 1 2 3 4 5 6; b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')

    assert qtab.upsert([6, 6.0, 'f'], inplace=False).py() == q_inserted_tab.py()
    assert qtab.py() != q_inserted_tab.py()

    qtab.upsert([6, 6.0, 'f'])
    assert qtab.py() == q_inserted_tab.py()


@pytest.mark.pandas_api
def test_keyed_table_insert_method(q):
    qtab = q('([a: 1 2 3 4 5] b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([a: 1 2 3 4 5 6] b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')

    assert qtab.insert([6, 6.0, 'f'], inplace=False).py() == q_inserted_tab.py()
    assert qtab.py() != q_inserted_tab.py()

    qtab.insert([6, 6.0, 'f'])
    assert qtab.py() == q_inserted_tab.py()


@pytest.mark.pandas_api
def test_keyed_table_upsert_method(q):
    qtab = q('([a: 1 2 3 4 5] b: 1.0 2.0 3.0 4.0 5.0; c: `a`b`c`d`e)')
    q_inserted_tab = q('([a: 1 2 3 4 5 6] b: 1.0 2.0 3.0 4.0 5.0 6.0; c: `a`b`c`d`e`f)')

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


def test_pythonic_query(kx):
    table = kx.q('([] x:`a`b`c;x1:1 2 3;x2:`a`e`g;x11:0 3 3;b:011b)')
    c='c'
    kx.q['cvar'] = c

    assert kx.q('~', table[0], table.select(where=kx.Column('x') == 'a'))
    assert kx.q('~', table[0], table.delete(where=kx.Column('x').isin(['b', 'c'])))
    assert kx.q('~', table[0], table.delete(where=kx.Column('b')))
    assert kx.q('~', table[1:3], table.select(where=kx.Column('b')))
    assert kx.q('~', table[2], table.select(where=kx.Column('x') == c))
    assert kx.q('~', table[0], table.select(where=kx.Column('x') == kx.Column('x2')))
    assert kx.q('~', table[2], table.select(where=kx.Column('x') == kx.Variable('cvar')))
    assert kx.q('~', table[1:3], table.select(where=kx.Column('x1') > 1))
    assert kx.q('~', table[1:3], table.select(where=kx.Column('x1') >= 2))
    assert kx.q('~', table[0:2], table.select(where=kx.Column('x').isin(['a', 'b'])))
    assert kx.q('~', table[1], table.select(where=kx.Column('x').isin('b')))
    assert kx.q('~', table[2], table.select(where=kx.Column('x').isin(kx.Variable('cvar'))))
    assert kx.q('~', table[0], table.select(where=kx.Column('x') == 'a'))
    assert kx.q('~', table[0], table.select(where=kx.ParseTree(kx.q.parse(b'x=`a')).enlist()))
    assert kx.q('~', table[0], table.select(where=kx.QueryPhrase([kx.q.parse(b'x=`a')])))
    assert kx.q('~', table[0], table.select(where=kx.QueryPhrase(kx.Column('x') == 'a')))
    assert kx.q('~', table[0:2], table.select(where=(kx.Column('x') == 'a')
                                              | (kx.Column('x') == 'b')))
    assert kx.q('~', table[0:2], table.select(where=(kx.Column('b')
                                                     == (kx.Column('x11') > kx.Column('x1')))))
    assert kx.q('~', table[2], table.select(where=kx.QueryPhrase(kx.Column('x1')
                                                                 == kx.Column('x1').max())))
    assert kx.q('~', table[2], table.select(where=kx.Column('x11').msum(2) > 4))
    assert all(kx.q('{update x11msum2:2 msum x11 from x}', table)
               == table.update({'x11msum2': kx.Column('x11').msum(2)}))
    assert all(kx.q('{select by neg b from x}', table)
               == table.select(by={'b': kx.Column('b').call('neg')}))
    kx.q('myneg:{neg x}')
    assert all(kx.q('{select by neg b from x}', table)
               == table.select(by={'b': kx.Column('b').call(kx.Variable('myneg'))}))
    assert all(kx.q('{select neg b from x}', table)
               == table.select(columns=kx.Column('b').call('neg')))
    assert all(kx.q('{select negb:neg b from x}', table)
               == table.select(columns=kx.Column('b', name='negb').call('neg')))
    assert all(kx.q('{select negb:neg b from x}', table)
               == table.select(columns=kx.Column(name='negb', value=[kx.q('neg'), 'b'])))
    assert all(kx.q('{exec neg b from x}', table)
               == table.exec(columns=kx.Column('b').call('neg')))
    assert ({'asA': 'a', 'negB': [kx.q('neg'), 'b']}
            == (kx.Column('a', name='asA') & kx.Column('b', name='negB').call('neg')).to_dict())
    assert kx.q('~', kx.q('{select x, negx1:neg x1 by x11, notB:not b from x}', table),
                table.select(columns=['x', kx.Column('x1', name='negx1').call('neg')],
                             by=['x11', kx.Column('b', name='notB').call('not')]))
    assert all(kx.q('{select max b from x}', table)
               == table.select(columns=kx.Column('b').max()))
    assert kx.q('~', kx.q('{select max b, x from x}', table),
                table.select(columns=kx.Column('b').max() & kx.Column('x')))
    assert kx.q('~', kx.q('{select maxB:max b from x}', table),
                table.select(columns=kx.Column('b', name='maxB').max()))
    assert kx.q('~', kx.q('{select maxB:max b from x}', table),
                table.select(columns={'maxB': kx.Column('b').max()}))
    t= kx.q('([] c1:30?`a`b`c;c2:30?`d`e`f;c3:30?4;c4:30?4)')
    a = kx.q('{select from x where c3=(max;c3) fby ([] c1;c4)}', t)
    b = t.select(where=kx.Column('c3') == [kx.q.fby, [kx.q.enlist, kx.q.max, 'c3'],
                                           kx.ParseTree.table(['c1', 'c4'])])
    assert kx.q('~', a, b)
    c = t.select(where=kx.Column('c3') == kx.Column.fby(['c1', 'c4'], 'max', 'c3', by_table=True))
    assert kx.q('~', a, c)
    d = t.select(where=kx.Column('c3') == kx.Column.fby({'c1': 'c1', 'c4': 'c4'}, 'max', 'c3'))
    assert kx.q('~', a, d)
    a = table.select(where=(kx.Column('x') == 'a') &(
        kx.Column('x1') == 1) & (kx.Column('x11') == 0))
    b = table.select(where=((kx.Column('x') == 'a') &(
        kx.Column('x1') == 1)) & (kx.Column('x11') == 0))
    c = table.select(where=(kx.Column('x') == 'a') &(
        (kx.Column('x1') == 1) & (kx.Column('x11') == 0)))
    assert kx.q('~', table[0], a)
    assert kx.q('~', table[0], b)
    assert kx.q('~', table[0], c)

    with pytest.raises(TypeError) as err:
        kx.Column('x') & 1
        assert 'cannot `&` off a `pykx.Column`.' in str(err)

    with pytest.raises(TypeError) as err:
        (kx.Column('x') & kx.Column('x')) & 1
        assert 'cannot `&` off a `pykx.QueryPhrase`.' in str(err)

    t= kx.q('''
             ([] Primary:`1a`1a`1a`1a`2e`2e`2e`2e;
             Name:`AXA`FLO`FLO`AXA`AXA`ROT`ROT`ROT; Count: 11 1 60 14 1 1 6 4)''')
    a = kx.q('''{select from x where
             i=({exec first ind from x where Count=max Count};([]Count;ind:i)) fby Primary}''', t)
    b = t.select(
        where=kx.Column('i')
        == [kx.q.fby, [kx.q.enlist, kx.q('{exec first ind from x where Count=max Count}'),
                       kx.ParseTree.table({'Count': 'Count', 'ind': 'i'})], kx.Column('Primary')])
    c = t.select(where=kx.Column('i')
                 == kx.ParseTree.fby('Primary', '{exec first ind from x where Count=max Count}',
                                     {'Count': 'Count', 'ind': 'i'}))
    assert kx.q('~', a, b)
    assert kx.q('~', a, c)

    t=kx.q('([] a:1 2 3;b:4 5 6;c:7 8 9;d:10 11 12)')
    kx.q['t'] = t
    a = kx.q('select a:a(,)/:b from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="/:"))
    c = t.select(kx.Column('a').call(b',', kx.Column('b'), iterator="/:"))
    d = t.select(kx.Column('a').call(kx.CharVector(','), kx.Column('b'), iterator="/:"))
    assert kx.q('~', a, b)
    assert kx.q('~', a, c)
    assert kx.q('~', a, d)
    a = kx.q('select a:a(,)\\:b from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="\\:"))
    assert kx.q('~', a, b)
    a = kx.q('select a:a(,)\\:/:b from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="\\:/:"))
    assert kx.q('~', a, b)
    a = kx.q('select a:a(,)/:\\:b from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="/:\\:"))
    assert kx.q('~', a, b)
    a = kx.q('select {99,x} each a from t')
    b = t.select(kx.Column('a').call('{99,x}', iterator="each"))
    assert kx.q('~', a, b)
    a = kx.q('select {99,x} peach a from t')
    b = t.select(kx.Column('a').call('{99,x}', iterator="peach"))
    assert kx.q('~', a, b)
    a = kx.q('select a:(,[a]) each b from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="each", project_args=[0]))
    assert kx.q('~', a, b)
    a = kx.q('select a:(,[b]) each a from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="each",
                                     col_arg_ind=1, project_args=[0]))
    assert kx.q('~', a, b)
    a = kx.q('select (,[;b]) each a  from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="each", project_args=[1]))
    assert kx.q('~', a, b)
    a = kx.q('select (+) scan a from t')
    b = t.select(kx.Column('a').call('+', iterator="scan"))
    assert kx.q('~', a, b)
    a = kx.q('select (+\\)a from t')
    b = t.select(kx.Column('a').call('+', iterator='\\'))
    assert kx.q('~', a, b)
    a = kx.q('select max a,(+) over b from t')
    b = t.select(kx.Column('a').max() & kx.Column('b').call('+', iterator="over"))
    assert kx.q('~', a, b)
    a = kx.q('select max a,(+/)b from t')
    b = t.select(kx.Column('a').max() & kx.Column('b').call('+', iterator="/"))
    assert kx.q('~', a, b)
    a = kx.q('select a:a(,)\'b from t')
    b = t.select(kx.Column('a').call(',', kx.Column('b'), iterator="'"))
    assert kx.q('~', a, b)
    a = kx.q('select (-) prior a from t')
    b = t.select(kx.Column('a').call('-', iterator="prior"))
    assert kx.q('~', a, b)

    assert 'x' in table.columns.py()
    assert 'x' not in table.delete(kx.Column('x')).columns.py()

    assert all([i in table.columns.py() for i in ['x', 'x1']])
    assert all([i not in table.delete(kx.Column('x') & kx.Column('x1')) for i in ['x', 'x1']])


def test_pythonic_query_ipc(kx, q_port):
    q = kx.SyncQConnection(port=q_port)
    q('table:([] x:`a`b`c;x1:1 2 3;x2:`a`e`g;x11:0 3 3;b:011b)')
    table = q['table']
    c='c'
    q['cvar'] = c
    assert q('{table[enlist 0]~x}', q.qsql.select('table', where=kx.Column('x') == 'a'))
    assert q('{table[1 2]~x}', q.qsql.select('table', where=kx.Column('b')))
    assert q('{table[enlist 2]~x}', q.qsql.select('table', where=kx.Column('x') == c))
    assert q('{table[enlist 0]~x}', q.qsql.select('table', where=kx.Column('x') == kx.Column('x2')))
    assert q('{table[enlist 2]~x}', q.qsql.select('table', where=kx.Column('x')
                                                  == kx.Variable('cvar')))
    assert q('{table[1 2]~x}', q.qsql.select('table', where=kx.Column('x1') > 1))
    assert q('{table[1 2]~x}', q.qsql.select('table', where=kx.Column('x1') >= 2))
    assert q('{table[0 1]~x}', q.qsql.select('table', where=kx.Column('x').isin(['a', 'b'])))
    assert q('{table[enlist 1]~x}', q.qsql.select('table', where=kx.Column('x').isin('b')))
    assert q('{table[enlist 2]~x}', q.qsql.select('table',
                                                  where=kx.Column('x').isin(kx.Variable('cvar'))))
    assert q('{table[enlist 0]~x}', q.qsql.select('table', where=kx.Column('x') == 'a'))
    assert q('{table[enlist 0]~x}', q.qsql.select('table',
                                                  where=kx.ParseTree(q.parse(b'x=`a')).enlist()))
    assert q('{table[enlist 0]~x}', q.qsql.select('table',
                                                  where=kx.QueryPhrase([q.parse(b'x=`a')])))
    assert q('{table[enlist 0]~x}', q.qsql.select('table',
                                                  where=kx.QueryPhrase(kx.Column('x') == 'a')))
    assert q('{table[0 1]~x}', q.qsql.select('table', where=(kx.Column('x') == 'a')
                                             | (kx.Column('x') == 'b')))
    assert q('{table[0 1]~x}',
             q.qsql.select('table', where=(kx.Column('b') == (kx.Column('x11') > kx.Column('x1')))))
    assert q('{table[enlist 2]~x}',
             q.qsql.select('table', where=kx.QueryPhrase(kx.Column('x1') == kx.Column('x1').max())))
    assert q('{table[enlist 2]~x}', q.qsql.select('table', where=kx.Column('x11').msum(2) > 4))
    assert q('{y~update x11msum2:2 msum x11 from table}',
             q.qsql.update('table', ({'x11msum2': kx.Column('x11').msum(2)})))
    assert q('{y~select by neg b from table}',
             q.qsql.select('table', by={'b': kx.Column('b').call('neg')}))
    q('myneg:{neg x}')
    assert q('{y~select by neg b from table}',
             q.qsql.select('table', by={'b': kx.Column('b').call(kx.Variable('myneg'))}))
    assert q('{y~select neg b from table}', q.qsql.select('table',
                                                          columns=kx.Column('b').call('neg')))
    assert q('{y~select negb:neg b from table}',
             q.qsql.select('table', columns=kx.Column('b', name='negb').call('neg')))
    assert q('{y~select negb:neg b from table}',
             q.qsql.select('table', columns=kx.Column(name='negb', value=[q('neg'), 'b'])))
    assert q('{y~exec neg b from table}', q.qsql.exec('table', columns=kx.Column('b').call('neg')))
    assert ({'asA': 'a', 'negB': [q('neg'), 'b']}
            == (kx.Column('a', name='asA')& kx.Column('b', name='negB').call('neg')).to_dict())
    assert q('~', q('{select x, negx1:neg x1 by x11, notB:not b from table}',
                    q.qsql.select('table',
                                  columns=['x', kx.Column('x1', name='negx1').call('neg')],
                                  by=['x11', kx.Column('b', name='notB').call('not')])))
    assert q('~', q('{select max b from x}', table),
             q.qsql.select('table', columns=kx.Column('b').max()))
    assert q('~', q('{select max b, x from x}', table),
             q.qsql.select('table', columns=kx.Column('b').max() & kx.Column('x')))
    assert q('~', q('{select maxB:max b from x}', table),
             q.qsql.select('table', columns=kx.Column('b', name='maxB').max()))
    assert q('~', q('{select maxB:max b from x}', table),
             q.qsql.select('table', columns={'maxB': kx.Column('b').max()}))

    a = q.qsql.select('table', where=(kx.Column('x') == 'a') &(
        kx.Column('x1') == 1) & (kx.Column('x11') == 0))
    b = q.qsql.select('table', where=((kx.Column('x') == 'a') &(
        kx.Column('x1') == 1)) & (kx.Column('x11') == 0))
    c = q.qsql.select('table', where=(kx.Column('x') == 'a') &(
        (kx.Column('x1') == 1) & (kx.Column('x11') == 0)))
    assert q('{x[enlist 0]~y}', table, a)
    assert q('{x[enlist 0]~y}', table, b)
    assert q('{x[enlist 0]~y}', table, c)

    q('t:([] c1:30?`a`b`c;c2:30?`d`e`f;c3:30?4;c4:30?4)')
    a = q('select from t where c3=(max;c3) fby ([] c1;c4)')
    b = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby(['c1', 'c4'], 'max', 'c3',
                                                                  by_table=True))
    assert q('~', a, b).py()
    c = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby({'c1': 'c1', 'c4': 'c4'}, 'max',
                                                                  'c3'))
    assert q('~', a, c).py()
    d = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby(['c1', 'c4'], 'max',
                                                                  kx.Column('c3'), by_table=True))
    assert q('~', a, d).py()
    e = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby(kx.Column('c1') & kx.Column('c4'),
                                                                  'max', kx.Column('c3')))
    assert q('~', a, e).py()
    f = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby(kx.toq({'c1': 'c1', 'c4': 'c4'}),
                                                                  'max', 'c3'))
    assert q('~', a, f).py()
    a = q('select from t where c3=(max;c3) fby c1')
    b = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby(kx.Column('c1'), 'max',
                                                                  kx.Column('c3')))
    assert q('~', a, b).py()

    a = q('select from t where c3=({max x`c3};([] c3;c4)) fby c1')
    b = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby('c1', '{max x`c3}', ['c3', 'c4'],
                                                                  data_table=True))
    assert q('~', a, b).py()
    c = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby('c1', '{max x`c3}',
                                                                  {'c3': 'c3', 'c4': 'c4'}))
    assert q('~', a, c).py()
    d = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby('c1', '{max x`c3}', ['c3', 'c4'],
                                                                  data_table=True))
    assert q('~', a, d).py()
    e = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby('c1', '{max x`c3}',
                                                                  kx.Column('c3')
                                                                  & kx.Column('c4')))
    assert q('~', a, e).py()
    f = q.qsql.select('t', where=kx.Column('c3') == kx.Column.fby('c1', '{max x`c3}',
                                                                  kx.toq({'c3': 'c3', 'c4': 'c4'})))
    assert q('~', a, f).py()

    a = q.qsql.select('t', columns=kx.Column('c3').min().name('min_c3')
                      & kx.Column('c4').mavg(4).max(), by=kx.Column('c1'))
    b = q('select min_c3:min c3, max 4 mavg c4 by c1 from t')
    assert q('~', a, b)

    q('''t:([] Primary:`1a`1a`1a`1a`2e`2e`2e`2e;
         Name:`AXA`FLO`FLO`AXA`AXA`ROT`ROT`ROT; Count: 11 1 60 14 1 1 6 4)''')
    a = q('''select from t where
             i=({exec first ind from x where Count=max Count};([]Count;ind:i)) fby Primary''')
    b = q.qsql.select(
        't', where=kx.Column('i') == kx.ParseTree.fby(
            'Primary', '{exec first ind from x where Count=max Count}',
            {'Count': 'Count', 'ind': 'i'}))
    assert q('~', a, b).py()

    t=q('([] a:1 2 3;b:4 5 6;c:7 8 9;d:10 11 12)')
    q['t'] = t
    a = q('select a:a(,)/:b from t')
    b = q.qsql.select('t', kx.Column('a').call(',', kx.Column('b'), iterator="/:"))
    c = q.qsql.select('t', kx.Column('a').call(b',', kx.Column('b'), iterator="/:"))
    d = q.qsql.select('t', kx.Column('a').call(kx.CharVector(','), kx.Column('b'), iterator="/:"))
    assert q('~', a, b).py()
    assert q('~', a, c).py()
    assert q('~', a, d).py()
    a = q('select a:a(,)\\:b from t')
    b = q.qsql.select('t', kx.Column('a').call(',', kx.Column('b'), iterator="\\:"))
    assert q('~', a, b).py()
    a = q('select a:a(,)\\:/:b from t')
    b = q.qsql.select('t', kx.Column('a').call(',', kx.Column('b'), iterator="\\:/:"))
    assert q('~', a, b).py()
    a = q('select a:a(,)/:\\:b from t')
    b = q.qsql.select('t', kx.Column('a').call(',', kx.Column('b'), iterator="/:\\:"))
    assert q('~', a, b).py()
    a = q('select {99,x} each a from t')
    b = q.qsql.select('t', kx.Column('a').call('{99,x}', iterator="each"))
    assert q('~', a, b).py()
    a = q('select {99,x} peach a from t')
    b = q.qsql.select('t', kx.Column('a').call('{99,x}', iterator="peach"))
    assert q('~', a, b).py()
    a = q('select a:(,[a]) each b from t')
    b = q.qsql.select('t', kx.Column('a').call(',',
                                               kx.Column('b'), iterator="each", project_args=[0]))
    assert q('~', a, b).py()
    a = q('select a:(,[b]) each a from t')
    b = q.qsql.select('t', kx.Column('a').call(',', kx.Column('b'), iterator="each",
                                               col_arg_ind=1, project_args=[0]))
    assert q('~', a, b).py()
    a = q('select (,[;b]) each a  from t')
    b = q.qsql.select('t', kx.Column('a').call(',',
                                               kx.Column('b'), iterator="each", project_args=[1]))
    assert q('~', a, b).py()
    a = q('select (+) scan a from t')
    b = q.qsql.select('t', kx.Column('a').call('+', iterator="scan"))
    assert q('~', a, b).py()
    a = q('select (+\\)a from t')
    b = q.qsql.select('t', kx.Column('a').call('+', iterator='\\'))
    assert q('~', a, b).py()
    a = q('select max a,(+) over b from t')
    b = q.qsql.select('t', kx.Column('a').max() & kx.Column('b').call('+', iterator="over"))
    assert q('~', a, b).py()
    a = q('select max a,(+/)b from t')
    b = q.qsql.select('t', kx.Column('a').max() & kx.Column('b').call('+', iterator="/"))
    assert q('~', a, b).py()
    a = q('select a:a(,)\'b from t')
    b = q.qsql.select('t', kx.Column('a').call(',', kx.Column('b'), iterator="'"))
    assert q('~', a, b).py()
    a = q('select (-) prior a from t')
    b = q.qsql.select('t', kx.Column('a').call('-', iterator="prior"))
    assert q('~', a, b).py()


@pytest.mark.unlicensed
def test_column_licensed(kx):
    if not kx.licensed:
        with pytest.raises(kx.LicenseException) as err:
            kx.Column('s')
            assert "kx.Column" in str(err)

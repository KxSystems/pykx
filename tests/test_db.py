import os
import shutil
import warnings

# Do not import pykx here - use the `kx` fixture instead!
import pytest


@pytest.mark.order(1)
def test_creation(kx):
    # Definition of qtab would break kx.DB prior to use of .Q.pt
    kx.q('qtab:([]100?1f;100?1f)')
    db = kx.DB(path='db')
    tab = kx.Table(data={
        'date': kx.q('2015.01.01 2015.01.01 2015.01.02 2015.01.02'),
        'ti': kx.q('09:30:00 09:31:00 09:30:00 09:31:00'),
        'p': kx.q('101 102 101.5 102.5'),
        'sz': kx.q('100 200 150 210'),
        'sym': kx.q('`a`b`b`c')
    })
    db.create(tab, 't', 'date', by_field='sym', sym_enum='sym')
    assert db.tables == ['t']


@pytest.mark.order(2)
def test_create_errors(kx):
    db = kx.DB(path='err_db')
    tab = kx.Table(data={
        'date': kx.q('2015.01.01 2015.01.01 2015.01.02 2015.01.02'),
        'ti': kx.q('09:30:00 09:31:00 09:30:00 09:31:00'),
        'p': kx.q('101 102 101.5 102.5'),
        'sz': kx.q('100 200 150 210'),
        'sym': kx.q('`a`b`b`c')
    })
    with pytest.raises(kx.QError) as err:
        db.create(1, 't', 'ti', by_field='sym', sym_enum='sym')
    assert 'Supplied table must be' in str(err.value)
    with pytest.raises(kx.QError) as err:
        db.create(tab, 't', 'p', by_field='sym', sym_enum='sym')
    assert 'Unsupported type:' in str(err.value)
    with pytest.raises(kx.QError) as err:
        db.create(tab, 't', 'no_col', by_field='sym', sym_enum='sym')
    assert 'Partition column no_col not in supplied' in str(err.value)


@pytest.mark.order(3)
def test_load_1(kx):
    db = kx.db.DB()
    assert db.tables is None
    db.load('db')
    assert db.tables == ['t']
    assert type(db.t) == kx.PartitionedTable # noqa: E721
    with pytest.raises(kx.QError) as err:
        db.load('../db')
    assert 'Attempting to reload existing' in str(err.value)
    with pytest.raises(kx.QError) as err:
        db.load('test')
    assert 'Only one kdb+ database' in str(err.value)
    with pytest.raises(kx.QError) as err:
        db.load('../pyproject.toml', overwrite=True)
    assert 'Provided path is a file' in str(err.value)
    with pytest.raises(kx.QError) as err:
        db.load('doesNotExist', overwrite=True)
    assert 'Unable to find object at specified path' in str(err.value)


@pytest.mark.order(4)
def test_load_2(kx):
    db = kx.DB(path='db')
    assert db.tables == ['t']
    assert type(db.t) == kx.PartitionedTable # noqa: E721


@pytest.mark.order(5)
def test_list(kx):
    db = kx.DB()
    db.load('db')
    print(db.tables)
    db_cols = db.list_columns('t')
    assert db_cols == ['sym', 'ti', 'p', 'sz']
    with pytest.raises(kx.QError) as err:
        db.list_columns('no_tab')
    assert 'Column listing not possible' in str(err.value)


@pytest.mark.order(6)
def test_column_add(kx):
    db = kx.DB()
    db.load('db')
    assert ['sym', 'ti', 'p', 'sz'] == db.list_columns('t')
    db.add_column('t', 'vol', kx.IntAtom.null)
    db_cols = db.list_columns('t')
    assert ['sym', 'ti', 'p', 'sz', 'vol'] == db_cols


@pytest.mark.order(7)
def test_column_reorder(kx):
    db = kx.DB()
    db.load('db')
    db.reorder_columns('t', ['vol', 'sym', 'sz', 'p', 'ti'])
    assert ['vol', 'sym', 'sz', 'p', 'ti'] == db.list_columns('t')


@pytest.mark.order(8)
def test_column_rename(kx):
    db = kx.DB()
    db.load('db')
    db.rename_column('t', 'p', 'price')
    assert ['vol', 'sym', 'sz', 'price', 'ti'] == db.list_columns('t')
    with pytest.raises(kx.QError) as err:
        db.rename_column('t', 'no_col', 'upd')
    assert "Specified column 'no_col'" in str(err.value)


@pytest.mark.order(9)
def test_column_delete(kx):
    db = kx.DB()
    db.load('db')
    db.delete_column('t', 'vol')
    assert ['sym', 'sz', 'price', 'ti']== db.list_columns('t')
    with pytest.raises(kx.QError) as err:
        db.delete_column('t', 'no_col')
    assert "Specified column 'no_col'" in str(err.value)


@pytest.mark.order(10)
def test_column_find(kx):
    db = kx.DB()
    db.load('db')
    assert None == db.find_column('t', 'price') # noqa: E711
    with pytest.raises(kx.QError) as err:
        db.find_column('t', 'no_col')
    assert 'Requested column not found' in str(err.value)


@pytest.mark.order(11)
def test_column_set_attr(kx):
    db = kx.DB()
    db.load('db')
    assert 'g' not in kx.q.qsql.exec(kx.q.meta(db.t), columns='a')
    db.set_column_attribute('t', 'sym', 'grouped')
    assert 'g' in kx.q.qsql.exec(kx.q.meta(db.t), columns='a')
    with pytest.raises(kx.QError) as err:
        db.set_column_attribute('t', 'no_col', 'unique')
    assert "Specified column 'no_col'" in str(err.value)


@pytest.mark.order(12)
def test_column_clear_attr(kx):
    db = kx.DB()
    db.load('db')
    assert 'g' in kx.q.qsql.exec(kx.q.meta(db.t), columns='a')
    db.clear_column_attribute('t', 'sym')
    assert 'g' not in kx.q.qsql.exec(kx.q.meta(db.t), columns='a')
    with pytest.raises(kx.QError) as err:
        db.clear_column_attribute('t', 'no_col')
    assert "Specified column 'no_col'" in str(err.value)


@pytest.mark.order(13)
def test_column_set_type(kx):
    db = kx.DB()
    db.load('db')
    assert b'f' in kx.q.qsql.exec(kx.q.meta(db.t), columns='t').py()
    db.set_column_type('t', 'price', kx.LongAtom)
    assert b'f' not in kx.q.qsql.exec(kx.q.meta(db.t), columns='t').py()
    with pytest.raises(kx.QError) as err:
        db.set_column_type('t', 'price', kx.GUIDAtom)
    assert "to type: <class 'pykx.wrappers.GUIDAtom'>" in str(err.value)
    with pytest.raises(kx.QError) as err:
        db.set_column_attribute('t', 'no_col', kx.GUIDAtom)
    assert "Specified column 'no_col'" in str(err.value)


@pytest.mark.order(14)
def test_column_copy(kx):
    db = kx.DB()
    db.load('db')
    assert ['sym', 'sz', 'price', 'ti'] == db.list_columns('t')
    db.copy_column('t', 'sz', 'size')
    assert ['sym', 'sz', 'price', 'ti', 'size'] == db.list_columns('t')
    assert all(kx.q.qsql.select(db.t, 'sz')['sz'] == kx.q.qsql.select(db.t, 'size')['size']) # noqa: E501
    with pytest.raises(kx.QError) as err:
        db.copy_column('t', 'no_col', 'new_name')
    assert "Specified column 'no_col'" in str(err.value)


@pytest.mark.order(15)
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_column_apply(kx):
    db = kx.DB()
    db.load('db')
    assert all([100, 200, 150, 210] == kx.q.qsql.select(db.t, 'size')['size'])
    db.apply_function('t', 'size', kx.q('2*'))
    assert all([200, 400, 300, 420] == kx.q.qsql.select(db.t, 'size')['size'])
    db.apply_function('t', 'size', lambda x: x.np()/2)
    assert all([100, 200, 150, 210] == kx.q.qsql.select(db.t, 'size')['size'])
    with pytest.raises(RuntimeError) as err:
        db.apply_function('t', 'size', 2)
    assert "Provided 'function' is not callable" in str(err.value)


@pytest.mark.order(16)
def test_table_rename(kx):
    db = kx.DB()
    db.load('db')
    assert db.tables == ['t']
    db.rename_table('t', 'trades')
    assert db.tables == ['trades']
    assert type(db.trades) == kx.PartitionedTable # noqa: E721


@pytest.mark.order(17)
def test_db_fill(kx):
    db = kx.DB(path='db')
    assert db.tables == ['trades']
    qtab = kx.Table(data={
        'col1': kx.random.random(1000, 10.0),
        'col2': kx.random.random(1000, 10)
    })
    db.create(qtab, 'newtab', kx.q('2015.01.02'))
    with pytest.raises(kx.QError) as err:
        db.partition_count()
    assert '2015.01.01/newtab. OS reports: No such file or directory' in str(err.value)
    db.fill_database()
    parts = db.partition_count()
    all(kx.q.qsql.exec(parts.values(), 'newtab') == [0, 1000])


@pytest.mark.order(18)
def test_load_warning(kx):
    kx.q('`:./db/2015.01.01/table/ set .Q.en[`:./db;]([] ti:09:30:00 09:31:00; p:101 102f; sz:100 200; sym:`a`b)') # noqa: E501
    kx.q('`:./db/2015.01.02/table/ set .Q.en[`:./db;]([] ti:09:30:00 09:31:00; p:101.5 102.5; sz:150 210;sym:`b`c)') # noqa: E501
    db = kx.db.DB()
    assert db.tables is None
    with warnings.catch_warnings(record=True) as w:
        db.load('db')
    assert 'A database table "table" would overwrite' in str(w[-1].message)
    assert type(db.table) != kx.PartitionedTable # noqa: E721
    assert type(db.table.table) == kx.PartitionedTable # noqa: E721


@pytest.mark.order(19)
def test_compress(kx):
    zd_cache = kx.q.z.zd
    compress = kx.Compress(kx.CompressionAlgorithm.gzip, level=8)
    db = kx.DB(path='db')
    qtab = kx.Table(data={
        'col1': kx.random.random(1000, 10.0),
        'col2': kx.random.random(1000, 10)
    })
    db.create(qtab, 'comptab', kx.q('2015.01.02'), compress=compress)
    db.fill_database()
    assert zd_cache == kx.q.z.zd
    compress_info = kx.q('-21!key`:./2015.01.02/comptab/col1')
    assert type(compress_info) == kx.Dictionary
    assert compress_info['algorithm'].py() == 2
    assert compress_info['zipLevel'].py() == 8


def test_enumerate(kx):
    tab = kx.Table(data={
        'date': kx.q('2015.01.01 2015.01.01 2015.01.02 2015.01.02'),
        'ti': kx.q('09:30:00 09:31:00 09:30:00 09:31:00'),
        'p': kx.q('101 102 101.5 102.5'),
        'sz': kx.q('100 200 150 210'),
        'sym': kx.q('`a`b`b`c')
    })
    db = kx.DB(path='db')
    entab = db.enumerate(tab)
    assert 20 == entab['sym'].t
    assert 'sym' == kx.q.key(entab['sym'])
    assert type(kx.q.value(entab['sym'])) == kx.SymbolVector # noqa: E721
    entab1 = db.enumerate(tab, sym_file='mysym')
    assert 20 == entab1['sym'].t
    assert 'mysym' == kx.q.key(entab1['sym'])
    assert type(kx.q.value(entab1['sym'])) == kx.SymbolVector # noqa: E721


def test_partition_count(kx):
    db = kx.DB(path='db')
    fullview = db.partition_count()
    assert type(fullview) == kx.Dictionary # noqa: E721
    assert 2 == len(fullview)
    subview = db.partition_count(subview=kx.q('2015.01.02'))
    assert type(subview) == kx.Dictionary # noqa: E721
    assert 1 == len(subview)


def test_subview(kx):
    db = kx.DB(path='db')
    db.subview([kx.q('2015.01.01')])
    qtab = kx.q.qsql.select(db.trades)
    assert type(qtab) == kx.Table # noqa: E721
    assert 2 == len(qtab)
    db.subview()
    qtab = kx.q.qsql.select(db.trades)
    assert type(qtab) == kx.Table # noqa: E721
    assert 4 == len(qtab)


@pytest.mark.isolate
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_beta():
    import pykx as kx
    with pytest.raises(kx.QError) as err:
        kx.DB()
    assert 'Attempting to use a beta feature "Data' in str(err.value)


@pytest.mark.order(-1)
def test_cleanup(kx):
    shutil.rmtree('db')
    assert True

import os
import shutil

# Do not import pykx here - use the `kx` fixture instead!
import pytest


def custom_api(x, y):
    """
    A Python custom API which will be made available
    on a process
    """
    return kx.q.til(x) + y # noqa: F821


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_tick_init(kx):
    trade_schema = kx.schema.builder({
        'time': kx.TimespanAtom,
        'sym': kx.SymbolAtom,
        'px': kx.FloatAtom})
    tick = kx.tick.TICK(port=5030)
    assert tick('1b')
    assert tick('system"p"').py() == 5030
    assert tick('.tick.tabs').py() == []
    with pytest.raises(kx.QError) as err:
        tick.set_tables({'quote': kx.schema.builder({'px': kx.FloatAtom})})
    assert "'time' and 'sym' must be first" in str(err.value)
    tick.set_tables({'trade': trade_schema})
    assert tick('.tick.tabs').py() == ['trade']
    tick.stop()

    tick = kx.tick.TICK(port=5030, tables={'trades': trade_schema})
    assert tick('1b')
    assert tick('.tick.tabs').py() == ['trades']
    tick.stop()

    tick.restart()
    assert tick('1b')
    assert tick('.tick.tabs').py() == ['trades']
    tick.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_tick_start(kx):
    trade_schema = kx.schema.builder({
        'time': kx.TimespanAtom,
        'sym': kx.SymbolAtom,
        'px': kx.FloatAtom})
    tick = kx.tick.TICK(port=5030, tables={'trades': trade_schema})
    assert tick('.tick.tabs').py() == ['trades']
    with pytest.raises(kx.QError) as err:
        tick('.u.t')
    assert '.u.t' in str(err)
    tick.start()
    assert tick('.u.t').py() == ['trades']
    tick.stop()

    tick = kx.tick.TICK(port=5030)
    with pytest.raises(kx.QError) as err:
        tick.start()
    assert 'Unable to initialise TICKERPLANT without tables' in str(err.value)
    tick.stop()

    tick.restart()
    assert tick('1b')
    assert tick('.tick.tabs').py() == []
    tick.stop()

    tick = kx.tick.TICK(port=5030, tables={'trade': trade_schema}, log_directory='tick_logs')
    tick.start()
    assert 'tick_logs' in os.listdir()
    assert f'log{kx.DateAtom("today")}' in os.listdir('tick_logs')
    tick.stop()
    shutil.rmtree('tick_logs')


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_tick_chained(kx):
    trade_schema = kx.schema.builder({
        'time': kx.TimespanAtom,
        'sym': kx.SymbolAtom,
        'px': kx.FloatAtom})
    tick = kx.tick.TICK(port=5030, tables={'trades': trade_schema})
    tick.start()
    tick_chained = kx.tick.TICK(port=5031, chained=True)
    tick_chained.start({'tickerplant': 'localhost:5030'})
    assert isinstance(tick_chained('trades'), kx.Table)
    tick.stop()
    tick_chained.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_tick_apis(kx):
    tick = kx.tick.TICK(port=5030)
    tick.register_api('custom_func', custom_api)
    with pytest.raises(kx.QError) as err:
        tick('custom_func', 5, 2)
    assert "name 'kx' is not defined" in str(err.value)
    tick.libraries({'kx': 'pykx'})
    assert tick('custom_func', 5, 2).py() == [2, 3, 4, 5, 6]
    tick.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_tick_timer(kx):
    tick = kx.tick.TICK(port=5030)
    assert tick('system"t"').py() == 100
    tick.set_timer(500)
    assert tick('system"t"').py() == 500
    tick.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_rtp_vanilla(kx):
    import time
    trade_schema = kx.schema.builder({
        'time': kx.TimespanAtom,
        'sym': kx.SymbolAtom,
        'px': kx.FloatAtom})
    quote_schema = kx.schema.builder({
        'time': kx.TimespanAtom,
        'sym': kx.SymbolAtom,
        'sz': kx.FloatAtom})
    tick = kx.tick.TICK(port=5030, tables={'trades': trade_schema, 'quotes': quote_schema})
    tick.start()

    rdb = kx.tick.RTP(port=5031)
    rdb.start({'tickerplant': 'localhost:5030'})
    rdb.set_tables({'px': trade_schema})
    assert isinstance(rdb('px'), kx.Table)
    assert isinstance(rdb('trades'), kx.Table)
    assert isinstance(rdb('quotes'), kx.Table)
    assert len(rdb('trades')) == 0
    assert len(rdb('quotes')) == 0
    # Publish data to tickerplant
    with kx.SyncQConnection(port=5030) as q:
        q.upd('trades', [kx.q.z.N, 'AAPL', 1.0])
    time.sleep(1)
    assert len(rdb('trades')) == 1
    assert len(rdb('quotes')) == 0
    rdb.stop()

    rdb = kx.tick.RTP(port=5031, subscriptions=['trades'])
    rdb.start({'tickerplant': 'localhost:5030'})
    assert isinstance(rdb('trades'), kx.Table)
    assert len(rdb('trades')) == 0
    with pytest.raises(kx.QError) as err:
        rdb('quotes')
    assert 'quotes' in str(err.value)
    tick.stop()
    rdb.stop()

    rdb = kx.tick.RTP(port=5031)
    with pytest.raises(kx.QError) as err:
        rdb.pre_processor(lambda x, y: y)
    assert 'Pre-processing of incoming' in str(err.value)
    with pytest.raises(kx.QError) as err:
        rdb.post_processor(lambda x, y: y)
    assert 'Post-processing of incoming' in str(err.value)
    rdb.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_rtp_custom_api(kx):
    rdb = kx.tick.RTP(port=5031, apis={'custom_api': custom_api})
    with pytest.raises(kx.QError) as err:
        rdb('custom_api', 5, 2)
    assert "name 'kx' is not defined" in str(err.value)
    rdb.libraries({'kx': 'pykx'})
    assert rdb('custom_api', 5, 2).py() == [2, 3, 4, 5, 6]
    rdb.stop()

    rdb = kx.tick.RTP(
        port=5031,
        apis={'custom_api': custom_api},
        libraries={'kx': 'pykx'})
    assert rdb('custom_api', 5, 2).py() == [2, 3, 4, 5, 6]
    rdb.stop()


def _pre_process(table, message):
    return message + 1


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_rtp_pre_proc(kx):
    rte = kx.tick.RTP(port=5031, libraries={'kx': 'pykx'}, vanilla=False)
    assert rte('.tick.RTPPreProc', 'test', 1) == 1
    rte.pre_processor(_pre_process)
    assert rte('.tick.RTPPreProc', 'test', 1) == 2
    rte.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_rtp_timer(kx):
    rtp = kx.tick.RTP(port=5031)
    assert rtp('system"t"').py() == 0
    rtp.set_timer(500)
    assert rtp('system"t"').py() == 500
    rtp.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_hdb_vanilla(kx):
    trade_schema = kx.schema.builder({
        'time': kx.TimespanAtom,
        'sym': kx.SymbolAtom,
        'px': kx.FloatAtom})
    hdb = kx.tick.HDB(port=5032)
    assert hdb('1b')
    with pytest.raises(kx.QError) as err:
        hdb('custom_api', 5, 2)
    assert "custom_api" in str(err.value)
    hdb.set_tables({'px': trade_schema})
    assert isinstance(hdb('px'), kx.Table)
    hdb.register_api('custom_api', custom_api)
    with pytest.raises(kx.QError) as err:
        hdb('custom_api', 5, 2)
    assert "name 'kx' is not defined" in str(err.value)
    hdb.libraries({'kx': 'pykx'})
    assert hdb('custom_api', 5, 2).py() == [2, 3, 4, 5, 6]
    hdb.stop()

    hdb = kx.tick.HDB(
        port=5032,
        apis={'custom_api': custom_api},
        libraries={'kx': 'pykx'})
    assert hdb('custom_api', 5, 2).py() == [2, 3, 4, 5, 6]
    hdb.stop()


def _gateway_func(x):
    rdb_data = gateway.call_port('rdb', b'{x+1}', x) # noqa: F821
    hdb_data = gateway.call_port('hdb', b'{x+2}', x) # noqa: F821
    return rdb_data + hdb_data


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_gateway_vanilla(kx):
    trade = kx.schema.builder({
        'time': kx.TimespanAtom, 'sym': kx.SymbolAtom,
        'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        'px': kx.FloatAtom})
    tick = kx.tick.TICK(port=5030, tables={'trade': trade})
    tick.start()
    hdb = kx.tick.HDB(port=5031)
    hdb.start(database='/tmp/db')
    rdb = kx.tick.RTP(port=5032)
    rdb.start({'tickerplant': 'localhost:5030'})
    gw = kx.tick.GATEWAY(
        port=5033,
        connections={'rdb': 'localhost:5032', 'hdb': 'localhost:5031'},
        apis={'custom_api': _gateway_func})
    gw.start()
    with kx.SyncQConnection(port=5033) as q:
        data = q('custom_api', 2)
    assert isinstance(data, kx.LongAtom)
    gw.stop()
    hdb.stop()
    rdb.stop()
    tick.stop()


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_basic_infra(kx):
    trade = kx.schema.builder({
        'time': kx.TimespanAtom,
        'sym': kx.SymbolAtom,
        'px': kx.FloatAtom})
    basic = kx.tick.BASIC(tables={'trade': trade}, log_directory='basic_logs')
    basic.start()
    assert 'basic_logs' in os.listdir()
    assert f'log{kx.DateAtom("today")}' in os.listdir('basic_logs')
    assert basic.hdb is None

    with kx.SyncQConnection(port=5011) as q:
        tab = q('trade')
    assert isinstance(tab, kx.Table)
    assert len(tab) == 0
    with kx.SyncQConnection(port=5010) as q:
        q.upd('trade', [kx.q.z.N, 'AAPL', 1.0])
    with kx.SyncQConnection(port=5011) as q:
        tab = q('trade')
    assert isinstance(tab, kx.Table)
    assert len(tab) == 1
    basic.stop()

    # Test restart will replay messages
    basic = kx.tick.BASIC(tables={'trade': trade}, log_directory='basic_logs')
    basic.start()
    with kx.SyncQConnection(port=5011) as q:
        tab = q('trade')
    assert isinstance(tab, kx.Table)
    assert len(tab) == 1
    basic.stop()

    # Test restart with hard_reset set will reset
    basic = kx.tick.BASIC(
        tables={'trade': trade},
        log_directory='basic_logs',
        hard_reset=True)
    basic.start()
    with kx.SyncQConnection(port=5011) as q:
        tab = q('trade')
    assert isinstance(tab, kx.Table)
    assert len(tab) == 0
    basic.stop()
    shutil.rmtree('basic_logs')


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_process_kill(kx):
    hdb = kx.tick.HDB(port=5012)
    hdb.start(database='db')
    with kx.SyncQConnection(port=5012) as q:
        ret = q('1b')
    assert ret
    kx.util.kill_q_process(port=5012)
    with pytest.raises(kx.QError) as err:
        kx.SyncQConnection(port=5012)
    assert any([x in str(err.value) for x in ['Connection refused', 'Connection reset']])


@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_init_args(kx):
    hdb0 = kx.tick.HDB(port=5012)
    assert 0 == hdb0('\\g')
    hdb0.stop()

    hdb1 = kx.tick.HDB(port=5012, init_args=['-g', '1'])
    assert 1 == hdb1('\\g')
    hdb1.stop()

    with pytest.raises(TypeError) as err:
        kx.tick.HDB(port=5012, init_args=10)
    assert 'must be a list' in str(err.value)

    with pytest.raises(TypeError) as err:
        kx.tick.HDB(port=5012, init_args=['test', 1])
    assert 'str type objects' in str(err.value)

import os
from pathlib import Path
from platform import system
import shutil
import warnings

# Do not import pykx here - use the `kx` fixture instead!
import pytest


@pytest.mark.isolate
def test_system_call(q):
    assert (q.system('echo "1"') == q('"1"')).all()


@pytest.mark.isolate
@pytest.mark.parametrize('num_threads', range(3))
def test_qargs_s_flag(num_threads):
    os.environ['QARGS'] = f'-s {num_threads}'
    import pykx as kx
    assert kx.q.system.max_num_threads == num_threads


@pytest.mark.isolate
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_qargs_s_flag_missing():
    os.environ['QARGS'] = '--licensed -s'
    with pytest.raises(Exception, match='ValueError: Missing argument for'):
        import pykx as kx # noqa: F401


@pytest.mark.isolate
@pytest.mark.parametrize('num_threads', (1.5, 1.0, 'hmmm'))
@pytest.mark.skipif(
    os.getenv('PYKX_THREADING') is not None,
    reason='Not supported with PYKX_THREADING'
)
def test_qargs_s_flag_invalid(num_threads):
    os.environ['QARGS'] = f'--licensed -s {num_threads}'
    with pytest.raises(Exception, match='ValueError: Invalid argument for'):
        import pykx as kx # noqa: F401


@pytest.mark.ipc
def test_num_threads(kx, q):
    with pytest.raises(AttributeError):
        # The max number of threads available to q is fixed on startup
        q.system.max_num_threads = 0

    assert isinstance(q.system.max_num_threads, int)
    assert isinstance(q.system.num_threads, int)

    if isinstance(q, kx.QConnection):
        assert q.system.max_num_threads == 0
        assert q.system.num_threads == 0
        return

    assert q.system.max_num_threads > 0

    orig_num_threads = q.system.num_threads
    assert q.system.num_threads > 0
    q.system.num_threads = 0
    assert q.system.num_threads == 0
    q.system.num_threads = 2
    assert q.system.num_threads == 2
    q.system.num_threads = orig_num_threads
    assert q.system.num_threads == orig_num_threads

    with pytest.raises(ValueError):
        q.system.num_threads = 1000000


@pytest.mark.isolate
def test_system_tables():
    import pykx as kx
    assert kx.q.system.tables().py() == []
    kx.q('qtab: ([] til 10; 2 + til 10)')
    kx.q('r: ([] til 10; 2 + til 10)')
    assert kx.q.system.tables().py() == ['qtab', 'r']
    kx.q('.foo.tab:([]10?1f;10?1f)')
    kx.q('foo.bar:([]10?1f)')
    assert kx.q.system.tables('.foo').py() == ['tab']
    assert kx.q.system.tables('foo').py() == ['bar']


@pytest.mark.isolate
def test_system_cd():
    import pykx as kx
    current_dir = kx.q.system.cd()
    kx.q.system.cd('/')
    assert all(kx.q.system.cd() == kx.q('enlist "/"'))
    kx.q.system.cd(current_dir)
    assert all(kx.q.system.cd() == current_dir)


@pytest.mark.isolate
def test_system_functions():
    import pykx as kx
    assert kx.q.system.functions() == kx.q('`symbol$()')
    kx.q('\\d .foo')
    kx.q('func: {x + 3}')
    kx.q('\\d .')
    kx.q('foo.bar: {x+1}')
    assert all(kx.q.system.functions('foo') == kx.q('enlist `bar'))
    assert all(kx.q.system.functions('.foo') == kx.q('enlist `func'))


@pytest.mark.isolate
def test_system_random_seed():
    import pykx as kx
    original_seed = kx.q.system.random_seed
    kx.q.system.random_seed = 55
    assert original_seed != kx.q.system.random_seed


@pytest.mark.isolate
def test_system_variables():
    import pykx as kx
    assert kx.q.system.variables() == kx.q('`$()')
    kx.q('a: 5')
    assert all(kx.q.system.variables() == kx.q('enlist `a'))
    print(kx.q.system.variables('.pykx'))
    assert all(kx.q.system.variables('.pykx') == kx.q('`debug`i`pykxDir`pykxExecutable`util'))
    kx.q('pykx.a:til 10;pykx.b:20')
    assert all(kx.q.system.variables('pykx') == kx.q('`a`b'))


@pytest.mark.isolate
def test_system_workspace():
    import pykx as kx
    ws = kx.q.system.workspace
    assert isinstance(ws, kx.Dictionary)
    ws = ws.py()
    ks = ['used', 'heap', 'peak', 'wmax', 'mmap', 'mphy', 'syms', 'symw']
    assert all([x in ws.keys() for x in ks])


@pytest.mark.isolate
def test_system_week_offset():
    import pykx as kx
    assert kx.q('2i') == kx.q.system.week_offset
    kx.q.system.week_offset = 5
    assert kx.q('5i') == kx.q.system.week_offset


@pytest.mark.isolate
def test_system_date_parsing():
    import pykx as kx
    assert kx.q.system.date_parsing == kx.q('0i')
    kx.q.system.date_parsing = 1
    assert kx.q.system.date_parsing == kx.q('1i')
    with pytest.raises(ValueError):
        kx.q.system.date_parsing = 2


@pytest.mark.isolate
def test_system_precision():
    import pykx as kx
    assert kx.q.system.precision == kx.q('7i')
    kx.q.system.precision = 2
    assert kx.q.system.precision == kx.q('2i')
    assert str(kx.q('3.14159')) == '3.1'


@pytest.mark.isolate
def test_system_utc():
    import pykx as kx
    assert kx.q.system.utc_offset == kx.q('0Ni')
    kx.q.system.utc_offset = 3
    assert kx.q.system.utc_offset == kx.q('3i')
    kx.q.system.utc_offset = 35
    assert kx.q.system.utc_offset == kx.q('35i')


@pytest.mark.isolate
def test_system_gc_mode():
    import pykx as kx
    assert kx.q.system.garbage_collection == kx.q('0i')
    kx.q.system.garbage_collection = 1
    assert kx.q.system.garbage_collection == kx.q('1i')
    with pytest.raises(ValueError):
        kx.q.system.garbage_collection = 2


@pytest.mark.isolate
def test_system_load():
    import pykx as kx
    try:
        with open('a.q', 'w') as f:
            f.write('avar: 5;\nafunc:{til x};')
        with open('b.q', 'w') as f:
            f.write('bvar: 15;\nbfunc:{9 + x};')
        kx.q.system.load('a.q')
        kx.q.system.load('b.q')
        assert kx.q('avar') == kx.q('5')
        assert kx.q('bvar') == kx.q('15')
        assert kx.q('afunc[5]~til 5')
        assert kx.q('bfunc[5]~14')
    finally:
        try:
            os.remove('a.q')
        except BaseException:
            pass
        try:
            os.remove('b.q')
        except BaseException:
            pass


@pytest.mark.isolate
def test_system_space_load(tmp_path):
    test_location = tmp_path/'test directory'
    os.makedirs(test_location, exist_ok=True)
    cache_dir = os.getcwd()
    file_location = test_location/'load_file.q'
    with open(file_location, 'w') as f:
        f.write('.pykx_test.system.variable:1b')
    import pykx as kx
    kx.q.system.load(file_location)
    assert kx.q('.pykx_test.system.variable')
    assert cache_dir == os.getcwd()

    kx.q('.pykx_test.system.variable:0b')
    if system() == 'Windows':
        file_location = test_location/'..\\test directory\\\\\\load_file.q'
    else:
        file_location = test_location/'../test directory///load_file.q'
    kx.q.system.load(file_location)
    assert kx.q('.pykx_test.system.variable')
    assert cache_dir == os.getcwd()

    test_splay = test_location/'splay/'
    kx.q('{x set ([]10?1f;10?1f)}', test_splay)

    def test_load_splay(test_splay):
        cd = os.getcwd()
        loaded = kx.q.system.load(test_splay)
        assert cd == os.getcwd()
        assert loaded.py() == 'splay'
        assert isinstance(kx.q['splay'], kx.Table)
        kx.q('delete splay from `.')
        assert cache_dir == os.getcwd()

    test_load_splay(test_splay) # Path
    test_load_splay(str(test_splay)) # String
    test_load_splay(kx.toq(test_splay)) # Symbol with leading :
    test_load_splay(kx.toq(str(test_splay))) # Symbol without leading :
    test_load_splay(kx.CharVector(str(test_splay))) # CharVector
    test_load_splay(str(test_splay) + '/') # String with trailing /
    # Symbol with leading :  with trailing /
    test_load_splay(kx.q('{`$string[x],"/"}', kx.toq(test_splay)))
    # Symbol without leading : with trailing /
    test_load_splay(kx.q('{`$string[x],"/"}', kx.toq(str(test_splay))))
    # CharVector with trailing /
    test_load_splay(kx.q('{x,"/"}', kx.CharVector(str(test_splay))))

    file_move_location = test_location/'move_file.q'
    with open(file_move_location, 'w') as f:
        f.write('.pykx_test.move.variable:1b;system"cd .."')
    kx.q.system.load(file_move_location)
    assert kx.q('.pykx_test.move.variable')
    assert cache_dir != os.getcwd()
    os.chdir(cache_dir)


@pytest.mark.isolate
def test_system_namespace():
    import pykx as kx
    assert kx.q.system.namespace() == kx.q('`.')
    kx.q.system.namespace('pykx')
    assert kx.q.system.namespace() == kx.q('`.pykx')
    kx.q.system.namespace('')
    assert kx.q.system.namespace() == kx.q('`.')
    kx.q.system.namespace('pykx')
    assert kx.q.system.namespace() == kx.q('`.pykx')
    kx.q.system.namespace('.')
    assert kx.q.system.namespace() == kx.q('`.')


@pytest.mark.isolate
def test_system_rename():
    import pykx as kx
    try:
        with open('z.q', 'w') as f:
            f.write('zvar: 23;')
        kx.q.system.rename('z.q', 'y.q')
        kx.q.system.load('y.q')
        assert kx.q('23~zvar')
    finally:
        try:
            os.remove('z.q')
        except BaseException:
            pass
        try:
            os.remove('y.q')
        except BaseException:
            pass


@pytest.mark.isolate
def test_system_console_size():
    import pykx as kx
    kx.q('tab: ([idx: til 100] x: 100?`foo`bar`baz`qux; y:100? 5000f; z:100?("hello"; "there";'
         ' "table"))')
    assert str(kx.q('tab')) != ('idx| x ..\n---| --..\n0  | ba..\n1  | qu..\n2  | ba..\n3  |'
                                ' ba..\n4  | fo..\n..')
    console = kx.q.system.console_size.py()
    kx.q.system.console_size = [10, 10]
    assert len(str(kx.q('tab'))) == len('idx| x ..\n---| --..\n0  | ba..\n1  | qu..\n2  | ba..\n3  '
                                        '| ba..\n4  | fo..\n..')
    kx.q.system.console_size = console


@pytest.mark.isolate
def test_system_display_size():
    import pykx as kx
    kx.q('tab: ([idx: til 100] x: 100?`foo`bar`baz`qux; y:100? 5000f; z:100?("hello"; "there";'
         ' "table"))')
    assert str(kx.q('tab')) != ('idx| x ..\n---| --..\n0  | ba..\n1  | qu..\n2  | ba..\n3  |'
                                ' ba..\n4  | fo..\n..')
    kx.q.system.display_size = [10, 10]
    assert len(str(kx.q('tab'))) == len('idx| x ..\n---| --..\n0  | ba..\n1  | qu..\n2  | ba..\n3  '
                                        '| ba..\n4  | fo..\n..')


@pytest.mark.isolate
def test_system_tables_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        assert q.system.tables().py() == []
        q('qtab: ([] til 10; 2 + til 10)')
        q('r: ([] til 10; 2 + til 10)')
        assert q.system.tables().py() == ['qtab', 'r']


@pytest.mark.isolate
def test_system_cd_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        current_dir = q.system.cd()
        q.system.cd('/')
        assert all(q.system.cd() == q('enlist "/"'))
        q.system.cd(current_dir)
        assert all(q.system.cd() == current_dir)


@pytest.mark.isolate
def test_system_functions_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        q('print: {til x}')
        assert all(q.system.functions() == q('enlist `print'))
        q('.foo.func: {x + 3}')
        q('foo.bar:{x+2}')
        assert all(q.system.functions('foo') == q('enlist `bar'))
        assert all(q.system.functions('.foo') == q('enlist `func'))


@pytest.mark.isolate
def test_system_random_seed_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        original_seed = q.system.random_seed
        q.system.random_seed = 55
        assert original_seed != q.system.random_seed


@pytest.mark.isolate
def test_system_variables_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        assert q.system.variables() == q('`$()')
        q('a: 5')
        assert all(q.system.variables() == q('enlist `a'))
        q('.pykx.i: 5')
        q('.pykx.pykxDir: til 10')
        q('pykx.a: til 10')
        q('pykx.b: til 10')
        assert all(q.system.variables('.pykx') == q('`i`pykxDir'))
        assert all(q.system.variables('pykx') == q('`a`b'))


@pytest.mark.isolate
def test_system_workspace_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        ws = q.system.workspace
        assert isinstance(ws, kx.Dictionary)
        ws = ws.py()
        ks = ['used', 'heap', 'peak', 'wmax', 'mmap', 'mphy', 'syms', 'symw']
        assert all([x in ws.keys() for x in ks])


@pytest.mark.isolate
def test_system_week_offset_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        assert q('2i') == q.system.week_offset
        q.system.week_offset = 5
        assert q('5i') == q.system.week_offset


@pytest.mark.isolate
def test_system_date_parsing_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        assert q.system.date_parsing == q('0i')
        q.system.date_parsing = 1
        assert q.system.date_parsing == q('1i')
        with pytest.raises(ValueError):
            q.system.date_parsing = 2


@pytest.mark.isolate
def test_system_precision_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        assert q.system.precision == q('7i')
        with pytest.raises(kx.QError):
            q.system.precision = 2


@pytest.mark.isolate
def test_system_utc_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        assert q.system.utc_offset == q('0Ni')
        q.system.utc_offset = 3
        assert q.system.utc_offset == q('3i')
        q.system.utc_offset = 35
        assert q.system.utc_offset == q('35i')


@pytest.mark.isolate
def test_system_gc_mode_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        assert q.system.garbage_collection == q('0i')
        q.system.garbage_collection = 1
        assert q.system.garbage_collection == q('1i')
        with pytest.raises(ValueError):
            q.system.garbage_collection = 2


@pytest.mark.isolate
def test_system_load_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        try:
            with open('a.q', 'w') as f:
                f.write('avar: 5;\nafunc:{til x};')
            with open('b.q', 'w') as f:
                f.write('bvar: 15;\nbfunc:{9 + x};')
            q.system.load('a.q')
            q.system.load('b.q')
            assert q('avar') == q('5')
            assert q('bvar') == q('15')
            assert q('afunc[5]~til 5')
            assert q('bfunc[5]~14')
        finally:
            try:
                os.remove('a.q')
            except BaseException:
                pass
            try:
                os.remove('b.q')
            except BaseException:
                pass


@pytest.mark.isolate
def test_system_namespace_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        with pytest.raises(kx.QError):
            q.system.namespace()
        with pytest.raises(kx.QError):
            q.system.namespace('pykx')


@pytest.mark.isolate
def test_system_rename_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        try:
            with open('z.q', 'w') as f:
                f.write('zvar: 23;')
            q.system.rename('z.q', 'y.q')
            q.system.load('y.q')
            assert q('23~zvar')
        finally:
            try:
                os.remove('z.q')
            except BaseException:
                pass
            try:
                os.remove('y.q')
            except BaseException:
                pass


@pytest.mark.isolate
def test_system_console_size_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        q('tab: ([idx: til 100] x: 100?`foo`bar`baz`qux; y:100? 5000f; z:100?("hello"; "there";'
          ' "table"))')
        assert str(q('tab')) != ('idx| x ..\n---| --..\n0  | ba..\n1  | qu..\n2  | ba..\n3  |'
                                 ' ba..\n4  | fo..\n..')
        with pytest.raises(kx.QError):
            q.system.console_size = [10, 10]


@pytest.mark.isolate
def test_system_display_size_ipc(q_port):
    import pykx as kx
    with kx.SyncQConnection(port=q_port) as q:
        q('tab: ([idx: til 100] x: 100?`foo`bar`baz`qux; y:100? 5000f; z:100?("hello"; "there";'
          ' "table"))')
        assert str(q('tab')) != ('idx| x ..\n---| --..\n0  | ba..\n1  | qu..\n2  | ba..\n3  |'
                                 ' ba..\n4  | fo..\n..')
        with pytest.raises(kx.QError):
            q.system.display_size = [10, 10]


@pytest.mark.isolate
def test_suppress_warning_false():
    os.environ['PYKX_SUPPRESS_WARNINGS'] = 'False'
    import pykx as kx
    path = Path('Test Folder')
    os.makedirs(path, exist_ok=True)
    with open(path/'test.q', 'w') as f:
        f.write('j:"junk"')
    with warnings.catch_warnings(record=True) as w:
        kx.q.system.load('Test Folder/test.q')
        msg_string = ""
        for i in w:
            message = str(i.message)
            msg_string = msg_string + message
        assert "space" in msg_string
    shutil.rmtree('Test Folder')


@pytest.mark.isolate
def test_suppress_warning_true():
    os.environ['PYKX_SUPPRESS_WARNINGS'] = 'True'
    import pykx as kx
    path = Path('Test Folder')
    os.makedirs(path, exist_ok=True)
    with open(path/'test.q', 'w') as f:
        f.write('j:"junk"')
    with warnings.catch_warnings(record=True) as w:
        kx.q.system.load('Test Folder/test.q')
        msg_string = ""
        for i in w:
            message = str(i.message)
            msg_string = msg_string + message
        assert "space" not in msg_string
    shutil.rmtree('Test Folder')

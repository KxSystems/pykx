import shutil

import pytest


def save_helper(location, code, instructions):
    from pykx import nbextension
    nbextension.q(instructions, code)
    f = open(location, "r")
    assert f.read() == code
    shutil.rmtree(location.split('/')[0])


@pytest.mark.unlicensed
def test_save_unexecuted(kx):
    location = 'testfolder/file.q'
    code = 't:1 2 3\nt * 19\njunk junk junk\n'
    instructions = f'--save {location} --execute False'
    save_helper(location, code, instructions)


def test_save_executed(kx):
    location = 'testfolder/file.q'
    code = 't:1 2 3\nt * 19\n'
    instructions = f'--save {location}'
    save_helper(location, code, instructions)


@pytest.mark.unlicensed
def test_save_unexecuted_unlicensed(kx):
    location = 'testfolder/file.q'
    code = 't:1 2 3\nt * 19\njunk junk junk\n'
    instructions = f'--save {location} --execute False'
    save_helper(location, code, instructions)


def test_save_locked(q, kx):
    from pykx import nbextension
    location = 'secretfolder/file.q_'
    code = 'secret:3 6 9\nsecret_func:{x % 3}\n'
    nbextension.q(f'--save {location} --execute False', code)
    q(('\\l') +" "+ location)
    res = q('secret_func secret')
    assert (res == q('t:3 6 9;t % 3')).all()
    assert (str(q('secret_func')) == ("locked"))
    shutil.rmtree(location.split('/')[0])
    code = 'secret:3 6 9\nsecret_func:{x % 3}\n'
    nbextension.q(f'--save {location}', code)
    q(('\\l') +" "+ location)
    res = q('secret_func secret')
    assert (res == q('t:3 6 9;t % 3')).all()
    assert (str(q('secret_func')) == ("locked"))
    shutil.rmtree(location.split('/')[0])


@pytest.mark.unlicensed
def test_save_ipc(kx, q_port):
    port = str(q_port)
    location = 'testfolder/file.q'
    code = 't:1 2 3\nt * 19\n'
    instructions = f'--port {port} --save {location}'
    save_helper(location, code, instructions)


@pytest.mark.unlicensed
def test_save_ipc_locked(q, q_port):
    from pykx import nbextension
    port = str(q_port)
    location = 'secretfolder/file.q_'
    code = 'secret:3 6 9\nsecret_func:{x % 3}\n'
    nbextension.q(f'--save {location} --execute False --port {port}', code)
    q(('\\l') +" "+ location)
    res = q('secret_func secret')
    assert (res == q('t:3 6 9;t % 3')).all()
    assert (str(q('secret_func')) == ("locked"))
    shutil.rmtree(location.split('/')[0])

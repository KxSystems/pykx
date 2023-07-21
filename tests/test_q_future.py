from platform import system

import pytest


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_internal_await(kx, q_port, event_loop):
    with kx.QConnection(port=q_port) as q:
        til_ten = q('til 10').py()
        til_twenty = q('til 20').py()

    async with kx.AsyncQConnection(port=q_port) as q:
        calls = [q('til 10'), q('til 20')]
        assert til_twenty == calls[1]._await().py()
        assert til_ten == calls[0]._await().py()
        assert til_ten == (await q('til 10').__async_await__()).py()

    async with kx.AsyncQConnection(port=q_port, event_loop=event_loop) as q:
        calls = [q('til 10'), q('til 20')]
        assert til_ten == (await calls[0]).py()
        assert til_twenty == (await calls[1]).py()


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_q_future_callbacks(kx, q_port):
    def _callback(x):
        pass

    if system() == 'Windows':
        async with kx.AsyncQConnection(port=q_port) as q:
            # TODO: This test passes but the lambda causes a random other test to fail on windows
            q_future = q('til 10')
            assert len(q_future._callbacks) == 0
            q_future.add_done_callback(_callback)
            assert len(q_future._callbacks) == 1
            assert 1 == q_future.remove_done_callback(_callback)
            assert len(q_future._callbacks) == 0
            assert (await q_future).py() == [x for x in range(10)]
    else:
        async with kx.AsyncQConnection(port=q_port) as q:
            q_future = q('til 10')
            assert len(q_future._callbacks) == 0
            q_future.add_done_callback(
                lambda x: x.set_result([int(x) + 1 for x in x.result().py()])
            )
            assert len(q_future._callbacks) == 1
            q_future.add_done_callback(_callback)
            assert len(q_future._callbacks) == 2
            assert 1 == q_future.remove_done_callback(_callback)
            assert len(q_future._callbacks) == 1
            assert (await q_future) == [x + 1 for x in range(10)]


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_q_future_errors(kx, q_port):
    def foo():
        pass

    async with kx.AsyncQConnection(port=q_port) as q:
        call = q('til 5')
        with pytest.raises(kx.NoResults):
            call.result()
        assert not call.cancelled()
        assert not call.done()
        call.cancel()
        assert call.cancelled()
        assert call.done()
        with pytest.raises(kx.FutureCancelled):
            call.result()
        with pytest.raises(kx.QError):
            await q('zzz')
        with pytest.raises(kx.PyKXException):
            call.get_loop()
        q_future = q('til 10')
        with pytest.raises(kx.NoResults):
            raise q_future.exception()
        q_future.cancel()
        with pytest.raises(kx.FutureCancelled):
            raise q_future.exception()
        with pytest.raises(kx.QError):
            q_future = q('zzz')
            try:
                await q_future
            except kx.QError:
                raise q_future.exception()

import asyncio
from platform import system
import sys

import pytest


@pytest.mark.asyncio
@pytest.mark.unlicensed
async def test_internal_await(kx, q_port, event_loop):
    with kx.QConnection(port=q_port) as q:
        til_ten = q('til 10').py()
        til_twenty = q('til 20').py()

    async with kx.AsyncQConnection(port=q_port) as q:
        calls = [q('til 10'), q('til 20')]
        assert til_twenty == (await calls[1]).py()
        assert til_ten == (await calls[0]).py()
        assert til_ten == (await q('til 10')).py()

    async with kx.AsyncQConnection(port=q_port, event_loop=event_loop) as q:
        calls = [q('til 10'), q('til 20')]
        assert til_ten == (await calls[0]).py()
        assert til_twenty == (await calls[1]).py()


@pytest.mark.asyncio
@pytest.mark.unlicensed
@pytest.mark.xfail(
    reason="Super flaky with all the different behaviours of futures between asyncio versions."
)
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
            q_future.add_done_callback(
                lambda x: print(x)
            )
            assert len(q_future._callbacks) == 1
            q_future.add_done_callback(_callback)
            assert len(q_future._callbacks) == 2
            assert 1 == q_future.remove_done_callback(_callback)
            assert len(q_future._callbacks) == 1


@pytest.mark.asyncio
@pytest.mark.unlicensed
@pytest.mark.xfail(
    reason="Super flaky with all the different behaviours of futures between asyncio versions."
)
async def test_q_future_errors(kx, q_port):
    def foo():
        pass

    async with kx.AsyncQConnection(port=q_port) as q:
        call = q('til 5')
        with pytest.raises(asyncio.exceptions.InvalidStateError):
            call.result()
        assert not call.cancelled()
        assert not call.done()
        call.cancel()
        await asyncio.sleep(2)
        if sys.version_info.minor > 10:
            assert call.cancelled() or call.cancelling()
        with pytest.raises(asyncio.exceptions.InvalidStateError):
            call.result()
        with pytest.raises(kx.QError):
            await q('zzz')
        q_future = q('til 10')
        with pytest.raises(asyncio.exceptions.InvalidStateError):
            raise q_future.exception()
        q_future.cancel()
        with pytest.raises(asyncio.exceptions.InvalidStateError):
            raise q_future.exception()
        with pytest.raises(kx.QError):
            q_future = q('zzz')
            try:
                await q_future
            except kx.QError:
                raise q_future.exception()

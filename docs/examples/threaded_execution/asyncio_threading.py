import os
import asyncio
os.environ['PYKX_THREADING'] = '1'
os.environ['PYKX_BETA_FEATURES'] = '1'

import pykx as kx


table = kx.q('table: ([] a: 10?10; b: 10?10)')


def assert_result(res):
    # assert message from q process has the correct schema to be appended to the table
    return type(res) is kx.LongVector and len(res) == 2


async def upsert_threaded(q, calls):
    counter = calls
    while True:
        result = await q.poll_recv_async()
        if assert_result(result):
            kx.q.upsert('table', result)
            result = None
            counter -= 1
        if counter <= 0:
            break


async def main():
    N = 20
    calls = 1000
    conns = [await kx.RawQConnection(port=5001, event_loop=asyncio.get_event_loop()) for _ in range(N)] # noqa
    main_q_con = kx.SyncQConnection(port=5001)
    print('===== Initial Table =====')
    print(kx.q('table'))
    print('===== Initial Table =====')
    # Set the variable py_server on the q process pointing towards this processes IPC connection
    # We use neg to ensure the messages are sent async so no reply is expected from this process
    [await conns[i](f'py_server{i}: neg .z.w') for i in range(N)]
    query = 'send_data: {'
    for i in range(N):
        query += f'py_server{i}[2?100];'
    query = query[:-1] + '}'

    await conns[0](query)

    tasks = [asyncio.create_task(upsert_threaded(conns[i], calls)) for i in range(N)]
    main_q_con(f'do[{calls}; send_data[]]', wait=False)
    [await t for t in tasks]
    print(kx.q('table'))


if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        kx.shutdown_thread()

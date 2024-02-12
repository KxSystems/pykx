import pykx as kx

import asyncio


table = kx.q('([] a: 10?10; b: 10?10)')


def assert_result(res):
    # assert message from q process has the correct schema to be appended to the table
    return type(res) is kx.LongVector and len(res) == 2


async def main_loop(q):
    global table
    while True:
        result = await q.poll_recv_async()
        if assert_result(result):
            print(f'Recieved new table row from q: {result}')
            table = kx.q.upsert(table, result)
            print(table)
            result = None


async def main():
    global table
    async with kx.RawQConnection(port=5001, event_loop=asyncio.get_event_loop()) as q:
        print('===== Initital Table =====')
        print(table)
        print('===== Initital Table =====')
        # Set the variable py_server on the q process pointing towards this processes IPC connection
        # We use neg to ensure the messages are sent async so no reply is expected from this process
        await q('py_server: neg .z.w')

        await main_loop(q)


if __name__ == '__main__':
    asyncio.run(main())

import pykx as kx

import asyncio


table = kx.q('([] a: 10?10; b: 10?10)')


def assert_result(res):
    # assert message from q process has the correct schema to be appended to the table
    return type(res) is kx.LongVector and len(res) == 2


async def main_loop(q):
    global table
    iters = 200 # only run a limited number of iterations for the example
    while True:
        await asyncio.sleep(0.5) # allows other async tasks to run along side
        result = q.poll_recv() # this will return None if no message is available to be read
        if assert_result(result):
            print(f'Recieved new table row from q: {result}')
            table = kx.q.upsert(table, result)
            print(table)
            result = None
        iters -= 1
        if iters < 0:
            break


async def main():
    global table
    async with kx.RawQConnection(port=5001) as q:
        print('===== Initial Table =====')
        print(table)
        print('===== Initial Table =====')
        # Set the variable py_server on the q process pointing towards this processes IPC connection
        # We use neg to ensure the messages are sent async so no reply is expected from this process
        await q('py_server: neg .z.w')

        await main_loop(q)


if __name__ == '__main__':
    asyncio.run(main())

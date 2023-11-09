import asyncio
import sys


import pykx as kx

port = 5000
if len(sys.argv)>1:
    port = int(sys.argv[1])


def qval_sync(query):
    res = kx.q.value(query)
    print("sync")
    print(f'{query}\n{res}\n')
    return res


def qval_async(query):
    res = kx.q.value(query)
    print("async")
    print(f'{query}\n{res}\n')


async def main():
    # It is possible to add user validation by overriding the .z.pw function
    # kx.q.z.pw = lambda username, password: password == 'password'
    kx.q.z.pg = qval_sync
    kx.q.z.ps = qval_async
    async with kx.RawQConnection(port=port, as_server=True, conn_gc_time=20.0) as q:
        while True:
            await q.poll_recv_async()


if __name__ == "__main__":
    asyncio.run(main())

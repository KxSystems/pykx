import asyncio

import pykx as kx


def qval(query):
    res = kx.q.value(query)
    print(f'{query}\n{res}\n')
    return res


async def main():
    # It is possible to add user validation by overriding the .z.pw function
    # kx.q.z.pw = lambda username, password: password == 'password'
    kx.q.z.pg = qval
    async with kx.RawQConnection(port=5000, as_server=True, conn_gc_time=20.0) as q:
        while True:
            q.poll_recv()


if __name__ == "__main__":
    asyncio.run(main())

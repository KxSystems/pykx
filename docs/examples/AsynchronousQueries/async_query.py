import asyncio
import time

import pykx as kx


class ConnectionManager:
    connections = {}

    async def open(self, port):
        self.connections[port] = await kx.AsyncQConnection(port=port)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for v in self.connections.values():
            await v.close()

    def __call__(self, port, query, *args):
        return self.connections[port](query, *args)


async def main():
    async with ConnectionManager() as cm:
        await cm.open(5050)
        await cm.open(5051)
        start = time.monotonic_ns()
        queries = [
            cm(5050, '{system"sleep 10"; til x + y}', 6, 7),
            cm(5051, '{system"sleep 5"; til 10}[]')
        ]
        queries = [await x for x in queries]
        end = time.monotonic_ns()
        [print(x) for x in queries]
        print(f'took {(end - start) / 1_000_000_000} seconds')


if __name__ == '__main__':
    asyncio.run(main())

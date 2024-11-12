import pykx as kx

import sys
import asyncio

trade = kx.schema.builder({
    'time': kx.TimespanAtom, 'sym': kx.SymbolAtom,
    'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    'px': kx.FloatAtom})

quote = kx.schema.builder({
    'time': kx.TimespanAtom, 'sym': kx.SymbolAtom,
    'exchange': kx.SymbolAtom, 'bid': kx.FloatAtom,
    'ask': kx.FloatAtom, 'bidsz': kx.LongAtom,
    'asksz': kx.LongAtom})


async def main_loop(q, trade, quote):
    while True:
        await asyncio.sleep(0.005)
        result = q.poll_recv()
        if result is None:
            continue
        table = result[1]
        if table == 'trade':
            trade.upsert(result[2], inplace=True)
        elif table == 'quote':
            quote.upsert(result[2], inplace=True)
        sys.stdout.write(f"Trade count: {len(trade)}\r")
        sys.stdout.flush()


async def main():
    global quote
    global trade
    async with kx.RawQConnection(port=5010) as q:
        await q('.u.sub', 'trade', '')
        await q('.u.sub', 'quote', '')

        await main_loop(q, trade, quote)


if __name__ == '__main__':
    asyncio.run(main())

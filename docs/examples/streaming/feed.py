import pykx as kx

import argparse
import time

parser=argparse.ArgumentParser()

parser.add_argument(
    "--datapoints",
    help="Number of datapoints published per message",
    default=1,
    type=int)

parser.add_argument(
    "--timer",
    help="timer delay between published messages in seconds",
    default=1,
    type=float)

variables = vars(parser.parse_args())
datapoints = variables['datapoints']
timer = variables['timer']

print('Starting Feedhandler ...')
print(f'Publishing {datapoints} datpoint(s) every {timer} second(s)')

init = False


def main():
    global init
    symlist = ['AAPL', 'JPM', 'GOOG', 'BRK', 'WPO', 'IBM']
    exlist = ['NYSE', 'LON', 'CHI', 'HK']
    while True:
        sz = 10*kx.random.random(datapoints, 100)
        px = 20+kx.random.random(datapoints, 100.0)
        ask = kx.random.random(datapoints, 100.0)
        asksz = 10*kx.random.random(datapoints, 100)
        bd = ask - kx.random.random(datapoints, ask)
        bdsz = asksz - kx.random.random(datapoints, asksz)
        trade = [kx.random.random(datapoints, symlist),
                 kx.random.random(datapoints, exlist),
                 sz,
                 px]
        quote = [kx.random.random(datapoints, symlist),
                 kx.random.random(datapoints, exlist),
                 bd,
                 ask,
                 bdsz,
                 asksz]
        # Setting of not init for wait is intended to raise initial error
        # if the first message is unsuccessful
        with kx.SyncQConnection(port=5010, wait=not init, no_ctx=True) as q:
            q('.u.upd', 'trade', trade)
            if 0 == kx.random.random(1, 3)[0]:
                q('.u.upd', 'quote', quote)
        if not init:
            print('First message(s) sent, data-feed publishing ...')
            init=True
        if time != 0:
            time.sleep(timer)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Data feed stopped')

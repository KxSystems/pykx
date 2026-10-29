import asyncio

import pandas as pd
import pykx as kx


if kx.licensed:
    print("Running example in presence of licensed q")
else:
    print("Running example in absence of licensed q")


def python_analytic(qdata):
    # In the licensed case apply the calculation on all
    # historical data, in unlicensed case run on any newly
    # available data.
    if kx.licensed:
        dataframe = qdata.pd()
    else:
        dataframe = pd.DataFrame.from_dict(qdata)
    grouped = dataframe.groupby('sym', as_index=False)
    avg_by_group = grouped[['sz', 'px']].mean()
    return avg_by_group


# If licensed, define the upd function to be invoked
if kx.licensed:
    kx.q('upd:{x insert y}')


async def main():
    # Open a connection to the tickerplant. A RawQConnection is used as it is
    # the only connection type which accepts the asynchronous messages that a
    # tickerplant publishes to its subscribers; the other connection types
    # discard them.
    async with kx.RawQConnection('localhost', 5140) as sub_connection:
        # Subscribe to updates from the trade table
        await sub_connection('.u.sub', 'trade', '')

        with kx.SyncQConnection('localhost', 5130) as pub_connection:
            while True:
                # Yield to the event loop so that any published messages can be
                # read from the socket into the connection's receive queue
                await asyncio.sleep(0.005)

                # Retrieve the next published message, if one is available.
                # Messages take the form ('upd'; 'trade'; data)
                message = sub_connection.poll_recv()
                if message is None:
                    continue

                if kx.licensed:
                    # Apply the update to a table held in embedded q, then run
                    # the analytic over the full history accumulated so far
                    kx.q('upd', message[1], message[2])
                    trade = kx.q('trade')
                else:
                    # We cannot invoke upd in the absence of a license, so the
                    # analytic runs on the batch of data just received. The
                    # message is converted to Python before indexing as K
                    # objects cannot be indexed into in unlicensed mode.
                    trade = message.py()[2]

                # Invoke the Python analytic
                python_return = python_analytic(trade)
                # Send the Python analytic return to the predefined
                # connection on localhost:5130
                pub_connection('{show x}', python_return)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Subscriber suspended')

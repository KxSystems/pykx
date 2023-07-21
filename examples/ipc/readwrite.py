import select

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

# Open a connection to the defined process
sub_connection = kx.QConnection('localhost', 5140)
# Subscribe to updates from the trade table
sub_connection('{.u.sub[x;y]}', 'trade', '')
# Retrieve the file descriptor to be monitored for messages
sub_connection_fd = int(sub_connection.fileno())

pub_connection = kx.QConnection('localhost', 5130)

is_readable = [sub_connection_fd]
is_writeable = []
is_error = []

while is_readable:
    # Set file descriptors expected to monitored for
    # incoming data by Python's select loop functionality
    readable, writable, error = select.select(is_readable,
                                              is_writeable,
                                              is_error)

    # This will be triggered whenever anything is available on the
    # reading socket
    for _sock in readable:
        # Trigger read from socket by calling with empty char on fd,
        # this is required as embedded q is not actively listening
        # to open sockets so needs to be 'told' to do so.
        if kx.licensed:
            sub_connection(b'')
            trade = kx.q('trade')
        else:
            # In the unlicensed case the return is a list containing
            # ['upd','trade', data], we cannot invoke the upd in
            # the absence of a license so store the returned data to
            # be modified with the python analytic by indexing in for
            # modification
            trade = sub_connection(b'').py()[2]

        # Do nothing if the sync call returns nothing
        if trade is None:
            pass
        else:
            # Invoke the Python analytic
            python_return = python_analytic(trade)
            # Send the Python analytic return to the predefined
            # connection on localhost:5130
            pub_connection('{show x}', python_return)

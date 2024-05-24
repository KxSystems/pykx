# Set environment variables needed to run Steamlit integration
import os
os.environ['PYKX_BETA_FEATURES'] = 'true'

# This is optional but suggested as without it's usage caching
# is not supported within streamlit
os.environ['PYKX_THREADING'] = 'true'

import streamlit as st
import pykx as kx
import matplotlib.pyplot as plt


def main():
    st.header('PyKX Demonstration')
    connection = st.connection('pykx',
                               type=kx.streamlit.PyKXConnection,
                               port=5050,
                               username='user',
                               password='password')
    if connection.is_healthy():
        tab = connection.query('select from tab where size<11')
    else:
        raise kx.QError('Connection object was not deemed to be healthy')
    fig, x = plt.subplots()
    x.scatter(tab['size'], tab['price'])

    st.write('Queried kdb+ remote table')
    st.write(tab)

    st.write('Generated plot')
    st.pyplot(fig)


if __name__ == "__main__":
    try:
        main()
    finally:
        kx.shutdown_thread()

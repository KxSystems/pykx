# Do not import pykx here - use the `kx` fixture instead!
import pytest
import streamlit as st


def test_streamlit(kx, q_port):
    conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
                         host='localhost', port=q_port)
    assert kx.q('~', conn.query('til 5'), [0, 1, 2, 3, 4])

    conn.query('tab:([]10?1f;10?1f)')
    sql_loaded = conn.query('@[{value x;1b};"s) ";{0b}]')
    if sql_loaded:
        assert kx.q('~', conn.query('tab'), conn.query('select * from tab', format='sql'))
    assert kx.q('~', conn.query('select from tab where x>0.5'), conn.query('tab', where='x>0.5', format='qsql')) # noqa: E501
    assert conn.is_healthy()

    with pytest.raises(kx.QError) as err:
        conn.query('tab', format='unsupported')
    assert 'Unsupported format provided for query' in str(err.value)

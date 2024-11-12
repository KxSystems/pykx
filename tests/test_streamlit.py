import sys

# Do not import pykx here - use the `kx` fixture instead!
import pytest

if not sys.version_info < (3, 8):
    import streamlit as st


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
def test_streamlit(kx, q_port):
    conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
                         host='localhost', port=q_port)
    assert kx.q('~', conn.query('til 5'), [0, 1, 2, 3, 4])

    conn.query('tab:([]10?1f;10?1f)')
    sql_loaded = conn.query('@[{system"l ",x;1b};"s.k_";{0b}]')
    if sql_loaded:
        assert kx.q('~', conn.query('tab'), conn.query('select * from tab', format='sql'))
    assert kx.q('~', conn.query('select from tab where x>0.5'), conn.query('tab', where='x>0.5', format='qsql')) # noqa: E501
    assert conn.is_healthy()

    with pytest.raises(kx.QError) as err:
        conn.query('tab', format='unsupported')
    assert 'Unsupported format provided for query' in str(err.value)

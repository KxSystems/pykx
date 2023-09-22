# Do not import pykx here - use the `kx` fixture instead!

import pytest


def test_register_py_toq(q, kx):
    with pytest.raises(TypeError) as err_info:
        kx.toq(complex(1, 2))
    assert str(err_info.value) == "Cannot convert <class 'complex'> '(1+2j)' to K object"

    def complex_toq(data):
        return kx.toq([data.real, data.imag])
    kx.register.py_toq(complex, complex_toq)
    assert all(q('1 2f') == kx.toq(complex(1, 2)))

    def complex_toq_upd(data):
        return q('{`real`imag!(x;y)}', data.real, data.imag)

    with pytest.raises(Exception) as err_info:
        kx.register.py_toq(complex, complex_toq_upd)
    assert str(err_info.value) == "Attempting to overwrite already defined type :<class 'complex'>"

    kx.register.py_toq(complex, complex_toq_upd, overwrite=True)
    assert all(q('`real`imag!1 2f') == q('{x}', complex(1, 2)))

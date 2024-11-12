# Do not import pykx here - use the `kx` fixture instead!

import pytest


def test_dir(kx):
    assert ['column_function', 'py_toq'] == dir(kx.register)


def test_register_py_toq(q, kx):
    with pytest.raises(TypeError) as err_info:
        kx.toq(complex(1, 2))
    assert str(err_info.value) == (
        "Cannot convert <class 'complex'> '(1+2j)' to K object."
        " See pykx.register to register custom conversions.")

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


def test_register_column_function(q, kx):
    tab = kx.Table(data={
        'sym': kx.random.random(100, ['a', 'b', 'c']),
        'true': kx.random.random(100, 100.0),
        'pred': kx.random.random(100, 100.0)
    })
    with pytest.raises(AttributeError) as err_info:
        getattr(kx.Column, 'minmax') # noqa: B009
    assert str(err_info.value) == "type object 'Column' has no attribute 'minmax'"

    def min_max_scaler(self):
        return self.call('{(x-minData)%max[x]-minData:min x}')
    kx.register.column_function('minmax', min_max_scaler)
    assert 1.0 == tab.exec(kx.Column('pred').minmax().max()).py()

    with pytest.raises(Exception) as err_info:
        kx.register.column_function('minmax', min_max_scaler)
    assert str(err_info.value) == "Attribute minmax already defined, please use 'overwrite' keyword"

    def min_max_scaler(self):
        return self.call('{2*(x-minData)%max[x]-minData:min x}')
    kx.register.column_function('minmax', min_max_scaler, overwrite=True)
    assert 2.0 == tab.exec(kx.Column('pred').minmax().max()).py()

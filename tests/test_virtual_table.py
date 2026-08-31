from pathlib import Path

import pytest


class TestVirtualTable:

    def make_tab(self, kx):
        pq = kx.module.use('kx.pq')
        pq.t = kx.module.use('kx.pq.t')
        tab = pq.pq(Path('tests/data/types.parquet'))
        return tab

    def test_select(self, kx):
        tab = self.make_tab(kx)
        assert kx.q('~', tab.select(kx.Column('col0') & kx.Column('col1')),
                    kx.q('flip `col0`col1!(110100100b;1 2 3 4 5 6 7 8 9)')).py()

    def test_exec(self, kx):
        tab = self.make_tab(kx)
        assert kx.q('~', tab.exec(kx.Column('col1')), kx.q('1 2 3 4 5 6 7 8 9')).py()

    errStr = "pykx.VirtualTable objects cannot be operated on directly, you must .select() or .exec() data from them first" # noqa: E501

    # Conversion methods
    def test_cast(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).cast(kx.FloatAtom)
        assert self.errStr in str(err.value)

    def test_py(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).py()
        assert self.errStr in str(err.value)

    def test_np(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).np()
        assert self.errStr in str(err.value)

    def test_pd(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).pd()
        assert self.errStr in str(err.value)

    def test_pa(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).pa()
        assert self.errStr in str(err.value)

    # Utility methods
    def test_any(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).any()
        assert self.errStr in str(err.value)

    def test_all(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).all()
        assert self.errStr in str(err.value)

    def test_copy(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).copy()
        assert self.errStr in str(err.value)

    def test_is_atom(self, kx):
        assert self.make_tab(kx).is_atom is False

    # Comparison operators
    def test_lt(self, kx):
        with pytest.raises(kx.QError) as err:
            assert self.make_tab(kx) < self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_le(self, kx):
        with pytest.raises(kx.QError) as err:
            assert self.make_tab(kx) <= self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_eq(self, kx):
        with pytest.raises(kx.QError) as err:
            assert self.make_tab(kx) == self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_ne(self, kx):
        with pytest.raises(kx.QError) as err:
            assert self.make_tab(kx) != self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_gt(self, kx):
        with pytest.raises(kx.QError) as err:
            assert self.make_tab(kx) > self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_ge(self, kx):
        with pytest.raises(kx.QError) as err:
            assert self.make_tab(kx) >= self.make_tab(kx)
        assert self.errStr in str(err.value)

    # Arithmetic operators
    def test_add(self, kx):
        with pytest.raises(kx.QError) as err:
            assert self.make_tab(kx) + self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_radd(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__radd__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_sub(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx) - self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_rsub(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__rsub__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_mul(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx) * self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_rmul(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__rmul__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_truediv(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx) / self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_rtruediv(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__rtruediv__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_floordiv(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx) // self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_rfloordiv(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__rfloordiv__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_mod(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx) % self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_rmod(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__rmod__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_divmod(self, kx):
        with pytest.raises(kx.QError) as err:
            divmod(self.make_tab(kx), self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_rdivmod(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__rdivmod__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_pow(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx) ** self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_rpow(self, kx):
        with pytest.raises(kx.QError) as err:
            self.make_tab(kx).__rpow__(self.make_tab(kx))
        assert self.errStr in str(err.value)

    # Unary operators
    def test_neg(self, kx):
        with pytest.raises(kx.QError) as err:
            -self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_pos(self, kx):
        with pytest.raises(kx.QError) as err:
            +self.make_tab(kx)
        assert self.errStr in str(err.value)

    def test_abs(self, kx):
        with pytest.raises(kx.QError) as err:
            abs(self.make_tab(kx))
        assert self.errStr in str(err.value)

    def test_bool(self, kx):
        with pytest.raises(kx.QError) as err:
            bool(self.make_tab(kx))
        assert self.errStr in str(err.value)

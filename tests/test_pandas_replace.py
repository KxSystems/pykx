# Do not import pykx here - use the `kx` fixture instead!


def test_unkeyed_replace(kx, q):
    tab = kx.q('([] a:2 2 3; b:4 2 6; c:7 2 9; d:(`a;`b;`c); e:(1;2;`a))')
    assert all((tab.replace(2, 10).pd() == tab.pd().replace(2, 10)))
    assert all((tab.replace(1000, 1).pd() == tab.pd().replace(1000, 1)))
    assert all((tab.replace('a', 100).pd() == tab.pd().replace('a', 100)))
    assert all((tab.replace(2, 'a').pd() == tab.pd().replace(2, 'a')))
    assert all((tab.replace(3, "test").pd() == tab.pd().replace(3, "test")))

    replaced_tab = kx.q('([] a:2 2 3; b:((`a,2);2;6); c:7 2 9; d:(`a;`b;`c); e:(1;2;`a))')
    assert all((tab.replace(4, ('a', 2)) == replaced_tab))


def test_keyed_replace(kx, q):
    ktab = kx.q('([a:2 2 3]b:4 2 6; c:7 2 9; d:(`a;`b;`c); e:(1;2;`a))')
    assert all((ktab.replace(2, 10).pd() == ktab.pd().replace(2, 10)))
    assert all((ktab.replace(1000, 1).pd() == ktab.pd().replace(1000, 1)))
    assert all((ktab.replace('a', 100).pd() == ktab.pd().replace('a', 100)))
    assert all((ktab.replace(2, 'a').pd() == ktab.pd().replace(2, 'a')))
    assert all((ktab.replace(3, "test").pd() == ktab.pd().replace(3, "test")))

    replaced_ktab = kx.q('([a:2 2 3]b:((`a,2);2;6); c:7 2 9; d:(`a;`b;`c); e:(1;2;`a))')
    assert all(ktab.replace(4, ('a', 2)).values() == replaced_ktab.values())

"""Tests for the modules."""

import pytest


class TestSearchPath:
    DEFAULT_PATH = "/home/user/.kx/mod"

    # ── Helpers ───────────────────────────────────────────────────────────────────

    def reset_sp(self, kx):
        """Reset kx.module.SearchPath to the default state before each test."""
        kx.module._SP([self.DEFAULT_PATH])

    # ── Retrieve ──────────────────────────────────────────────────────────────────

    def test_retrieve_returns_list(self, kx):
        self.reset_sp(kx)
        result = kx.module.SearchPath()
        assert isinstance(result, list)

    def test_retrieve_contains_default(self, kx):
        self.reset_sp(kx)
        assert self.DEFAULT_PATH in kx.module.SearchPath()

    # ── Add ───────────────────────────────────────────────────────────────────────

    def test_add_appends_to_end(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath("/opt/mod", add=True)
        sp = kx.module.SearchPath()
        assert sp[-1] == "/opt/mod"

    def test_add_prepends_to_front(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath("/opt/mod", add=True, prepend=True)
        sp = kx.module.SearchPath()
        assert sp[0] == "/opt/mod"

    def test_add_duplicate_raises(self, kx):
        self.reset_sp(kx)
        with pytest.raises(ValueError, match="already on SearchPath"):
            kx.module.SearchPath(self.DEFAULT_PATH, add=True)

    def test_add_duplicate_allowed_with_flag(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath(self.DEFAULT_PATH, add=True, allow_duplicates=True)
        assert kx.module.SearchPath().count(self.DEFAULT_PATH) == 2

    def test_add_duplicate_allowed_prepend(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath(self.DEFAULT_PATH, add=True, prepend=True, allow_duplicates=True)
        sp = kx.module.SearchPath()
        assert sp[0] == self.DEFAULT_PATH
        assert sp.count(self.DEFAULT_PATH) == 2

    # ── Remove ────────────────────────────────────────────────────────────────────

    def test_remove_entry(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath("/opt/mod", add=True)
        kx.module.SearchPath("/opt/mod", remove=True)
        assert "/opt/mod" not in kx.module.SearchPath()

    def test_remove_all_occurrences(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath(self.DEFAULT_PATH, add=True, allow_duplicates=True)
        assert kx.module.SearchPath().count(self.DEFAULT_PATH) == 2
        kx.module.SearchPath(self.DEFAULT_PATH, remove=True)
        assert self.DEFAULT_PATH not in kx.module.SearchPath()

    def test_remove_missing_silent(self, kx):
        self.reset_sp(kx)
        result = kx.module.SearchPath("/nonexistent", remove=True)
        assert result is None  # no error, silent return

    def test_remove_missing_strict_raises(self, kx):
        self.reset_sp(kx)
        with pytest.raises(ValueError, match="not found in SearchPath"):
            kx.module.SearchPath("/nonexistent", remove=True, strict=True)

    def test_remove_preserves_other_entries(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath("/opt/mod", add=True)
        kx.module.SearchPath("/other/mod", add=True)
        kx.module.SearchPath("/opt/mod", remove=True)
        sp = kx.module.SearchPath()
        assert "/opt/mod" not in sp
        assert "/other/mod" in sp
        assert self.DEFAULT_PATH in sp

    # ── No keyword raises ─────────────────────────────────────────────────────────

    def test_no_keyword_raises(self, kx):
        self.reset_sp(kx)
        with pytest.raises(ValueError, match="Specify add=True or remove=True"):
            kx.module.SearchPath("/opt/mod")

    # ── Order integrity ───────────────────────────────────────────────────────────

    def test_order_append(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath("/a", add=True)
        kx.module.SearchPath("/b", add=True)
        kx.module.SearchPath("/c", add=True)
        assert kx.module.SearchPath() == [self.DEFAULT_PATH, "/a", "/b", "/c"]

    def test_order_prepend(self, kx):
        self.reset_sp(kx)
        kx.module.SearchPath("/a", add=True, prepend=True)
        kx.module.SearchPath("/b", add=True, prepend=True)
        assert kx.module.SearchPath() == ["/b", "/a", self.DEFAULT_PATH]


# TODO: Once Kdb-X is fully installed for testing in CI re-enable these tests.
"""
@pytest.mark.isolate
def test_q_module():
    import pykx as kx
    pq = kx.use('kx.pq')
    assert isinstance(pq, kx.module.QModule)
    pq.t = kx.use('kx.pq.t')
    assert isinstance(pq, kx.module.QModule)
    assert isinstance(pq.t, kx.module.QModule)
    assert isinstance(pq.pq, kx.wrappers.Lambda)
    assert isinstance(pq.t.mkT, kx.wrappers.Lambda)

    # assert module is singleton
    a = kx.use('kx.pq')
    assert isinstance(a, kx.module.QModule)
    assert isinstance(a.t, kx.module.QModule)


@pytest.mark.isolate
def test_q_module_auto():
    import pykx as kx
    pq = kx.use('kx.pq')
    assert isinstance(pq, kx.module.QModule)
    kx.use('kx.pq.t')
    assert isinstance(pq, kx.module.QModule)
    assert isinstance(pq.t, kx.module.QModule)
    assert isinstance(pq.pq, kx.wrappers.Lambda)
    assert isinstance(pq.t.mkT, kx.wrappers.Lambda)

    # assert module is singleton
    a = kx.use('kx.pq')
    assert isinstance(a, kx.module.QModule)
    assert isinstance(a.t, kx.module.QModule)
"""

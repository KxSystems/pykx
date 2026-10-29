"""Tests for the modules."""

import pytest


@pytest.fixture
def submodule_tree(tmp_path):
    """Write a q module which re-exports a name that is also a loadable submodule.

    Produces `kx.parent` (exporting `f` and `sub`) and `kx.parent.sub`, so that loading the
    parent first and the submodule second exercises `QModule.__setattr__`.
    """
    mod = tmp_path / 'kx' / 'parent'
    mod.mkdir(parents=True)
    (mod / 'init.q').write_text('f:{x+1}\nsub:([g:{x*2}])\nexport:([f;sub])\n')
    (mod / 'sub.q').write_text('g:{x*2}\nh:{x-1}\nexport:([g;h])\n')
    return str(tmp_path)


@pytest.fixture
def ordered_tree(tmp_path):
    """A module exporting names in a deliberately non-alphabetical order."""
    mod = tmp_path / 'kx'
    mod.mkdir(parents=True)
    (mod / 'ordered.q').write_text(
        'zulu:{x+1}\nalpha:{x+2}\nmike:{x+3}\nbravo:{x+4}\n'
        'export:([zulu;alpha;mike;bravo])\n')
    return str(tmp_path)


class TestExportOrder:
    """Iteration order must follow the module's export order, not set ordering.

    `_qkeys` is a set, and Python randomizes string hashing per process, so deriving the
    order from it made `a, b = kx.use(...)` bind different objects on different runs.
    """

    EXPECTED = ['zulu', 'alpha', 'mike', 'bravo']

    def test_unpack_follows_export_order(self, kx, ordered_tree):
        kx.module.SearchPath(ordered_tree, add=True, prepend=True)
        zulu, alpha, mike, bravo = kx.use('kx.ordered')
        # each lambda adds a distinct constant, so the values identify the binding
        assert [zulu(0).py(), alpha(0).py(), mike(0).py(), bravo(0).py()] == [1, 2, 3, 4]

    def test_iteration_follows_export_order(self, kx, ordered_tree):
        kx.module.SearchPath(ordered_tree, add=True, prepend=True)
        mod = kx.use('kx.ordered')
        assert [f(0).py() for f in mod] == [1, 2, 3, 4]

    def test_names_match_export_order(self, kx, ordered_tree):
        kx.module.SearchPath(ordered_tree, add=True, prepend=True)
        mod = kx.use('kx.ordered')
        assert object.__getattribute__(mod, '_names') == self.EXPECTED

    def test_repr_lists_exports_in_order(self, kx, ordered_tree):
        kx.module.SearchPath(ordered_tree, add=True, prepend=True)
        rows = [ln.split('|')[0].strip() for ln in repr(kx.use('kx.ordered')).split('\n')
                if '|' in ln]
        assert rows == self.EXPECTED


@pytest.fixture
def shapes_tree(tmp_path):
    """A module exporting each shape a member can take, including two that are type 99h."""
    mod = tmp_path / 'kx'
    mod.mkdir(parents=True)
    (mod / 'shapes.q').write_text(
        'cfg:`host`port!("localhost";5000);\n'      # plain data dictionary
        'kt:([sym:`a`b]px:1 2f);\n'                 # keyed table, also 99h
        't:([]a:1 2;b:3 4);\n'                      # unkeyed table
        '.sub.f:{x+1};\n.sub.g:{x*2};\nnested:.sub;\n'   # q namespace
        'f:{x+1};\n'
        'export:([cfg;kt;t;nested;f])\n')
    return str(tmp_path)


class TestMemberShapes:
    """Only q namespaces are presented as modules.

    A module may export ordinary data as a dictionary, and a keyed table is also type 99h
    and an instance of `Dictionary`. Wrapping either as a `QModule` stripped the behavior
    the caller needs: indexing, `len`, `.py()` and `.pd()`.
    """

    def load(self, kx, tree):
        kx.module.SearchPath(tree, add=True, prepend=True)
        return kx.use('kx.shapes')

    def test_namespace_is_a_module(self, kx, shapes_tree):
        s = self.load(kx, shapes_tree)
        assert isinstance(s.nested, kx.module.QModule)
        assert s.nested.f(1).py() == 2

    def test_plain_dictionary_stays_a_dictionary(self, kx, shapes_tree):
        s = self.load(kx, shapes_tree)
        assert isinstance(s.cfg, kx.Dictionary)
        assert not isinstance(s.cfg, kx.module.QModule)
        assert s.cfg['port'].py() == 5000
        assert s.cfg.py() == {'host': b'localhost', 'port': 5000}

    def test_keyed_table_stays_a_keyed_table(self, kx, shapes_tree):
        s = self.load(kx, shapes_tree)
        assert isinstance(s.kt, kx.KeyedTable)
        assert not isinstance(s.kt, kx.module.QModule)
        assert len(s.kt) == 2

    def test_table_and_function_unaffected(self, kx, shapes_tree):
        s = self.load(kx, shapes_tree)
        assert isinstance(s.t, kx.Table)
        assert len(s.t) == 2
        assert s.f(1).py() == 2

    def test_namespace_test_discriminates(self, kx):
        """The namespace marker must separate all three type-99h shapes."""
        from pykx.module import _is_q_namespace
        assert _is_q_namespace(kx.q('.tst.a:1; .tst'))
        assert not _is_q_namespace(kx.q('`host`port!("localhost";5000)'))
        assert not _is_q_namespace(kx.q('([sym:`a`b]px:1 2f)'))
        assert not _is_q_namespace(kx.q('meta ([]a:1 2)'))
        assert not _is_q_namespace(kx.q('()!()'))


@pytest.fixture
def clash_tree(tmp_path):
    """A module exporting Python keywords, and a dictionary using the `/(::) placeholder.

    That placeholder keeps a dictionary's values a general list, stopping q promoting them
    to a typed vector. It makes the dictionary indistinguishable from a q namespace.
    """
    mod = tmp_path / 'kx'
    mod.mkdir(parents=True)
    (mod / 'clash.q').write_text(
        'opts:``headers!((::);`Accept`X!("a";"b"));\n'
        'import:{x+1};\nclass:{x+3};\nlambda:{x+4};\nNone:{x+5};\n'
        'both:{x+9};\nboth_:{x+10};\n'
        'export:([opts;import;class;lambda;None;both;both_])\n')
    return str(tmp_path)


@pytest.fixture
def nested_clash_tree(tmp_path):
    """A module whose keyword exports sit inside a nested q namespace."""
    mod = tmp_path / 'kx'
    mod.mkdir(parents=True)
    (mod / 'nested.q').write_text(
        '.s.del:{x+1};\n.s.import:{x+2};\n.s.ok:{x+3};\n'
        'sub:.s;\n'
        'export:([sub])\n')
    return str(tmp_path)


@pytest.fixture
def proxy_tree(tmp_path):
    mod = tmp_path / 'kx' / 'proxymod'
    mod.mkdir(parents=True)
    (mod / 'init.q').write_text(
        'OFFSET:``BEGINNING`END!((::);-2;-1);\nf:{x+1};\nexport:([OFFSET;f])\n')
    return str(tmp_path)


class TestUnwrap:
    """`unwrap` recovers the dictionary behind a QModule."""

    def test_recovers_placeholder_dictionary(self, kx, clash_tree):
        kx.module.SearchPath(clash_tree, add=True, prepend=True)
        c = kx.use('kx.clash')
        # The `/(::) entry makes this look like a namespace, so it is presented as a module.
        assert isinstance(c.opts, kx.module.QModule)
        opts = kx.module.unwrap(c.opts)
        assert isinstance(opts, kx.Dictionary)
        assert len(opts) == 2
        assert opts['headers']['Accept'].py() == b'a'

    def test_roundtrips_with_wrap(self, kx):
        d = kx.q('`f`g!({x+1};{x*2})')
        assert kx.module.unwrap(kx.module.wrap(d)).py() == d.py()

    def test_returns_export_dict_of_a_loaded_module(self, kx, clash_tree):
        kx.module.SearchPath(clash_tree, add=True, prepend=True)
        exports = kx.module.unwrap(kx.use('kx.clash'))
        assert isinstance(exports, kx.Dictionary)
        assert 'import' in [k.py() for k in exports.keys()]

    @pytest.mark.parametrize('bad', [42, 'a string', None])
    def test_rejects_non_qmodule(self, kx, bad):
        with pytest.raises(TypeError, match='must be a QModule'):
            kx.module.unwrap(bad)

    def test_rejects_plain_dictionary(self, kx):
        with pytest.raises(TypeError, match='must be a QModule'):
            kx.module.unwrap(kx.q('`a`b!1 2'))

    def test_recovers_dictionary_from_namespace_proxy(self, kx, proxy_tree):
        kx.module.SearchPath(proxy_tree, add=True, prepend=True)
        m = kx.use('kx.proxymod')
        assert isinstance(m.OFFSET, kx.module.QNamespaceProxy)
        offsets = kx.module.unwrap(m.OFFSET)
        assert isinstance(offsets, kx.Dictionary)
        assert offsets['BEGINNING'].py() == -2
        assert offsets['END'].py() == -1

    def test_proxy_unwraps_to_the_same_dictionary_as_its_parent(self, kx, proxy_tree):
        kx.module.SearchPath(proxy_tree, add=True, prepend=True)
        m = kx.use('kx.proxymod')
        assert kx.module.unwrap(m.OFFSET).py() == kx.module.unwrap(m)['OFFSET'].py()


class TestKeywordEscape:
    """A single trailing underscore reaches an export Python cannot name directly."""

    @pytest.mark.parametrize('name,expected', [
        ('import_', 2), ('class_', 4), ('lambda_', 5), ('None_', 6),
    ])
    def test_trailing_underscore_reaches_keyword_export(self, kx, clash_tree, name, expected):
        kx.module.SearchPath(clash_tree, add=True, prepend=True)
        c = kx.use('kx.clash')
        assert getattr(c, name)(1).py() == expected

    def test_exact_name_wins_over_stripped(self, kx, clash_tree):
        """A module exporting both `both` and `both_` must not have them confused."""
        kx.module.SearchPath(clash_tree, add=True, prepend=True)
        c = kx.use('kx.clash')
        assert c.both(1).py() == 10       # {x+9}
        assert c.both_(1).py() == 11      # {x+10}, its own export

    def test_unknown_name_still_raises(self, kx, clash_tree):
        kx.module.SearchPath(clash_tree, add=True, prepend=True)
        c = kx.use('kx.clash')
        with pytest.raises(AttributeError):
            c.nosuchthing_

    def test_getattr_form_still_works(self, kx, clash_tree):
        kx.module.SearchPath(clash_tree, add=True, prepend=True)
        c = kx.use('kx.clash')
        assert getattr(c, 'import')(1).py() == 2

    def test_key_needs_no_escape(self, kx, clash_tree):
        kx.module.SearchPath(clash_tree, add=True, prepend=True)
        assert kx.use('kx.clash')['import'](1).py() == 2

    def test_escape_works_in_a_nested_namespace(self, kx, nested_clash_tree):
        """The escape must work at depth too, where a QNamespaceProxy is returned.

        `kx.ai.hnsw` exports `del`, and it is reached through a nested namespace.
        """
        kx.module.SearchPath(nested_clash_tree, add=True, prepend=True)
        n = kx.use('kx.nested')
        assert n.sub.del_(1).py() == 2
        assert n.sub.import_(1).py() == 3
        assert n.sub.ok(1).py() == 4          # non-keyword name unaffected
        with pytest.raises(AttributeError):
            n.sub.nosuch_

    def test_dir_returns_strings_at_depth(self, kx, nested_clash_tree):
        """`dir` must yield str so `keyword.iskeyword` can be used to find clashes."""
        import keyword
        kx.module.SearchPath(nested_clash_tree, add=True, prepend=True)
        names = dir(kx.use('kx.nested').sub)
        assert all(isinstance(x, str) for x in names)
        assert sorted(n for n in names if keyword.iskeyword(n)) == ['del', 'import']


class TestWrap:
    """`kx.module.wrap` presents a q dictionary of names with attribute access."""

    def qdict(self, kx):
        return kx.q('`f`g`n!({x+1};{x*2};42)')

    def test_returns_qmodule(self, kx):
        mod = kx.module.wrap(self.qdict(kx))
        assert isinstance(mod, kx.module.QModule)

    def test_dir_lists_string_keys(self, kx):
        mod = kx.module.wrap(self.qdict(kx))
        assert sorted(dir(mod)) == ['f', 'g', 'n']
        assert all(isinstance(name, str) for name in dir(mod))

    def test_members_accessible_and_callable(self, kx):
        mod = kx.module.wrap(self.qdict(kx))
        assert mod.f(1).py() == 2
        assert mod.g(21).py() == 42
        assert mod.n.py() == 42

    def test_assignment_refused(self, kx):
        mod = kx.module.wrap(self.qdict(kx))
        with pytest.raises(TypeError, match='has no q namespace to write to'):
            mod.f = kx.q('{x-1}')

    def test_assignment_creates_no_q_global(self, kx):
        mod = kx.module.wrap(self.qdict(kx))
        with pytest.raises(TypeError):
            mod.f = 42
        assert not kx.q('@[{1b;value x};`.f;{0b}]').py()

    def test_loaded_modules_still_allow_assignment(self, kx, submodule_tree):
        """The refusal must apply only to wrapped dictionaries."""
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        parent = kx.use('kx.parent')
        parent.f = kx.q('{x+100}')          # must not raise
        assert parent.f(1).py() == 101

    @pytest.mark.parametrize('bad', [42, 'a string', b'bytes', None])
    def test_rejects_non_dictionary(self, kx, bad):
        with pytest.raises(TypeError, match='must be a q dictionary'):
            kx.module.wrap(bad)

    def test_rejects_q_non_dictionary(self, kx):
        with pytest.raises(TypeError, match='must be a q dictionary'):
            kx.module.wrap(kx.q('til 3'))

    def test_exported_from_module_namespace(self, kx):
        assert 'wrap' in kx.module.__all__
        assert 'wrap' in dir(kx.module)


class TestSubmoduleRegistration:
    """A submodule must register on its parent rather than being written back to q."""

    def test_parent_then_submodule(self, kx, submodule_tree):
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        parent = kx.use('kx.parent')
        # Previously raised TypeError: a QModule has no q representation, so re-exported
        # submodule names took the `q('set';...)` branch of __setattr__.
        sub = kx.use('kx.parent.sub')
        assert isinstance(sub, kx.module.QModule)
        assert isinstance(parent.sub, kx.module.QModule)
        assert parent.sub is sub

    def test_submodule_members_callable(self, kx, submodule_tree):
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        parent = kx.use('kx.parent')
        sub = kx.use('kx.parent.sub')
        assert sub.g(21).py() == 42
        assert sub.h(21).py() == 20
        assert parent.sub.g(21).py() == 42
        assert parent.f(1).py() == 2

    def test_submodule_load_order_agnostic(self, kx, submodule_tree):
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        sub = kx.use('kx.parent.sub')
        parent = kx.use('kx.parent')
        assert kx.use('kx.parent.sub') is sub
        assert isinstance(parent.sub, kx.module.QModule)

    def test_submodule_cached(self, kx, submodule_tree):
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        kx.use('kx.parent')
        assert kx.use('kx.parent.sub') is kx.use('kx.parent.sub')

    def test_reexported_submodule_listed_once(self, kx, submodule_tree):
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        parent = kx.use('kx.parent')
        kx.use('kx.parent.sub')
        assert sorted(dir(parent)) == ['f', 'sub']
        assert len(list(iter(parent))) == 2
        # The registered submodule wins over the plain dict of the same name.
        assert repr(parent).count('sub') == 1
        assert 'sub| pykx.QModule' in repr(parent)

    def test_setting_non_module_still_reaches_q(self, kx, submodule_tree):
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        parent = kx.use('kx.parent')
        parent.f = kx.q('{x+100}')
        assert parent.f(1).py() == 101


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


@pytest.fixture
def failing_tree(tmp_path):
    """A module whose body signals, so loading it fails with a distinctive q error."""
    mod = tmp_path / 'kx'
    mod.mkdir(parents=True)
    (mod / 'boom.q').write_text("'`module_body_failed\n")
    return str(tmp_path)


class TestLoadFailure:
    """A failed `use` reports the q error, not just the module name."""

    def test_message_carries_the_q_error(self, kx, failing_tree):
        kx.module.SearchPath(failing_tree, add=True, prepend=True)
        with pytest.raises(ImportError) as err:
            kx.use('kx.boom')
        assert 'Failed to load kx.boom' in str(err.value)
        assert 'module_body_failed' in str(err.value)

    def test_missing_module_names_the_q_error(self, kx):
        with pytest.raises(ImportError) as err:
            kx.use('kx.nosuchmodule')
        assert 'Failed to load kx.nosuchmodule' in str(err.value)
        assert 'notfound' in str(err.value)

    def test_qerror_is_chained_as_the_cause(self, kx, failing_tree):
        kx.module.SearchPath(failing_tree, add=True, prepend=True)
        with pytest.raises(ImportError) as err:
            kx.use('kx.boom')
        assert isinstance(err.value.__cause__, kx.exceptions.QError)
        assert 'module_body_failed' in str(err.value.__cause__)


@pytest.fixture
def shadow_tree(tmp_path):
    """A module whose `export` publishes a wrapper under the name of a private variable.

    Written twice, as a single file and as a directory with an `init.q`, because the two
    used to resolve differently.
    """
    src = ('f:{"private ",x}\n'
           'counter:0\n'
           'bump:{.z.m.counter+:1}\n'
           'export:([f:{"exported ",x}; counter; bump])\n')
    (tmp_path / 'kx').mkdir(parents=True)
    (tmp_path / 'kx' / 'shadowfile.q').write_text(src)
    (tmp_path / 'kx' / 'shadowdir').mkdir()
    (tmp_path / 'kx' / 'shadowdir' / 'init.q').write_text(src)
    return str(tmp_path)


@pytest.fixture
def collision_tree(tmp_path):
    """Two modules whose leaf name is the same, under different parents."""
    for parent, value in (('one', 1), ('two', 2)):
        d = tmp_path / parent
        d.mkdir(parents=True)
        (d / 'leaf.q').write_text(f'v:{value}\nexport:([v])\n')
    return str(tmp_path)


@pytest.fixture
def fileno_tree(tmp_path):
    """A module exporting a name `toq` duck-types on."""
    (tmp_path / 'kx').mkdir(parents=True)
    (tmp_path / 'kx' / 'fileno.q').write_text('fileno:{42}\nf:{x+1}\nexport:([fileno;f])\n')
    return str(tmp_path)


PACKAGINGS = ['kx.shadowfile', 'kx.shadowdir']


class TestExportResolution:
    """An exported name resolves to its `export` entry, the way it does in q.

    Attribute access read the module's private namespace first and fell back to `export`
    only when the name was absent there, so a module exporting a wrapper under the name of
    one of its own private variables handed Python the private object while q handed back
    the exported one.
    """

    def load(self, kx, tree, name):
        kx.module.SearchPath(tree, add=True, prepend=True)
        return kx.use(name)

    @pytest.mark.parametrize('name', PACKAGINGS)
    def test_attribute_is_the_export_and_matches_q(self, kx, shadow_tree, name):
        mod = self.load(kx, shadow_tree, name)
        assert mod.f(b'x').py() == b'exported x' == kx.q(f'(use`{name})`f')(b'x').py()

    def test_key_and_iteration_agree_with_the_attribute(self, kx, shadow_tree):
        """Packaging is irrelevant here, so one is enough."""
        mod = self.load(kx, shadow_tree, 'kx.shadowdir')
        assert mod['f'](b'x').py() == b'exported x'
        assert next(iter(mod))(b'x').py() == b'exported x'      # `f` is exported first

    @pytest.mark.parametrize('name', PACKAGINGS)
    def test_exported_value_agrees_with_q(self, kx, shadow_tree, name):
        """An exported value is the snapshot `export` took, which is what q reports."""
        mod = self.load(kx, shadow_tree, name)
        mod.bump()
        mod.bump()
        assert mod.counter.py() == kx.q(f'(use`{name})`counter').py()


class TestNamespaceResolution:
    """The q namespace behind a module is found whatever the module is packaged as.

    `_resolve_qns` compared the recorded path against the module name and `name + '.k'`
    only, so a module living in one `.q`, `.q_` or `.so` file resolved to no namespace:
    assignment then wrote a stray global into the root namespace instead.
    """

    def load(self, kx, tree, name):
        kx.module.SearchPath(tree, add=True, prepend=True)
        return kx.use(name)

    def qns(self, mod):
        return object.__getattribute__(mod, '_qns')

    @pytest.mark.parametrize('name', PACKAGINGS)
    def test_namespace_is_resolved(self, kx, shadow_tree, name):
        mod = self.load(kx, shadow_tree, name)
        assert self.qns(mod).endswith(name.split('.')[-1])

    @pytest.mark.parametrize('name', PACKAGINGS)
    def test_assignment_reaches_the_module(self, kx, shadow_tree, name):
        mod = self.load(kx, shadow_tree, name)
        mod.counter = kx.q('99')
        assert kx.q('value', kx.SymbolAtom(self.qns(mod) + '.counter')).py() == 99
        assert mod.counter.py() == 99                                  # read agrees
        assert 'counter' not in [v.py() for v in kx.q('system"v ."')]   # no stray global

    def test_leaf_name_does_not_pick_the_wrong_namespace(self, kx, collision_tree):
        """Whole path tails are compared, so two modules named `leaf` stay apart."""
        kx.module.SearchPath(collision_tree, add=True, prepend=True)
        one, two = kx.use('one.leaf'), kx.use('two.leaf')
        assert self.qns(one) != self.qns(two)
        one.v = kx.q('11')
        assert kx.q('value', kx.SymbolAtom(self.qns(one) + '.v')).py() == 11
        assert kx.q('value', kx.SymbolAtom(self.qns(two) + '.v')).py() == 2

    def test_assignment_refused_without_a_namespace(self, kx):
        """A namespace-shaped member of a wrapped dictionary has nothing to write to."""
        mod = kx.module.wrap(kx.q('`sub`f!((``a!((::);1));{x})'))
        with pytest.raises(TypeError, match='has no q namespace to write to'):
            mod.sub.a = kx.q('2')


class TestReprAgreement:
    """`repr` must name the type attribute access returns."""

    def rows(self, mod):
        return [tuple(p.strip() for p in ln.split('|', 1))
                for ln in repr(mod).split('\n') if '|' in ln]

    def test_every_row_matches_its_attribute(self, kx, shapes_tree):
        kx.module.SearchPath(shapes_tree, add=True, prepend=True)
        mod = kx.use('kx.shapes')
        for name, shown in self.rows(mod):
            assert shown == 'pykx.' + type(getattr(mod, name)).__name__

    def test_nested_namespace_is_not_listed_as_a_dictionary(self, kx, shapes_tree):
        kx.module.SearchPath(shapes_tree, add=True, prepend=True)
        assert 'nested| pykx.QNamespaceProxy' in repr(kx.use('kx.shapes'))

    def test_proxy_repr_names_its_own_class(self, kx, proxy_tree):
        kx.module.SearchPath(proxy_tree, add=True, prepend=True)
        assert repr(kx.use('kx.proxymod').OFFSET).startswith('pykx.QNamespaceProxy(')


class TestMappingProtocol:
    """A module and a nested namespace both read as a mapping of their exports."""

    def load(self, kx, tree, name='kx.proxymod'):
        kx.module.SearchPath(tree, add=True, prepend=True)
        return kx.use(name)

    def test_module_subscripting(self, kx, proxy_tree):
        m = self.load(kx, proxy_tree)
        assert m['f'](1).py() == 2

    def test_proxy_reachable_by_attribute_and_by_key(self, kx, proxy_tree):
        """Both routes must work; the proxy used to be attribute-only."""
        m = self.load(kx, proxy_tree)
        assert m.OFFSET.BEGINNING.py() == -2
        assert m.OFFSET['BEGINNING'].py() == -2

    def test_symbol_key_accepted(self, kx, proxy_tree):
        m = self.load(kx, proxy_tree)
        assert m[kx.SymbolAtom('f')](1).py() == 2

    def test_missing_key_raises_keyerror(self, kx, proxy_tree):
        m = self.load(kx, proxy_tree)
        with pytest.raises(KeyError):
            m['nosuchexport']

    def test_non_string_key_rejected(self, kx, proxy_tree):
        m = self.load(kx, proxy_tree)
        with pytest.raises(TypeError, match='must be a string'):
            m[0]

    def test_contains_checks_names(self, kx, proxy_tree):
        m = self.load(kx, proxy_tree)
        assert 'f' in m
        assert 'nosuchexport' not in m

    def test_len_counts_exports(self, kx, proxy_tree):
        m = self.load(kx, proxy_tree)
        assert len(m) == 2 == len(dir(m))
        assert len(m.OFFSET) == 2                   # and a nested namespace counts its own

    def test_len_includes_a_registered_submodule(self, kx, submodule_tree):
        kx.module.SearchPath(submodule_tree, add=True, prepend=True)
        parent = kx.use('kx.parent')
        kx.use('kx.parent.sub')
        assert len(parent) == 2
        assert parent['sub'] is kx.use('kx.parent.sub')


class TestConversion:
    """A module converts to the q dictionary behind it."""

    def test_toq_returns_the_export_dictionary(self, kx, shadow_tree):
        kx.module.SearchPath(shadow_tree, add=True, prepend=True)
        mod = kx.use('kx.shadowdir')
        converted = kx.toq(mod)
        assert isinstance(converted, kx.Dictionary)
        assert converted.py() == kx.module.unwrap(mod).py()

    def test_toq_of_a_nested_namespace(self, kx, proxy_tree):
        kx.module.SearchPath(proxy_tree, add=True, prepend=True)
        converted = kx.toq(kx.use('kx.proxymod').OFFSET)
        assert isinstance(converted, kx.Dictionary)
        assert converted['BEGINNING'].py() == -2

    def test_assignment_into_q(self, kx, shadow_tree):
        """`kx.q['name'] = mod` used to raise, leaving no way to bind a module for q."""
        kx.module.SearchPath(shadow_tree, add=True, prepend=True)
        kx.q['shadowmod'] = kx.use('kx.shadowdir')
        assert kx.q('shadowmod`f')(b'x').py() == b'exported x'

    def test_ktype_is_applied_to_the_dictionary(self, kx, shadow_tree):
        kx.module.SearchPath(shadow_tree, add=True, prepend=True)
        mod = kx.use('kx.shadowdir')
        assert isinstance(kx.toq(mod, ktype=kx.Dictionary), kx.Dictionary)

    def test_a_fileno_export_does_not_look_like_a_file_handle(self, kx, fileno_tree):
        """The module check must precede toq's duck-typed ones.

        A module resolves its exported names as attributes, so a module exporting `fileno`
        satisfies `hasattr(x, 'fileno')` and would be converted as an open file descriptor.
        """
        kx.module.SearchPath(fileno_tree, add=True, prepend=True)
        mod = kx.use('kx.fileno')
        assert hasattr(mod, 'fileno')                   # what would mislead the dispatch
        converted = kx.toq(mod)
        assert isinstance(converted, kx.Dictionary)
        assert sorted(k.py() for k in converted.keys()) == ['f', 'fileno']


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

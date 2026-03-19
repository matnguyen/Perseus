import importlib
import pytest

MODULE = "perseus.utils.tax_utils"
GLOBALS_MODULE = "perseus.utils.globals"
CONSTANTS_MODULE = "perseus.utils.constants"
    
"""
Tests for normalize_taxid
"""
def test_normalize_int_with_lineage(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    # fake lineage: last element should be returned
    def fake_get_lineage(tid):
        assert tid == 123
        return [1, 2, 1234]

    globals_mod.NCBI = type("NCBIStub", (), {"get_lineage": staticmethod(fake_get_lineage)})

    m.normalize_taxid.cache_clear()
    out = m.normalize_taxid(123)
    assert out == 1234


def test_normalize_str_int(monkeypatch):
    m = importlib.import_module(MODULE)

    def fake_get_lineage(tid):
        assert tid == 123
        return [1, 2, 123]

    m.NCBI = type("NCBIStub", (), {"get_lineage": staticmethod(fake_get_lineage)})

    m.normalize_taxid.cache_clear()
    out = m.normalize_taxid("123")
    assert out == 123


def test_normalize_taxonomy_string(monkeypatch):
    """Handles strings like '(123)' or 'foo bar (123)'."""
    m = importlib.import_module(MODULE)

    def fake_get_lineage(tid):
        assert tid == 123
        return [1, 2, 123]

    m.NCBI = type("NCBIStub", (), {"get_lineage": staticmethod(fake_get_lineage)})

    m.normalize_taxid.cache_clear()
    out = m.normalize_taxid("something (123)")
    assert out == 123   # last of lineage


def test_normalize_uses_tid_when_no_lineage(monkeypatch):
    """If lineage is empty, returns tid unchanged."""
    m = importlib.import_module(MODULE)

    def fake_get_lineage(tid):
        return []

    m.NCBI = type("NCBIStub", (), {"get_lineage": staticmethod(fake_get_lineage)})

    m.normalize_taxid.cache_clear()
    out = m.normalize_taxid(999)
    assert out == 999


def test_normalize_on_get_lineage_exception(monkeypatch):
    """If NCBI.get_lineage raises, we fall back to the parsed tid."""
    m = importlib.import_module(MODULE)

    def fake_get_lineage(tid):
        raise RuntimeError("DB error")

    m.NCBI = type("NCBIStub", (), {"get_lineage": staticmethod(fake_get_lineage)})

    m.normalize_taxid.cache_clear()
    out = m.normalize_taxid("999")
    assert out == 999

"""
Tests for fetch_maps
"""
def test_fetch_maps_good_path(monkeypatch):
    m = importlib.import_module(MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    class FakeNCBI:
        def get_lineage(self, tid):
            assert tid == 60
            return [2, 10, 20, 30, 40, 50, 60]

        def get_descendant_taxa(self, tid, collapse_subspecies=False, intermediate_nodes=True):
            assert tid == 60
            return [61, 62]

        def get_rank(self, ids):
            mapping = {
                2: "superkingdom",
                10: "phylum",
                20: "class",
                30: "order",
                40: "family",
                50: "genus",
                60: "species",
            }
            return {int(t): mapping.get(int(t)) for t in ids}

    def fake_get_ncbi(db_dir):
        assert str(db_dir) == "/fake/db"
        return FakeNCBI()

    monkeypatch.setattr(m, "get_ncbi", fake_get_ncbi)

    tid, lineage, descendants, canon = m.fetch_maps((60, "/fake/db"))

    assert tid == 60
    assert lineage == [2, 10, 20, 30, 40, 50, 60]
    assert descendants == {61, 62}
    assert canon["superkingdom"] == 2
    assert canon["phylum"] == 10
    assert canon["class"] == 20
    assert canon["order"] == 30
    assert canon["family"] == 40
    assert canon["genus"] == 50
    assert canon["species"] == 60

    for r in constants_mod.CANONICAL_RANKS:
        assert r in canon


def test_fetch_maps_error_path(monkeypatch):
    m = importlib.import_module(MODULE)
    
    class RaisingNCBI:
        def get_lineage(self, tid):
            raise RuntimeError("boom")

        def get_descendant_taxa(self, tid, collapse_subspecies=False, intermediate_nodes=True):
            raise RuntimeError("boom")

        def get_rank(self, ids):
            raise RuntimeError("boom")

    monkeypatch.setattr(m, "get_ncbi", lambda db_dir: RaisingNCBI())

    tid, lineage, descendants, canon = m.fetch_maps((999, "/fake/db"))

    assert tid == 999
    assert lineage == []
    assert descendants == set()
    assert set(canon.keys()) == set(m.CANONICAL_RANKS)
    assert all(v is None for v in canon.values())

"""
Tests for get_lineage_path
"""
def test_get_lineage_path_int_taxid():
    """
    For a normal taxid, get_lineage_path should just delegate to NCBI.get_lineage
    and return the same list.
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    m.get_lineage_path.cache_clear()
    tid = 60
    expected = globals_mod.NCBI.get_lineage(tid)
    out = m.get_lineage_path(tid)

    assert out == expected
    # sanity: our FakeNCBI lineage contains the tid itself
    assert out[-1] == 60


def test_get_lineage_path_string_taxid():
    """
    Accepts strings and converts to int correctly.
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    m.get_lineage_path.cache_clear()
    expected = globals_mod.NCBI.get_lineage(60)
    out = m.get_lineage_path("60")

    assert out == expected


def test_get_lineage_path_on_exception(monkeypatch):
    """
    If NCBI.get_lineage raises, get_lineage_path should return [].
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    def boom(tid):
        raise RuntimeError("DB failure")

    # patch the global NCBI used inside get_lineage_path
    globals_mod.NCBI.get_lineage = boom
    m.get_lineage_path.cache_clear()

    out = m.get_lineage_path(999999)
    assert out == []


def test_get_lineage_path_caching(monkeypatch):
    """
    Basic sanity: once cached, subsequent calls should not depend on NCBI changes.
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    # use real FakeNCBI once
    m.get_lineage_path.cache_clear()
    first = m.get_lineage_path(60)

    # now make NCBI.get_lineage misbehave; cached result should still be returned
    def boom(tid):
        raise RuntimeError("should not be called if cached")

    globals_mod.NCBI.get_lineage = boom

    second = m.get_lineage_path(60)
    assert second == first

"""
Tests for lineage_to_rank_map
"""
def test_lineage_to_rank_map_basic():
    """
    For a normal lineage from FakeNCBI (species 60), we should map:
      superkingdom -> 2
      phylum       -> 10
      class        -> 20
      order        -> 30
      family       -> 40
      genus        -> 50
      species      -> 60
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    lineage = globals_mod.NCBI.get_lineage(60)  # from FakeNCBI in conftest
    rank_map = m.lineage_to_rank_map(lineage, constants_mod.CANONICAL_RANKS)

    assert rank_map["superkingdom"] == 2
    assert rank_map["phylum"] == 10
    assert rank_map["class"] == 20
    assert rank_map["order"] == 30
    assert rank_map["family"] == 40
    assert rank_map["genus"] == 50
    assert rank_map["species"] == 60

    # all canonical ranks present as keys
    assert set(rank_map.keys()) == set(constants_mod.CANONICAL_RANKS)


def test_lineage_to_rank_map_empty_lineage():
    """
    If lineage is empty, we should get all canonical ranks mapped to None.
    """
    m = importlib.import_module(MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    rank_map = m.lineage_to_rank_map([], constants_mod.CANONICAL_RANKS)

    assert set(rank_map.keys()) == set(constants_mod.CANONICAL_RANKS)
    assert all(v is None for v in rank_map.values())


def test_lineage_to_rank_map_partial_lineage(monkeypatch):
    """
    If some ranks are missing from the lineage, only present ranks should be filled;
    others stay None.
    """
    m = importlib.import_module(MODULE)

    # Fake a short lineage that has only superkingdom and family
    short_lineage = [2, 40]  # 2=superkingdom, 40=family in FakeNCBI

    def fake_get_rank(ids):
        # Only provide ranks for the IDs we care about; others None
        mapping = {
            2: "superkingdom",
            40: "family",
        }
        return {int(t): mapping.get(int(t), None) for t in ids}

    # Monkeypatch NCBI.get_rank just for this test
    m.NCBI.get_rank = fake_get_rank

    rank_map = m.lineage_to_rank_map(short_lineage, m.CANONICAL_RANKS)

    assert rank_map["superkingdom"] == 2
    assert rank_map["family"] == 40

    # Other canonical ranks should exist but be None
    for r in m.CANONICAL_RANKS:
        if r not in ("superkingdom", "family"):
            assert rank_map[r] is None

"""
Tests for get_canonical_taxid_for_rank
"""
def test_get_canonical_taxid_for_rank_basic():
    """
    Using FakeNCBI’s tiny taxonomy (2→10→20→30→40→50→60),
    get_canonical_taxid_for_rank(60, CANONICAL_RANKS) should map:
      superkingdom -> 2
      phylum       -> 10
      class        -> 20
      order        -> 30
      family       -> 40
      genus        -> 50
      species      -> 60
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    mapping = m.get_canonical_taxid_for_rank(
        60,
        constants_mod.CANONICAL_RANKS,
        globals_mod.NCBI,      
    )

    assert set(mapping.keys()) == set(m.CANONICAL_RANKS)
    assert mapping["superkingdom"] == 2
    assert mapping["phylum"] == 10
    assert mapping["class"] == 20
    assert mapping["order"] == 30
    assert mapping["family"] == 40
    assert mapping["genus"] == 50
    assert mapping["species"] == 60


def test_get_canonical_taxid_for_rank_missing_taxid(monkeypatch):
    """
    If NCBI.get_lineage raises, function should return all canonical
    ranks mapped to None.
    """
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    def boom(tid):
        raise RuntimeError("nope")

    # only patch get_lineage; get_rank won’t be reached
    globals_mod.NCBI.get_lineage = boom

    mapping = m.get_canonical_taxid_for_rank(
        999999,
        m.CANONICAL_RANKS,
        m.NCBI,
    )

    assert set(mapping.keys()) == set(m.CANONICAL_RANKS)
    assert all(val is None for val in mapping.values())

"""
Tests for get_taxid_rank_raw
"""
def test_get_taxid_rank_raw_known():
    """
    For a known taxid (60), should return the raw rank string ("species")
    as provided by FakeNCBI.get_rank.
    """
    m = importlib.import_module(MODULE)

    m.get_taxid_rank_raw.cache_clear()
    rank = m.get_taxid_rank_raw(60)

    assert rank == "species"


def test_get_taxid_rank_raw_unknown():
    """
    For a taxid that FakeNCBI doesn't know, get_rank returns {tid: None},
    so get_taxid_rank_raw should return None.
    """
    m = importlib.import_module(MODULE)

    m.get_taxid_rank_raw.cache_clear()
    rank = m.get_taxid_rank_raw(999999)

    assert rank is None


def test_get_taxid_rank_raw_exception_and_cache(monkeypatch):
    """
    If NCBI.get_rank raises, get_taxid_rank_raw should return None.
    Also sanity-check that caching works.
    """
    m = importlib.import_module(MODULE)

    # normal behavior once
    m.get_taxid_rank_raw.cache_clear()
    first = m.get_taxid_rank_raw(60)
    assert first == "species"

    # now force errors; cached value should still be returned
    def boom(ids):
        raise RuntimeError("should not be called if cached")

    m.NCBI.get_rank = boom
    second = m.get_taxid_rank_raw(60)
    assert second == "species"

    # for a new taxid (not cached), error → None
    third = m.get_taxid_rank_raw(123456)
    assert third is None

"""
Tests for canonicalize_rank
"""
@pytest.mark.parametrize("raw,expected", [
    ("Kingdom", "superkingdom"),     # special-case mapping
    ("kingdom", "superkingdom"),
    ("superkingdom", "superkingdom"),
    ("phylum", "phylum"),
    ("Class", "class"),
    ("order", "order"),
    ("family", "family"),
    ("genus", "genus"),
    ("species", "species"),
    ("strain", "strain"),
    ("subclass", "class"),    # TODO: decide on handling sub-canonical ranks
    ("weird_rank", None),
    ("", None),
    (None, None),
])
def test_canonicalize_rank_cases(raw, expected):
    m = importlib.import_module(MODULE)
    assert m.canonicalize_rank(raw) == expected

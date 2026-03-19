import importlib
import pytest

# Path to module under test
MODULE = "perseus.commands.extract"  
NCBI_MODULE = "perseus.utils.tax_utils"
GLOBALS_MODULE = "perseus.utils.globals"
CONSTANTS_MODULE = "perseus.utils.constants"

# ---- Fake NCBI / ETE3 ----
class FakeNCBI:
    """
    Minimal behaviors 
      - get_lineage(int)
      - get_rank(list[int] or [int])
      - get_descendant_taxa(int, ...)
    Set up a tiny tax graph:
      superkingdom(2) -> phylum(10) -> class(20) -> order(30) -> family(40) -> genus(50) -> species(60)
      Plus a sibling species(61) under the same genus(50)
    """
    def __init__(self, dbfile=None):
        self.dbfile = dbfile or "/tmp/fake.sqlite"

    def get_lineage(self, tid):
        tid = int(tid)
        # Build chain bottom-up
        chain = {
            2: [2],
            10: [2,10],
            20: [2,10,20],
            30: [2,10,20,30],
            40: [2,10,20,30,40],
            50: [2,10,20,30,40,50],
            60: [2,10,20,30,40,50,60],
            61: [2,10,20,30,40,50,61],
        }
        return chain.get(tid, [tid])

    def get_rank(self, ids):
        # Accept either list[int] or lineage list
        ranks = {
            2: "superkingdom",
            10: "phylum",
            20: "class",
            30: "order",
            40: "family",
            50: "genus",
            60: "species",
            61: "species",
        }
        return {int(t): ranks.get(int(t), None) for t in ids}

    def get_descendant_taxa(self, tid, collapse_subspecies=False, intermediate_nodes=True):
        # Descendants in the tiny graph
        descendants = {
            2:  [10,20,30,40,50,60,61],
            10: [20,30,40,50,60,61],
            20: [30,40,50,60,61],
            30: [40,50,60,61],
            40: [50,60,61],
            50: [60,61],
            60: [],
            61: [],
        }
        return descendants.get(int(tid), [])

@pytest.fixture(autouse=True)
def patch_ncbi(monkeypatch):
    """
    Patch the module's NCBI and NCBITaxa class so tests don't touch real ETE3 DB
    Also reset LRU caches between tests
    """
    m = importlib.import_module(NCBI_MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)
    constants_mod = importlib.import_module(CONSTANTS_MODULE)

    # Replace class and the global instance
    monkeypatch.setattr(m, "NCBITaxa", FakeNCBI, raising=True)
    globals_mod.NCBI = FakeNCBI()

    # Reset caches that rely on NCBI
    for name in ("cached_get_rank", "get_lineage_path", "get_taxid_to_rank",
                 "get_descendants", "normalize_taxid", "get_taxid_rank_raw"):
        getattr(m, name).cache_clear()

    # Provide sane defaults for shared maps used by compute_bin_features
    globals_mod._shared_lineage_map = {
        60: globals_mod.NCBI.get_lineage(60),
        61: globals_mod.NCBI.get_lineage(61),
        50: globals_mod.NCBI.get_lineage(50),
    }
    # canonical map (rank->taxid) per taxid
    def canon_for(t):
        lineage = globals_mod.NCBI.get_lineage(t)
        return m.get_canonical_taxid_for_rank(t, constants_mod.CANONICAL_RANKS, globals_mod.NCBI)
    globals_mod._shared_canonical_map = {t: canon_for(t) for t in (60,61,50,40,30,20,10,2)}
    # descendants map
    globals_mod._shared_descendant_map = {t: set(globals_mod.NCBI.get_descendant_taxa(t)) for t in (60,61,50,40,30,20,10,2)}
    yield

@pytest.fixture
def tmp_outdir(tmp_path, monkeypatch):
    """Set the module's shared output directory to a tmp path for writer tests"""
    m = importlib.import_module(MODULE)
    m._shared_out_dir = str(tmp_path)
    return tmp_path

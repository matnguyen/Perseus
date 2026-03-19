import importlib
import os
from pathlib import Path

import pytest

# Replace this with the actual module path containing these functions
MODULE = "perseus.features.init"
GLOBALS_MODULE = "perseus.utils.globals"


@pytest.mark.dev
def test_effective_nprocs_from_slurm_env(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    out = m.effective_nprocs()

    assert out == 8


@pytest.mark.dev
def test_effective_nprocs_from_slurm_env_invalid_falls_back_to_affinity(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not_an_int")
    monkeypatch.setattr(m.os, "sched_getaffinity", lambda pid: {0, 1, 2})

    out = m.effective_nprocs()

    assert out == 3


@pytest.mark.dev
def test_effective_nprocs_affinity_fallback_to_cpu_count(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

    def fake_sched_getaffinity(pid):
        raise OSError("no affinity")

    monkeypatch.setattr(m.os, "sched_getaffinity", fake_sched_getaffinity)
    monkeypatch.setattr(m.os, "cpu_count", lambda: 12)

    out = m.effective_nprocs()

    assert out == 12


@pytest.mark.dev
def test_effective_nprocs_cpu_count_none_returns_one(monkeypatch):
    m = importlib.import_module(MODULE)

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

    def fake_sched_getaffinity(pid):
        raise OSError("no affinity")

    monkeypatch.setattr(m.os, "sched_getaffinity", fake_sched_getaffinity)
    monkeypatch.setattr(m.os, "cpu_count", lambda: None)

    out = m.effective_nprocs()

    assert out == 1


def test_cleanup_ete3_tmpdir_deletes_existing_dir(tmp_path):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    tmpdir = tmp_path / "ete3_tmp"
    tmpdir.mkdir()
    (tmpdir / "taxa.sqlite").write_text("db")

    globals_mod._ete3_tmpdir = str(tmpdir)

    m.cleanup_ete3_tmpdir()

    assert not tmpdir.exists()


def test_cleanup_ete3_tmpdir_noop_when_missing(tmp_path):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    tmpdir = tmp_path / "does_not_exist"
    globals_mod._ete3_tmpdir = str(tmpdir)

    m.cleanup_ete3_tmpdir()

    assert not tmpdir.exists()


def test_cleanup_ete3_tmpdir_noop_when_none():
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    globals_mod._ete3_tmpdir = None

    m.cleanup_ete3_tmpdir()

    assert globals_mod._ete3_tmpdir is None


def test_init_ncbi_private_db_copies_sqlite_and_sets_globals(monkeypatch, tmp_path):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    db_path = tmp_path / "db"
    db_path.mkdir()
    sqlite_src = db_path / "taxa.sqlite"
    sqlite_src.write_text("fake sqlite db")

    fake_ncbi = object()

    monkeypatch.setattr(m, "get_ncbi", lambda path: fake_ncbi)

    m._init_ncbi_private_db(str(db_path))

    tmpdir = Path(globals_mod._ete3_tmpdir)
    copied_sqlite = tmpdir / "taxa.sqlite"

    assert tmpdir.exists()
    assert copied_sqlite.exists()
    assert copied_sqlite.read_text() == "fake sqlite db"
    assert globals_mod.NCBI is fake_ncbi


def test_init_worker_sets_globals_and_registers_cleanup(monkeypatch, tmp_path):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    calls = {"db_path": None, "registered": None}

    def fake_init_ncbi_private_db(db_path):
        calls["db_path"] = db_path

    def fake_register(fn):
        calls["registered"] = fn

    monkeypatch.setattr(m, "_init_ncbi_private_db", fake_init_ncbi_private_db)
    monkeypatch.setattr(m.atexit, "register", fake_register)

    tc = {"a": 1}
    lineage_map = {1: [1, 2]}
    descendant_map = {1: [2, 3]}
    canonical_map = {1: "species"}
    out_dir = str(tmp_path / "out")
    manifest_paths = ["a.parquet", "b.parquet"]

    m.init_worker(
        tc=tc,
        lineage_map=lineage_map,
        descendant_map=descendant_map,
        canonical_map=canonical_map,
        out_dir=out_dir,
        db_path="/fake/db",
        shard_size=5000,
        target_length=2048,
        to_dtype="float16",
        manifest_paths=manifest_paths,
        mess_true_file="truth.tsv",
        mess_input_file="input.tsv",
    )

    assert globals_mod._shared_tax_context == tc
    assert globals_mod._shared_lineage_map == lineage_map
    assert globals_mod._shared_descendant_map == descendant_map
    assert globals_mod._shared_canonical_map == canonical_map
    assert globals_mod._shared_out_dir == out_dir
    assert globals_mod._shared_shard_size == 5000
    assert globals_mod._shared_target_length == 2048
    assert globals_mod._shared_to_dtype == "float16"
    assert globals_mod._shared_manifest_paths == manifest_paths
    assert globals_mod._shared_mess_true_file == "truth.tsv"
    assert globals_mod._shared_mess_input_file == "input.tsv"

    assert calls["db_path"] == "/fake/db"
    assert calls["registered"] is m.cleanup_ete3_tmpdir


def test_next_worker_part_name_increments(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    globals_mod._worker_part_idx = 0
    monkeypatch.setattr(m.os, "getpid", lambda: 12345)

    p1 = m._next_worker_part_name()
    p2 = m._next_worker_part_name()
    p3 = m._next_worker_part_name(ext="csv")

    assert p1 == "part-p12345-000001.parquet"
    assert p2 == "part-p12345-000002.parquet"
    assert p3 == "part-p12345-000003.csv"


def test_next_worker_part_name_uses_existing_index(monkeypatch):
    m = importlib.import_module(MODULE)
    globals_mod = importlib.import_module(GLOBALS_MODULE)

    globals_mod._worker_part_idx = 41
    monkeypatch.setattr(m.os, "getpid", lambda: 999)

    out = m._next_worker_part_name()

    assert out == "part-p999-000042.parquet"
    assert globals_mod._worker_part_idx == 42
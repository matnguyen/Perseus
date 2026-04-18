import os
import shutil
import tempfile
import atexit

import perseus.utils.globals as globals_mod
from perseus.utils.tax_utils import (
    get_ncbi,
    get_gtdb                                     
)

def effective_nprocs():
    """
    Determine the effective number of CPU cores available to this process

    Returns:
        int: Number of CPU cores
    """
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        try:
            return max(1, int(os.environ['SLURM_CPUS_PER_TASK']))
        except Exception:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def cleanup_ete4_tmpdir():
    """
    Cleanup the temporary ete4 DB directory if it exists
    """
    tmpdir = getattr(globals_mod, "_ete4_tmpdir", None)
    print(f"[CLEANUP] Worker PID {os.getpid()} cleaning up temp dir: {tmpdir}")
    if tmpdir and os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
        print(f"[CLEANUP] Deleted temp dir: {tmpdir}")
    else:
        print(f"[CLEANUP] Temp dir not found or already deleted: {tmpdir}")


def _init_ete4_private_db(db_path: str):
    """
    Create a private copy of the ETE4 SQLite DB for this worker to avoid
    read-lock contention on NFS and reduce D-state stalls
    """
    tmpdir = tempfile.mkdtemp(prefix="perseus_ete4db_")
    dst_db = os.path.join(tmpdir, "taxa.sqlite")
    sqlite_path = os.path.join(db_path, "taxa.sqlite")
    shutil.copy2(sqlite_path, dst_db)

    globals_mod._ete4_tmpdir = tmpdir
    if globals_mod.DB_TYPE == "ncbi":
        globals_mod.DB = get_ncbi(db_path)
    elif globals_mod.DB_TYPE == "gtdb":
        globals_mod.DB = get_gtdb(db_path)
    else:
        raise ValueError(f"Unsupported DB_TYPE: {globals_mod.DB_TYPE}")


def init_worker(
    tc, 
    lineage_map, 
    descendant_map, 
    canonical_map, 
    out_dir,
    db_path,
    shard_size=4096, 
    target_length=1024,
    to_dtype="float32",
    manifest_paths=None,
):
    """
    Init for the feature-extraction pool: set globals + private NCBI DB copy
    """
    globals_mod._shared_tax_context    = tc
    globals_mod._shared_lineage_map    = lineage_map
    globals_mod._shared_descendant_map = descendant_map
    globals_mod._shared_canonical_map  = canonical_map
    globals_mod._shared_out_dir        = out_dir

    globals_mod._shared_shard_size     = int(shard_size)
    globals_mod._shared_target_length  = int(target_length)
    globals_mod._shared_to_dtype       = str(to_dtype)
    globals_mod._shared_manifest_paths = manifest_paths  

    _init_ete4_private_db(db_path)
    atexit.register(cleanup_ete4_tmpdir)


def _next_worker_part_name(ext="parquet"):
    """
    Generate a unique part/shard name for this worker
    """
    globals_mod._worker_part_idx += 1
    return f"part-p{os.getpid()}-{globals_mod._worker_part_idx:06d}.{ext}"
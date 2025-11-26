import os
import shutil
import tempfile
from ete3 import NCBITaxa

import taxoncnn.utils.globals as globals_mod

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


def _init_ncbi_private_db():
    """
    Create a private copy of the ETE3 SQLite DB for this worker to avoid
    read-lock contention on NFS and reduce D-state stalls
    """
    global NCBI
    tmp = NCBITaxa()
    try:
        src_db = tmp.dbfile
    except Exception:
        src_db = os.path.expanduser("~/.etetoolkit/taxa.sqlite")
    if not os.path.exists(src_db):
        _ = NCBITaxa()
        src_db = _.dbfile

    tmpdir = tempfile.mkdtemp(prefix="ete3db_")
    dst_db = os.path.join(tmpdir, "taxa.sqlite")
    shutil.copy2(src_db, dst_db)
    NCBI = NCBITaxa(dbfile=dst_db)


def init_worker(tc, lineage_map, descendant_map, canonical_map, out_dir,
                write_format="parquet", shard_size=4096, target_length=1024,
                to_dtype="float32", manifest_paths=None, 
                mess_true_file=None, mess_input_file=None):
    """
    Init for the feature-extraction pool: set globals + private NCBI DB copy
    """
    globals_mod._shared_tax_context    = tc
    globals_mod._shared_lineage_map    = lineage_map
    globals_mod._shared_descendant_map = descendant_map
    globals_mod._shared_canonical_map  = canonical_map
    globals_mod._shared_out_dir        = out_dir

    globals_mod._shared_write_format   = write_format
    globals_mod._shared_shard_size     = int(shard_size)
    globals_mod._shared_target_length  = int(target_length)
    globals_mod._shared_to_dtype       = str(to_dtype)
    globals_mod._shared_manifest_paths = manifest_paths  # Manager.list shared across workers
    globals_mod._shared_mess_true_file = mess_true_file
    globals_mod._shared_mess_input_file = mess_input_file

    _init_ncbi_private_db()


def _next_worker_part_name(ext="parquet"):
    """
    Generate a unique part/shard name for this worker
    """
    globals_mod._worker_part_idx += 1
    return f"part-p{os.getpid()}-{globals_mod._worker_part_idx:06d}.{ext}"
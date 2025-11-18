# Shared global variables for multiprocessing workers
_shared_tax_context    = None
_shared_lineage_map    = None
_shared_descendant_map = None
_shared_canonical_map  = None
NCBI                   = None  # each worker gets its own handle (and its own DB copy)
_shared_out_dir        = None
_worker_part_idx       = 0

# Shared writing controls for multiprocessing workers
_shared_write_format   = "parquet"   # "parquet" | "shards"
_shared_shard_size     = 4096
_shared_target_length  = 0           # 0 = pad to shard max
_shared_to_dtype       = "float32"   # "float32" | "float16" | "bfloat16"
_shared_manifest_paths = None        # manager list for shard paths
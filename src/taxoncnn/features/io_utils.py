import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import torch
import logging
from threading import Thread
from queue import Queue

import taxoncnn.features.globals as globals
from taxoncnn.features.constants import CANONICAL_RANKS
from taxoncnn.features.init import _next_worker_part_name
from taxoncnn.features.features import (
    _torch_dtype,
    _resample_TN_to_T
)

logger = logging.getLogger(__name__)


def _write_rows_streaming_parquet(rows, max_batch_rows=256, use_half=False, quantize_u8=False):
    """
    Writes a Parquet part (nested lists); flushes every max_batch_rows rows

    Args:
        rows (Iterable[dict]): Feature rows to write. Iterable of {
            'seq_id': str, 'taxon': int, 'bins': list[list[float]],
            'label_any': int, 'label_rank': int, 'labels_per_rank': list[int],
            'pred_rank': str|None, 'rank_index': int
        }
        max_batch_rows (int, optional): Max rows per batch. Defaults to 256
        use_half (bool, optional): Use float16 for bins if True. Defaults to False
        quantize_u8 (bool, optional): Quantize bins to uint8 if True. Defaults to False

    Returns:
        dict or None: Returns dict with number of rows written and filename, or None if no rows
    """
    if not rows:
        return None

    bins_scalar_type = pa.float32()
    if use_half:
        try:
            bins_scalar_type = pa.float16()
        except Exception:
            bins_scalar_type = pa.float32()
    if quantize_u8:
        bins_scalar_type = pa.uint8()

    fname = _next_worker_part_name("parquet")
    fpath = os.path.join(globals._shared_out_dir, fname)

    writer = None
    wrote = 0

    def _flush_batch(buf):
        nonlocal writer, wrote
        if not buf:
            return
        seq_ids = pa.array([r['seq_id'] for r in buf], type=pa.string())
        taxa    = pa.array([int(r['taxon']) for r in buf], type=pa.int32())

        # legacy single label kept as label_any for compatibility
        labels_any   = pa.array([int(r.get('label_any', 0)) for r in buf], type=pa.int8())
        labels_rank  = pa.array([int(r.get('label_rank', 0)) for r in buf], type=pa.int8())
        rank_index   = pa.array([int(r.get('rank_index', -1)) for r in buf], type=pa.int8())
        pred_rank    = pa.array([str(r.get('pred_rank')) if r.get('pred_rank') is not None else None for r in buf], type=pa.string())

        # list<int8> per-rank
        lpr_field = pa.list_(pa.int8())
        lpr_arr   = pa.array([r.get('labels_per_rank', []) for r in buf], type=lpr_field)

        bins_field = pa.list_(pa.list_(bins_scalar_type))
        if quantize_u8:
            scaled = []
            for r in buf:
                scaled.append([[int(round(x * 255.0)) for x in row] for row in r['bins']])
            bins_arr = pa.array(scaled, type=bins_field)
        else:
            bins_arr = pa.array([r['bins'] for r in buf], type=bins_field)

        # legacy 'label' column (mirror of label_any)
        tbl = pa.table({
            'seq_id': seq_ids,
            'taxon': taxa,
            'bins': bins_arr,
            'label': labels_any,          # legacy
            'label_any': labels_any,      # new
            'label_rank': labels_rank,    # new
            'labels_per_rank': lpr_arr,   # new
            'pred_rank': pred_rank,       # new
            'rank_index': rank_index,     # new
        })

        if writer is None:
            writer = pq.ParquetWriter(fpath, tbl.schema,
                                      compression="zstd",
                                      use_dictionary=True,
                                      write_statistics=False)
        writer.write_table(tbl, row_group_size=len(buf))
        wrote += len(buf)
        buf.clear()

    batch = []
    for r in rows:
        batch.append(r)
        if len(batch) >= max_batch_rows:
            _flush_batch(batch)
    _flush_batch(batch)

    if writer is not None:
        writer.close()

    return {'rows': wrote, 'file': fname}

def _write_rows_streaming_shards(rows, max_batch_rows=4096, target_length=1024, to_dtype="float32"):
    """
    Writes a shard .pt containing:
        x: [N, C=28, T]  (resampled to target_length if >0; else pad to max T in shard)
        y: [N] (float32, legacy mirror of y_any)
        y_any: [N] (float32)
        y_rank: [N] (float32)
        y_per_rank: [N, R] (float32, R=len(CANONICAL_RANKS))
        rank_index: [N] (int8)
        seq_id (list[str]), taxon (list[int]), lengths: [N] (if target_length==0)

    Args:
        rows (Iterable[dict]): Feature rows to write. Iterable of {
            'seq_id': str, 'taxon': int, 'bins': list[list[float]],
            'label_any': int, 'label_rank': int, 'labels_per_rank': list[int],
            'pred_rank': str|None, 'rank_index': int
        }
        max_batch_rows (int, optional): Max rows per batch. Defaults to 4096
        target_length (int, optional): Resample to this length if >0. Defaults to 1024
        to_dtype (str, optional): Target dtype for x. Defaults to "float32"

    Returns:
        dict or None: Returns dict with number of rows written and filename, or None if no rows
    """
    if not rows:
        return None

    C = 22
    R = len(CANONICAL_RANKS)
    fname = _next_worker_part_name("pt")
    fpath = os.path.join(globals._shared_out_dir, fname)

    X_list, y_any_list, y_rank_list, y_per_rank_list = [], [], [], []
    id_list, tax_list, true_tax_list, len_list, rank_index_list = [], [], [], [], []
    T_max = 0
    tl = int(target_length)
    for r in rows:
        arr = np.asarray(r['bins'], dtype=np.float32)  # [T,28]
        if arr.ndim == 1: arr = arr[0:1]
        if arr.shape[1] != C:
            continue
        if tl > 0 and arr.shape[0] != tl:
            arr = _resample_TN_to_T(arr, tl)
        T = arr.shape[0]
        T_max = max(T_max, T)
        X_list.append(torch.from_numpy(arr.T))   # [28,T]

        # labels
        y_any_list.append(float(r.get('label_any', 0)))
        y_rank_list.append(float(r.get('label_rank', 0)))
        lpr = r.get('labels_per_rank', [0]*R)
        if len(lpr) != R:
            tmp = np.zeros(R, dtype=np.float32)
            m = min(R, len(lpr))
            tmp[:m] = np.asarray(lpr[:m], dtype=np.float32)
            lpr = tmp.tolist()
        y_per_rank_list.append(torch.tensor(lpr, dtype=torch.float32))

        id_list.append(str(r.get('seq_id', "")))
        tax_list.append(int(r.get('taxon', -1)))
        true_tax_list.append(int(r.get('true_taxon', -1)))  # <--- ADD THIS LINE
        rank_index_list.append(int(r.get('rank_index', -1)))
        len_list.append(T)
        
        if len(X_list) >= max_batch_rows:
            break

    if not X_list:
        return None

    # Pad or stack
    dt = _torch_dtype(to_dtype)
    if tl > 0:
        T_final = tl
        x = torch.stack([xi.to(dt) for xi in X_list], dim=0)  # [N,28,T]
    else:
        T_final = T_max
        x = torch.zeros(len(X_list), C, T_final, dtype=dt)
        for i, xi in enumerate(X_list):
            t = xi.shape[-1]
            x[i, :, :t] = xi.to(dt)

    y_any  = torch.tensor(y_any_list, dtype=torch.float32)
    y_rank = torch.tensor(y_rank_list, dtype=torch.float32)
    y_pr   = torch.stack(y_per_rank_list, dim=0)  # [N,R]
    # legacy mirror
    y_legacy = y_any.clone()

    bundle = {
        "x": x,
        "y": y_legacy,              # legacy
        "y_any": y_any,
        "y_rank": y_rank,
        "y_per_rank": y_pr,
        "seq_id": id_list,
        "taxon": tax_list,
        "true_taxon": true_tax_list,  
        "rank_index": torch.tensor(rank_index_list, dtype=torch.int8),
    }
    if tl <= 0:
        bundle["lengths"] = torch.tensor(len_list, dtype=torch.int32)

    # Write shard
    torch.save(bundle, fpath)

    # record path into shared manifest list (thread/process-safe)
    try:
        if globals._shared_manifest_paths is not None:
            globals._shared_manifest_paths.append(fpath)
    except Exception:
        pass

    return {'rows': len(X_list), 'file': fname}


def prefetch(iterable, bufsize=32):
    """
    Prefetches items from iterable using a separate thread

    Args:
        iterable (Iterable[T]): Input iterable
        bufsize (int, optional): Buffer size. Defaults to 32

    Yields:
        T: Items from iterable
    """
    q = Queue(maxsize=bufsize)
    SENTINEL = object()
    def _fill():
        for item in iterable:
            q.put(item)
        q.put(SENTINEL)
    Thread(target=_fill, daemon=True).start()
    while True:
        item = q.get()
        if item is SENTINEL:
            return
        yield item
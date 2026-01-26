import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import torch
import logging
from threading import Thread
from queue import Queue

import perseus.utils.globals as globals
from perseus.utils.constants import CANONICAL_RANKS
from perseus.features.init import _next_worker_part_name
from perseus.features.features import (
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
        # labels_any   = pa.array([int(r.get('label_any', 0)) for r in buf], type=pa.int8())
        # labels_rank  = pa.array([int(r.get('label_rank', 0)) for r in buf], type=pa.int8())
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
            # 'label': labels_any,          # legacy
            # 'label_any': labels_any,      # new
            # 'label_rank': labels_rank,    # new
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
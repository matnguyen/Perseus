import numpy as np
import pandas as pd
from alive_progress import alive_bar

def select_one_row_per_seq(
    df: pd.DataFrame,
    sequence_col: str = "sequence_id",
    ranks=None,
    thresholds=0.5,                  # float or dict {rank: thr}
    lineage_filter_col: str | None = "perseus_in_lineage",
    prefer_lineage: bool = True,     # if True: restrict to lineage rows when available; else just tie-break
    tie_breaker: str = "sum_to_rank" # "sum_to_rank" or "p_only" or "deep_then_sum"
):
    """
      1) Choose the deepest rank r such that max(prob_r) over candidates >= threshold_r.
      2) Choose the candidate row with highest prob_r at that rank.
      3) Optional lineage handling:
         - If prefer_lineage=True and lineage_filter_col is provided:
             use only lineage rows IF the contig has any lineage rows; otherwise fallback to all rows.
         - If prefer_lineage=False: do not filter; lineage tie-break not implemented.

    Additionally:
      - Rows with no finite prob_* values across all ranks are skipped (treated as non-candidates).

    thresholds:
      - float: same threshold for all ranks
      - dict: per-rank thresholds, e.g. {"species":0.8,"genus":0.7,...}

    tie_breaker:
      - "p_only": only prob at chosen rank
      - "sum_to_rank": then break ties by sum of probs up to chosen rank
      - "deep_then_sum": then break ties by (sum to rank), then by deeper-rank mass (sum of probs below rank)
    """
    df = df.copy()

    # infer ranks from prob_* columns if not provided
    if ranks is None:
        base = ["superkingdom", "phylum", "class", "order", "family", "genus", "species", "subspecies"]
        ranks = [r for r in base if f"prob_{r}" in df.columns]
    if not ranks:
        raise ValueError("No prob_* rank columns found.")

    prob_cols = [f"prob_{r}" for r in ranks]
    P = df[prob_cols].to_numpy(dtype=np.float64, copy=False)  # shape [N, R]
    R = len(ranks)

    # thresholds per rank
    if isinstance(thresholds, (int, float)):
        thr = np.full(R, float(thresholds), dtype=np.float64)
    elif isinstance(thresholds, dict):
        thr = np.array([float(thresholds.get(r, 0.5)) for r in ranks], dtype=np.float64)
    else:
        raise TypeError("thresholds must be a float or dict{rank:thr}")

    # group indices by contig (positional indices into df)
    g = df.groupby(sequence_col, sort=False).indices  # dict: contig -> np array of row idx (positional)
    out_pos = np.empty(len(g), dtype=np.int64)

    # precompute cumulative sums for tie-breakers (treat NaN/inf as 0)
    P0 = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    cumsum = np.cumsum(P0, axis=1)  # sum up to each rank (inclusive)

    def _choose_rank(Pblock: np.ndarray) -> int:
        """Return chosen rank index (0..R-1) or -1 if none pass."""
        if Pblock.size == 0 or (not np.isfinite(Pblock).any()):
            return -1
        best_per_rank = np.nanmax(Pblock, axis=0)  # [R]
        ok = best_per_rank >= thr
        if not ok.any():
            return -1
        # deepest passing rank
        return int(np.where(ok)[0].max())

    def _fallback_deepest_finite_rank(Pblock: np.ndarray) -> int:
        """Deepest rank index with any finite support across candidates; -1 if none."""
        if Pblock.size == 0 or (not np.isfinite(Pblock).any()):
            return -1
        best_per_rank = np.nanmax(Pblock, axis=0)
        finite = np.isfinite(best_per_rank)
        return int(np.where(finite)[0].max()) if finite.any() else -1

    k = 0
    with alive_bar(len(g), title="Selecting best row", force_tty=True) as bar:
        for contig, idxs in g.items():
            idxs = np.asarray(idxs, dtype=np.int64)

            # optional: lineage restriction with safe NaN handling
            use_idxs = idxs
            if lineage_filter_col is not None and prefer_lineage:
                # df.index[use_idxs] converts positional -> label; loc expects labels
                lin_mask = (
                    df.loc[df.index[use_idxs], lineage_filter_col]
                    .fillna(False)
                    .astype(bool)
                    .to_numpy()
                )
                if lin_mask.any():
                    use_idxs = use_idxs[lin_mask]  # restrict
                # else: fallback to all idxs

            # drop rows that are "all NaN" (no finite probs at any rank)
            Pblock = P[use_idxs, :]  # [n, R]
            row_has_data = np.isfinite(Pblock).any(axis=1)
            if not row_has_data.any():
                # nothing usable for this contig; deterministic fallback to first original row
                out_pos[k] = idxs[0]
                k += 1
                bar()
                continue
            use_idxs = use_idxs[row_has_data]
            Pblock = Pblock[row_has_data]

            # choose rank by threshold rule; otherwise fallback to deepest finite rank
            r_ix = _choose_rank(Pblock)
            if r_ix < 0:
                r_ix = _fallback_deepest_finite_rank(Pblock)
                if r_ix < 0:
                    out_pos[k] = idxs[0]
                    k += 1
                    bar()
                    continue

            # choose row with highest prob at chosen rank
            p_at_r = Pblock[:, r_ix]
            maxp = np.nanmax(p_at_r)
            ties = np.isclose(p_at_r, maxp, rtol=1e-12, atol=1e-15) & np.isfinite(p_at_r)
            tie_idxs = use_idxs[np.where(ties)[0]]

            if len(tie_idxs) == 1 or tie_breaker == "p_only":
                out_pos[k] = tie_idxs[0]
                k += 1
                bar()
                continue

            # tie-breakers
            if tie_breaker in ("sum_to_rank", "deep_then_sum"):
                sum_to_r = cumsum[tie_idxs, r_ix]  # inclusive sum up to chosen rank
                best = np.nanmax(sum_to_r)
                ties2 = np.isclose(sum_to_r, best, rtol=1e-12, atol=1e-15)
                tie2_idxs = tie_idxs[np.where(ties2)[0]]

                if len(tie2_idxs) == 1 or tie_breaker == "sum_to_rank":
                    out_pos[k] = tie2_idxs[0]
                    k += 1
                    bar()
                    continue

                # deep_then_sum: if still tied, prefer more mass below the chosen rank
                if r_ix < R - 1:
                    mass_below = (cumsum[tie2_idxs, R - 1] - cumsum[tie2_idxs, r_ix])
                else:
                    mass_below = np.zeros(len(tie2_idxs), dtype=np.float64)

                best2 = np.nanmax(mass_below)
                ties3 = np.isclose(mass_below, best2, rtol=1e-12, atol=1e-15)
                tie3_idxs = tie2_idxs[np.where(ties3)[0]]

                out_pos[k] = tie3_idxs[0]
                k += 1
                bar()
                continue

            # FINAL deterministic fallback (must be inside loop)
            out_pos[k] = tie_idxs[0]
            k += 1
            bar()
            continue

    # out_pos are positional indices into df's row order
    selected = df.iloc[out_pos].copy()

    # annotate chosen rank and chosen prob (computed consistently with selection logic)
    per_contig_rank = {}
    with alive_bar(len(g), title="Computing chosen ranks", force_tty=True) as bar:
        for contig, idxs in g.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            use_idxs = idxs

            if lineage_filter_col is not None and prefer_lineage:
                lin_mask = (
                    df.loc[df.index[use_idxs], lineage_filter_col]
                    .fillna(False)
                    .astype(bool)
                    .to_numpy()
                )
                if lin_mask.any():
                    use_idxs = use_idxs[lin_mask]

            Pblock = P[use_idxs, :]
            row_has_data = np.isfinite(Pblock).any(axis=1)
            if row_has_data.any():
                use_idxs = use_idxs[row_has_data]
                Pblock = Pblock[row_has_data]
            else:
                per_contig_rank[contig] = -1
                bar()
                continue

            r_ix = _choose_rank(Pblock)
            if r_ix < 0:
                r_ix = _fallback_deepest_finite_rank(Pblock)
            per_contig_rank[contig] = int(r_ix)
            bar()

    contig_of_pos = df[sequence_col].to_numpy()
    chosen_rank_ix = np.empty(len(out_pos), dtype=np.int64)
    chosen_rank = np.empty(len(out_pos), dtype=object)
    chosen_prob = np.empty(len(out_pos), dtype=np.float64)

    for i, pos in enumerate(out_pos):
        contig = contig_of_pos[pos]
        r_ix = per_contig_rank.get(contig, -1)
        chosen_rank_ix[i] = r_ix
        if r_ix >= 0:
            chosen_rank[i] = ranks[r_ix]
            v = P[pos, r_ix]
            chosen_prob[i] = float(v) if np.isfinite(v) else np.nan
        else:
            chosen_rank[i] = None
            chosen_prob[i] = np.nan

    selected["chosen_rank_ix"] = chosen_rank_ix
    selected["chosen_rank"] = chosen_rank
    selected["chosen_prob_at_rank"] = chosen_prob

    return selected
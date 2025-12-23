from perseus.utils.constants import (
    RANK_INDEX,
    NUM_RANKS
)

def compute_cutoff_and_exclusion(truth_lineage_map, world, excl_sets):
    """
    Decide cutoff rank and exclusion metadata for a sample.

    Args:
        truth_lineage_map (dict): mapping rank -> taxid for the sample (e.g. produced by lineage_to_rank_map)
        world (str): one of "CORE", "S", "G", "F" (or other names you use)
        excl_sets (dict): {'species': set(...), 'genera': set(...), 'families': set(...)}

    Returns:
        tuple (cutoff_rank_name (str), cutoff_idx (int), is_excluded (bool), excluded_level (str))
    """
    # default: allow species
    if world == "CORE":
        return "species", RANK_INDEX.get("species", NUM_RANKS - 1), False, "none"

    # species-exclusion world: if species excluded -> backoff to genus
    if world == "S":
        sp = truth_lineage_map.get("species")
        if sp is not None and str(sp) in excl_sets.get("species", ()):
            return "genus", RANK_INDEX.get("genus", 0), True, "species"
        return "species", RANK_INDEX.get("species", NUM_RANKS - 1), False, "none"

    # genus-exclusion world: if genus excluded -> backoff to family
    if world == "G":
        gn = truth_lineage_map.get("genus")
        if gn is not None and str(gn) in excl_sets.get("genera", ()):
            return "family", RANK_INDEX.get("family", 0), True, "genus"
        return "species", RANK_INDEX.get("species", NUM_RANKS - 1), False, "none"

    # family-exclusion world: if family excluded -> backoff to order
    if world == "F":
        fam = truth_lineage_map.get("family")
        if fam is not None and str(fam) in excl_sets.get("families", ()):
            return "order", RANK_INDEX.get("order", 0), True, "family"
        return "species", RANK_INDEX.get("species", NUM_RANKS - 1), False, "none"

    # fallback
    return "species", RANK_INDEX.get("species", NUM_RANKS - 1), False, "none"


def build_targets_from_cutoff(cutoff_idx):
    """
    Build per-rank binary targets from cutoff index.

    Assumes RANK_LEVELS ordering is consistent across the codebase. If you
    use coarse->fine ordering (e.g. kingdom..species), this function returns
    1 for ranks coarser-or-equal-to cutoff (i <= cutoff_idx).
    If your CANONICAL_RANKS uses fine->coarse, invert the comparison accordingly.

    Returns:
        list[int] length == NUM_RANKS (0/1)
    """
    # Example uses coarse->fine ordering: target = 1 for indices <= cutoff_idx
    return [1 if i <= cutoff_idx else 0 for i in range(NUM_RANKS)]
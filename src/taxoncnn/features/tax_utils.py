import re
import logging
from functools import lru_cache
from ete3 import NCBITaxa

import taxoncnn.features.globals as globals
from taxoncnn.features.constants import (
    CANONICAL_RANKS,
    RANK_INDEX
)

logger = logging.getLogger(__name__)


def canonicalize_rank(rank):
    """
    Convert a rank string to its canonical form

    Args:
        rank (str): Taxonomic rank

    Returns:
        str: Canonicalized rank
    """
    if not rank:
        return None
    rank = rank.lower()
    if rank == 'kingdom':
        return 'superkingdom'
    if rank in CANONICAL_RANKS:
        return rank
    match = re.match(r'^(super|sub|infra|parv)?(domain|superkingdom|kingdom|phylum|class|order|family|genus|species|strain)$', rank)
    if match:
        canonical = match.group(2)
        return 'superkingdom' if canonical == 'kingdom' else canonical
    return None


def get_canonical_taxid_for_rank(taxid, canonical_ranks, ncbi):
    """
    Get the canonical taxid for each rank in canonical_ranks for a given taxid

    Args:
        taxid (int): Taxonomic ID
        canonical_ranks (list): List of canonical ranks
        ncbi: NCBI taxonomy database object

    Returns:
        dict: Mapping from rank to canonical taxid
    """
    try:
        lineage = ncbi.get_lineage(int(taxid))
        ranks = ncbi.get_rank(lineage)
        taxid_at_rank = {r: None for r in canonical_ranks}
        for t in lineage:
            raw_rank = ranks.get(t)
            can = canonicalize_rank(raw_rank)
            if can in canonical_ranks and taxid_at_rank[can] is None:
                taxid_at_rank[can] = t
        return taxid_at_rank
    except Exception:
        return {r: None for r in canonical_ranks}


def fetch_maps(tid):
    """
    Fetch lineage, canonical, and descendant maps for a given taxid

    Args:
        tid (int): Taxonomic ID

    Returns:
        tuple: (lineage_map, canonical_map, descendant_map)
    """
    ncbi = NCBITaxa()
    try:
        lineage = ncbi.get_lineage(int(tid))
        descendants = set(ncbi.get_descendant_taxa(int(tid)))
        canonical_taxid = get_canonical_taxid_for_rank(tid, CANONICAL_RANKS, ncbi)
        return tid, lineage, descendants, canonical_taxid
    except Exception:
        logger.warning(f"Taxid {tid} not found in NCBI database, skipping.")
        return tid, [], set(), {r: None for r in CANONICAL_RANKS}


@lru_cache(maxsize=200000)
def get_taxid_rank_raw(taxid: int):
    """
    Get the raw rank string for a given taxid

    Args:
        taxid (int): Taxonomic ID

    Returns:
        str: Raw rank string
    """
    try:
        return globals.NCBI.get_rank([int(taxid)]).get(int(taxid), None)
    except Exception:
        return None
    
    
def lineage_to_rank_map(lineage, canonical_ranks):
    """
    Map each canonical rank to the corresponding taxid in the lineage

    Args:
        lineage (list): List of taxids in the lineage
        canonical_ranks (list): List of canonical ranks

    Returns:
        dict: Mapping from rank to taxid
    """

    if not lineage:
        return {r: None for r in canonical_ranks}
    ranks = globals.NCBI.get_rank(lineage)
    out = {r: None for r in canonical_ranks}
    for t in lineage:
        can = canonicalize_rank(ranks.get(t))
        if can in canonical_ranks and out[can] is None:
            out[can] = t
    return out

def predicted_rank(taxid):
    """
    Predict the rank for a given taxid

    Args:
        taxid (int): Taxonomic ID

    Returns:
        str: Predicted rank
    """
    raw = get_taxid_rank_raw(taxid)
    can = canonicalize_rank(raw)
    if can in RANK_INDEX:
        return can, RANK_INDEX[can]
    return None, -1


@lru_cache(maxsize=100000)
def cached_get_rank(taxid):
    """
    Get the rank for a taxid, using a cache for efficiency

    Args:
        taxid (int): Taxonomic ID

    Returns:
        str: Rank string
    """
    return globals.NCBI.get_rank([taxid])

@lru_cache(maxsize=100000)
def get_lineage_path(taxid):
    """
    Get the full lineage path for a given taxid

    Args:
        taxid (int): Taxonomic ID

    Returns:
        list[int]: List of taxids in the lineage
    """
    try:
        return globals.NCBI.get_lineage(int(taxid))
    except Exception:
        return []

@lru_cache(maxsize=100000)
def get_taxid_to_rank(taxid):
    """
    Get a mapping from taxid to rank for a given taxid

    Args:
        taxid (int): Taxonomic ID

    Returns:
        dict: Mapping from taxid to rank
    """
    try:
        return globals.NCBI.get_rank([int(taxid)]).get(int(taxid), None)
    except Exception:
        return None

@lru_cache(maxsize=100000)
def get_descendants(taxid):
    """
    Get all descendant taxids for a given taxid

    Args:
        taxid (int): Taxonomic ID

    Returns:
        list: List of descendant taxids
    """
    try:
        return globals.NCBI.get_descendant_taxa(int(taxid), collapse_subspecies=False, intermediate_nodes=True)
    except Exception:
        return []

@lru_cache(maxsize=200000)
def normalize_taxid(tid):
    """
    Normalize a taxid to its canonical form

    Args:
        tid (int): Taxonomic ID

    Returns:
        int: Normalized taxid
    """
    try:
        tid = int(tid)
    except Exception:
        tid = int(tid.split()[-1].strip('()'))
    try:
        lin = globals.NCBI.get_lineage(tid)
        return int(lin[-1]) if lin else tid
    except Exception:
        return tid
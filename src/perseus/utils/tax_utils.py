import re
import logging
import json
from functools import lru_cache
from pathlib import Path
from ete4 import (
    NCBITaxa,
    GTDBTaxa
)

import perseus.utils.globals as globals
from perseus.utils.constants import (
    CANONICAL_RANKS,
    RANK_INDEX
)

logger = logging.getLogger(__name__)

def detect_taxonomy_from_dir(db_dir: str):
    p = Path(db_dir) / "taxonomy.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        tx = data.get("taxonomy")
        if tx:
            db_type = tx.lower()
            globals.DB_TYPE = db_type
            return db_type
    except Exception:
        logger.warning(f"Failed to read taxonomy type from {p}, defaulting to None.")
        return None
    logger.warning(f"Taxonomy type not found in {p}, defaulting to None.")
    return None


def get_ncbi(db_dir):
    db_dir = Path(db_dir).expanduser().resolve()
    sqlite_path = db_dir / "taxa.sqlite"

    if not sqlite_path.exists():
        logger.error("ETE4 NCBI taxonomy database not found at %s", sqlite_path)
        logger.error("Run `perseus setup --db-dir %s` first", db_dir)
        raise SystemExit(1)

    return NCBITaxa(dbfile=str(sqlite_path))


def get_gtdb(db_dir):
    db_dir = Path(db_dir).expanduser().resolve()
    sqlite_path = db_dir / "taxa.sqlite"

    if not sqlite_path.exists():
        logger.error("ETE4 GTDB taxonomy database not found at %s", sqlite_path)
        logger.error("Run `perseus setup --db-dir %s --taxonomy gtdb` first", db_dir)
        raise SystemExit(1)

    return GTDBTaxa(dbfile=str(sqlite_path))


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
        return 'superkingdom' if canonical == 'kingdom' or canonical == 'domain' else canonical
    return None


def get_canonical_taxid_for_rank(taxid, canonical_ranks, db):
    """
    Get the canonical taxid for each rank in canonical_ranks for a given taxid

    Args:
        taxid (int): Taxonomic ID
        canonical_ranks (list): List of canonical ranks
        db: ETE4 taxonomy database object

    Returns:
        dict: Mapping from rank to canonical taxid
    """
    try:
        if globals.DB_TYPE == "gtdb":
            lineage = db._get_lineage(int(taxid))
            ranks = db._get_id2rank(lineage)
        elif globals.DB_TYPE == "ncbi":
            lineage = db.get_lineage(int(taxid))
            ranks = db.get_rank(lineage)
        else:
            logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
            raise SystemExit(1)
        taxid_at_rank = {r: None for r in canonical_ranks}
        for t in lineage:
            raw_rank = ranks.get(t)
            can = canonicalize_rank(raw_rank)
            if can in canonical_ranks and taxid_at_rank[can] is None:
                taxid_at_rank[can] = t
        return taxid_at_rank
    except Exception:
        return {r: None for r in canonical_ranks}


def fetch_maps(args):
    """
    Fetch lineage, canonical, and descendant maps for a given taxid

    Args:
        tid (int): Taxonomic ID

    Returns:
        tuple: (lineage_map, canonical_map, descendant_map)
    """
    tid, db_dir = args
    if globals.DB_TYPE == "gtdb":
        db = get_gtdb(db_dir)
    elif globals.DB_TYPE == "ncbi":
        db = get_ncbi(db_dir)
    else:
        logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
        raise SystemExit(1)

    try:
        if globals.DB_TYPE == "gtdb":
            lineage = db._get_lineage(int(tid))
        elif globals.DB_TYPE == "ncbi":
            lineage = db.get_lineage(int(tid))
        else:
            logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
            raise SystemExit(1)
        descendants = set(db.get_descendant_taxa(int(tid)))
        canonical_taxid = get_canonical_taxid_for_rank(tid, CANONICAL_RANKS, db)
        return tid, lineage, descendants, canonical_taxid
    except Exception:
        logger.warning(f"Taxid {tid} not found in ETE4 database, skipping.")
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
        if globals.DB_TYPE == "gtdb":
            return globals.DB._get_id2rank([int(taxid)]).get(int(taxid), None)
        elif globals.DB_TYPE == "ncbi":
            return globals.DB.get_rank([int(taxid)]).get(int(taxid), None)
        else:
            logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
            raise SystemExit(1)
    except Exception:
        logger.warning(f"Taxid {taxid} not found in NCBI database, returning None for raw rank.")
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
    if globals.DB_TYPE == "gtdb":
        ranks = globals.DB._get_id2rank(lineage)
    elif globals.DB_TYPE == "ncbi":
        ranks = globals.DB.get_rank(lineage)
    else:
        logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
        raise SystemExit(1)
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
    try:
        if globals.DB_TYPE == "gtdb":
            return globals.DB._get_id2rank([taxid])
        elif globals.DB_TYPE == "ncbi":
            return globals.DB.get_rank([taxid])
        else:
            logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
            raise SystemExit(1)
    except Exception:
        logger.warning(f"Taxid {taxid} not found in ETE4 database, returning empty rank mapping.")
        return {}

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
        if globals.DB_TYPE == "gtdb":
            lineage = globals.DB._get_lineage(int(taxid))
        elif globals.DB_TYPE == "ncbi":
            lineage = globals.DB.get_lineage(int(taxid))
        else:
            logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
            raise SystemExit(1)
        return lineage
    except Exception:
        logger.warning(f"Taxid {taxid} not found in ETE4 database, returning empty lineage.")
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
        if globals.DB_TYPE == "gtdb":
            return globals.DB._get_id2rank([int(taxid)]).get(int(taxid), None)
        elif globals.DB_TYPE == "ncbi":
            return globals.DB.get_rank([int(taxid)]).get(int(taxid), None)
        else:
            logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
            raise SystemExit(1)
    except Exception:
        logger.warning(f"Taxid {taxid} not found in ETE4 database, returning None for rank.")
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
        return globals.DB.get_descendant_taxa(int(taxid), collapse_subspecies=False, intermediate_nodes=True)
    except Exception:
        logger.warning(f"Taxid {taxid} not found in ETE4 database, returning empty descendant list.")
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
        if globals.DB_TYPE == "gtdb":
            lin = globals.DB._get_lineage(tid)
        elif globals.DB_TYPE == "ncbi":
            lin = globals.DB.get_lineage(tid)
        else:
            logger.error("Unsupported taxonomy database type: %s", globals.DB_TYPE)
            raise SystemExit(1)
        return int(lin[-1]) if lin else tid
    except Exception:
        logger.warning(f"Taxid {tid} not found in ETE4 database, returning original taxid.")
        return tid
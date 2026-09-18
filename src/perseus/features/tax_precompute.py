import perseus.utils.globals as globals

from perseus.utils.constants import CANONICAL_RANKS
from perseus.utils.tax_utils import (
    canonicalize_rank,
    get_canonical_taxid_for_rank,
)

from perseus.features.init import (
    _init_ncbi_private_db,
)


def init_taxonomy_map_worker(
    db_path,
):
    """
    Initialize a private ETE/NCBI database for a multiprocessing worker.
    """
    _init_ncbi_private_db(
        db_path
    )


def fetch_feature_maps(
    tid,
):
    """
    Precompute only taxonomy metadata required by Perseus
    feature extraction.

    Returns
    -------
    tuple
        (
            taxid,
            lineage,
            canonical_map,
            rank_idx,
        )
    """
    tid = int(tid)

    ncbi = globals.NCBI

    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    try:
        lineage = tuple(
            ncbi.get_lineage(
                tid
            )
        )

    except Exception:
        lineage = ()

    # --------------------------------------------------------
    # Canonical ancestor mapping
    # --------------------------------------------------------

    try:
        canonicals = (
            get_canonical_taxid_for_rank(
                tid,
                CANONICAL_RANKS,
                ncbi,
            )
        )

    except Exception:

        canonicals = {
            rank: None
            for rank in CANONICAL_RANKS
        }

    # --------------------------------------------------------
    # Taxid's own canonical rank
    # --------------------------------------------------------

    try:

        raw_rank = (
            ncbi.get_rank(
                [tid]
            ).get(
                tid
            )
        )

        canonical_rank = (
            canonicalize_rank(
                raw_rank
            )
        )

        try:
            rank_idx = (
                CANONICAL_RANKS.index(
                    canonical_rank
                )
            )

        except ValueError:
            rank_idx = -1

    except Exception:
        rank_idx = -1

    return (
        tid,
        lineage,
        canonicals,
        rank_idx,
    )
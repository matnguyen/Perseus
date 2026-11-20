
from pathlib import Path

import pytest
from ete3 import NCBITaxa

from perseus.commands.setup import setup_ete3
from perseus.utils.constants import CANONICAL_RANKS
from perseus.utils.tax_utils import (
    canonicalize_rank,
    get_canonical_taxid_for_rank,
)


@pytest.fixture
def gtdb_taxonomy(tmp_path):
    db = tmp_path / "kraken_gtdb"
    taxonomy = db / "taxonomy"
    taxonomy.mkdir(parents=True)

    taxa = [
        (1, 1, "no rank", "root"),
        (2, 1, "domain", "Bacteria"),
        (3, 2, "phylum", "Pseudomonadota"),
        (4, 3, "class", "Gammaproteobacteria"),
        (5, 4, "order", "Enterobacterales"),
        (6, 5, "family", "Enterobacteriaceae"),
        (7, 6, "genus", "Escherichia"),
        (8, 7, "species", "Escherichia coli"),
    ]

    with (taxonomy / "names.dmp").open("w") as names:
        for tid, _, _, name in taxa:
            names.write(
                f"{tid}\t|\t{name}\t|\t\t|\tscientific name\t|\n"
            )

    with (taxonomy / "nodes.dmp").open("w") as nodes:
        for tid, parent, rank, _ in taxa:
            nodes.write(
                f"{tid}\t|\t{parent}\t|\t{rank}\t|\n"
            )

    return db


@pytest.mark.parametrize(
    ("raw_rank", "expected"),
    [
        ("domain", "superkingdom"),
        ("kingdom", "superkingdom"),
        ("superkingdom", "superkingdom"),
        ("phylum", "phylum"),
        ("class", "class"),
        ("order", "order"),
        ("family", "family"),
        ("genus", "genus"),
        ("species", "species"),
        ("subspecies", "species"),
        ("subgenus", "genus"),
        ("superfamily", "family"),
        ("suborder", "order"),
        ("subclass", "class"),
        ("subphylum", "phylum"),
        ("strain", "strain"),
        ("no rank", None),
        ("", None),
        (None, None),
    ],
)
def test_gtdb_domain_mapping(raw_rank, expected):
    assert canonicalize_rank(raw_rank) == expected


def test_gtdb_taxonomy_setup(tmp_path, gtdb_taxonomy):
    ete_dir = tmp_path / "ete3_gtdb"

    setup_ete3(
        str(ete_dir),
        update=False,
        kraken_db=str(gtdb_taxonomy),
    )

    assert (ete_dir / "taxa.sqlite").is_file()

    taxonomy = NCBITaxa(
        dbfile=str(ete_dir / "taxa.sqlite")
    )

    lineage = taxonomy.get_lineage(8)
    assert lineage[-1] == 8

    ancestors = get_canonical_taxid_for_rank(
        8,
        CANONICAL_RANKS,
        taxonomy,
    )

    assert ancestors == {
        "superkingdom": 2,
        "phylum": 3,
        "class": 4,
        "order": 5,
        "family": 6,
        "genus": 7,
        "species": 8,
    }

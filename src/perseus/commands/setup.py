import argparse
import logging
from pathlib import Path
from ete4 import NCBITaxa
from ete4.ncbi_taxonomy.ncbiquery import update_local_taxdump

LOG = logging.getLogger(__name__)

def setup_ete4(path: str, update: bool) -> None:
    out_dir = Path(path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = out_dir / "taxa.sqlite"
    taxdump_path = out_dir / "taxdump.tar.gz"
    
    if not taxdump_path.exists():
        LOG.info("Downloading NCBI taxdump to %s", taxdump_path)
        update_local_taxdump(str(taxdump_path))

    if sqlite_path.exists():
        LOG.info("ETE4 taxonomic database already exists at %s", sqlite_path)
        if update:
            LOG.info("Forcing update of ETE4 taxonomic database")
        else:
            return
    else:
        LOG.info("Setting up ETE4 taxonomic database at %s", sqlite_path)

    ncbi = NCBITaxa(dbfile=str(sqlite_path), taxdump_file=str(taxdump_path))
    ncbi.update_taxonomy_database(taxdump_file=str(taxdump_path))

    LOG.info("ETE4 taxonomic database setup complete")

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Setup ETE4 taxonomic database for Perseus"
    )
    parser.add_argument(
        'db_dir',
        type=str,
        help="Directory where the ETE4 taxonomy database will be stored",
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help="Force update of the ETE4 taxonomy database even if it already exists",
    )

    args = parser.parse_args()
    setup_ete4(args.db_dir, args.update)

if __name__ == "__main__":
    main()
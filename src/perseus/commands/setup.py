import argparse
import logging
from pathlib import Path
from ete3 import NCBITaxa

LOG = logging.getLogger(__name__)

def setup_ete3(path: str, update: bool) -> None:
    out_dir = Path(path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = out_dir / "taxa.sqlite"

    if sqlite_path.exists():
        LOG.info("ETE3 taxonomic database already exists at %s", sqlite_path)
        if update:
            LOG.info("Forcing update of ETE3 taxonomic database")
        else:
            return
    else:
        LOG.info("Setting up ETE3 taxonomic database at %s", sqlite_path)

    ncbi = NCBITaxa(dbfile=str(sqlite_path))
    ncbi.update_taxonomy_database()
    
    # Remove taxdump.tar.gz
    taxdump_path = Path.cwd() / "taxdump.tar.gz"
    if taxdump_path.exists():
        taxdump_path.unlink()
        LOG.debug("Removed temporary file %s", taxdump_path)

    LOG.info("ETE3 taxonomic database setup complete")

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Setup ETE3 taxonomic database for Perseus"
    )
    parser.add_argument(
        'db_dir',
        type=str,
        help="Directory where the ETE3 taxonomy database will be stored",
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help="Force update of the ETE3 taxonomy database even if it already exists",
    )

    args = parser.parse_args()
    setup_ete3(args.db_dir, args.update)

if __name__ == "__main__":
    main()
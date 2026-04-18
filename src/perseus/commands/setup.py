import argparse
import logging
import json
from pathlib import Path

from ete4.ncbi_taxonomy.ncbiquery import update_local_taxdump as update_ncbi_taxdump
from ete4.gtdb_taxonomy.gtdbquery import update_local_taxdump as update_gtdb_taxdump
from ete4 import (
    NCBITaxa,
    GTDBTaxa
)

LOG = logging.getLogger(__name__)

def setup_ete4(taxonomy: str, path: str, update: bool) -> None:
    out_dir = Path(path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = out_dir / "taxa.sqlite"
    taxdump_path = out_dir / "taxdump.tar.gz"
    
    if taxonomy.lower() == 'ncbi':
        if not taxdump_path.exists():
            LOG.info("Downloading NCBI taxdump to %s", taxdump_path)
            update_ncbi_taxdump(str(taxdump_path))

        if sqlite_path.exists():
            LOG.info("ETE4 NCBI taxonomic database already exists at %s", sqlite_path)
            if update:
                LOG.info("Forcing update of ETE4 NCBI taxonomic database")
                update_ncbi_taxdump(str(taxdump_path))
            else:
                return
        else:
            LOG.info("Setting up NCBI ETE4 taxonomic database at %s", sqlite_path)

        ncbi = NCBITaxa(dbfile=str(sqlite_path), taxdump_file=str(taxdump_path))
        LOG.info("ETE4 NCBI taxonomic database setup complete")
        
    else:
        if not taxdump_path.exists():
            LOG.info("Downloading GTDB taxonomy data to %s", taxdump_path)
            update_gtdb_taxdump(str(taxdump_path))
            
        if sqlite_path.exists():
            LOG.info("ETE4 GTDB taxonomic database already exists at %s", sqlite_path)
            if update:
                LOG.info("Forcing update of ETE4 GTDB taxonomic database")
                update_gtdb_taxdump(str(taxdump_path))
            else:
                return
        else:
            LOG.info("Setting up GTDB ETE4 taxonomic database at %s", sqlite_path)
        
        gtdb = GTDBTaxa(dbfile=str(sqlite_path), taxdump_file=str(taxdump_path))
        LOG.info("ETE4 GTDB taxonomic database setup complete")
        
    meta = {"taxonomy": taxonomy.lower()}
    meta_path = out_dir / "taxonomy.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    LOG.info("Wrote taxonomy metadata to %s", meta_path)

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
    parser.add_argument(
        '--taxonomy',
        type=str,
        default='ncbi',
        help="Taxonomy source to use: 'ncbi' or 'gtdb' (default: 'ncbi')",
    )

    args = parser.parse_args()
    
    if args.taxonomy.lower() not in ['ncbi', 'gtdb']:
        LOG.error("Invalid taxonomy source: %s. Must be 'ncbi' or 'gtdb'.", args.taxonomy)
        return
    
    setup_ete4(args.taxonomy, args.db_dir, args.update)

if __name__ == "__main__":
    main()
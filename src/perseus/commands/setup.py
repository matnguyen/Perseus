import argparse
import logging
import tarfile
import tempfile
from pathlib import Path
from ete3 import NCBITaxa

LOG = logging.getLogger(__name__)


def build_custom_taxdump(
    names_path: Path,
    nodes_path: Path,
    output_path: Path,
    merged_path: Path | None = None
) -> None:
    """Create the minimal taxdump archive required by ETE3"""

    if not names_path.is_file():
        raise FileNotFoundError(f"names.dmp not found: {names_path}")
    if not nodes_path.is_file():
        raise FileNotFoundError(f"nodes.dmp not found: {nodes_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        if merged_path is None or not merged_path.is_file():
            merged_path = Path(tmpdir) / "merged.dmp"
            merged_path.touch()
            LOG.info("No merged.dmp found; creating an empty merged.dmp at %s", merged_path)
        else:
            LOG.info("Using existing merged.dmp at %s", merged_path)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(names_path, arcname="names.dmp")
            tar.add(nodes_path, arcname="nodes.dmp")
            tar.add(merged_path, arcname="merged.dmp")
        LOG.info("Created custom taxdump archive at %s", output_path)


def setup_ete3(
    path: str, 
    update: bool,
    kraken_db: str | None = None
) -> None:
    out_dir = Path(path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = out_dir / "taxa.sqlite"

    if sqlite_path.exists():
        LOG.info("ETE3 taxonomic database already exists at %s", sqlite_path)
        if update:
            LOG.info("Forcing update of ETE3 taxonomic database")
        else:
            return

    if kraken_db is None:
        LOG.info("Setting up ETE3 database using NCBI taxonomy")

        ncbi = NCBITaxa(dbfile=str(sqlite_path))
        ncbi.update_taxonomy_database()

    else:
        # Custom Kraken2 taxonomy
        kraken_db = Path(kraken_db).expanduser().resolve()

        taxonomy_dir = kraken_db / "taxonomy"

        names_path = taxonomy_dir / "names.dmp"
        nodes_path = taxonomy_dir / "nodes.dmp"
        merged_path = taxonomy_dir / "merged.dmp"

        LOG.info("Building ETE3 database from Kraken2 taxonomy at %s", taxonomy_dir)

        with tempfile.TemporaryDirectory() as tmp:
            taxdump_path = Path(tmp) / "taxdump.tar.gz"

            build_custom_taxdump(
                names_path,
                nodes_path,
                taxdump_path,
                merged_path,
            )

            NCBITaxa(
                dbfile=str(sqlite_path),
                taxdump_file=str(taxdump_path),
            )

    LOG.info("ETE3 taxonomic database setup complete: %s", sqlite_path)

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
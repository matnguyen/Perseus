
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from perseus.commands import setup


@pytest.fixture
def kraken_db(tmp_path):
    """Create a minimal Kraken2 taxonomy directory."""
    taxonomy = tmp_path / "kraken_db" / "taxonomy"
    taxonomy.mkdir(parents=True)

    (taxonomy / "names.dmp").write_text(
        "1\t|\troot\t|\t\t|\tscientific name\t|\n"
        "2\t|\tBacteria\t|\t\t|\tscientific name\t|\n"
    )
    (taxonomy / "nodes.dmp").write_text(
        "1\t|\t1\t|\tno rank\t|\n"
        "2\t|\t1\t|\tsuperkingdom\t|\n"
    )

    return taxonomy.parent


@pytest.mark.parametrize(
    "include_merged",
    [False, True],
)
def test_build_custom_taxdump(tmp_path, kraken_db, include_merged):
    taxonomy = kraken_db / "taxonomy"
    merged = taxonomy / "merged.dmp"
    output = tmp_path / "taxdump.tar.gz"

    if include_merged:
        merged.write_text("10\t|\t2\t|\n")

    setup.build_custom_taxdump(
        names_path=taxonomy / "names.dmp",
        nodes_path=taxonomy / "nodes.dmp",
        merged_path=merged,
        output_path=output,
    )

    assert output.is_file()

    with tarfile.open(output, "r:gz") as tar:
        assert set(tar.getnames()) == {
            "names.dmp",
            "nodes.dmp",
            "merged.dmp",
        }

        content = tar.extractfile("merged.dmp").read()

        if include_merged:
            assert content == b"10\t|\t2\t|\n"
        else:
            assert content == b""


@pytest.mark.parametrize(
    "missing_file",
    ["names.dmp", "nodes.dmp"],
)
def test_build_custom_taxdump_missing_file(
    tmp_path,
    kraken_db,
    missing_file,
):
    taxonomy = kraken_db / "taxonomy"
    (taxonomy / missing_file).unlink()

    with pytest.raises(FileNotFoundError):
        setup.build_custom_taxdump(
            names_path=taxonomy / "names.dmp",
            nodes_path=taxonomy / "nodes.dmp",
            merged_path=taxonomy / "merged.dmp",
            output_path=tmp_path / "taxdump.tar.gz",
        )


def test_setup_existing_database_skips(tmp_path, monkeypatch):
    db_dir = tmp_path / "ete3_db"
    db_dir.mkdir()
    (db_dir / "taxa.sqlite").touch()

    mock_ncbi = MagicMock()
    monkeypatch.setattr(setup, "NCBITaxa", mock_ncbi)

    setup.setup_ete3(str(db_dir), update=False)

    mock_ncbi.assert_not_called()


def test_setup_custom_kraken_database(
    tmp_path,
    kraken_db,
    monkeypatch,
):
    db_dir = tmp_path / "ete3_db"
    calls = []

    def fake_ncbi(*args, **kwargs):
        archive = Path(kwargs["taxdump_file"])

        # Inspect while the temporary directory still exists.
        assert archive.is_file()

        with tarfile.open(archive, "r:gz") as tar:
            assert set(tar.getnames()) == {
                "names.dmp",
                "nodes.dmp",
                "merged.dmp",
            }

        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(setup, "NCBITaxa", fake_ncbi)

    setup.setup_ete3(
        str(db_dir),
        update=False,
        kraken_db=str(kraken_db),
    )

    assert len(calls) == 1
    assert calls[0]["dbfile"] == str(db_dir / "taxa.sqlite")


def test_setup_default_ncbi(tmp_path, monkeypatch):
    mock_ncbi = MagicMock()
    monkeypatch.setattr(setup, "NCBITaxa", mock_ncbi)

    db_dir = tmp_path / "ete3_db"

    setup.setup_ete3(
        str(db_dir),
        update=False,
    )

    mock_ncbi.assert_called_once_with(
        dbfile=str(db_dir / "taxa.sqlite")
    )
    mock_ncbi.return_value.update_taxonomy_database.assert_called_once_with()

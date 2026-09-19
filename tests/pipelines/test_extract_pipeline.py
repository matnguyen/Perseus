import importlib
import pytest

MODULE = "perseus.commands.extract"


@pytest.mark.pipeline
def test_collect_unique_taxids_small(tmp_path):
    m = importlib.import_module(MODULE)

    # ----------------------------
    # Create a tiny fake Kraken TSV
    # ----------------------------
    kraken_path = tmp_path / "small.tsv"
    kraken_path.write_text(
        "C\tseq1\t(60)\t1000\t60:5 61:3\n"
        "C\tseq2\t(61)\t900\t60:2 10:1\n"
        "U\tseq3\t(50)\t500\t\n"
    )

    # ----------------------------
    # Run the function
    # ----------------------------
    taxids = m.collect_unique_taxids(str(kraken_path))

    # ----------------------------
    # Validate
    # ----------------------------
    assert taxids == {10, 60, 61}
    
@pytest.mark.pipeline
def test_collect_unique_taxids_deduplicates(tmp_path):
    m = importlib.import_module(MODULE)

    kraken_path = tmp_path / "small.tsv"
    kraken_path.write_text(
        "C\tseq1\t(60)\t1000\t60:5 61:3\n"
        "C\tseq2\t(60)\t900\t60:20 61:1\n"
    )

    taxids = m.collect_unique_taxids(str(kraken_path))

    assert taxids == {60, 61}


@pytest.mark.pipeline
def test_collect_unique_taxids_ignores_invalid_input(tmp_path):
    m = importlib.import_module(MODULE)

    kraken_path = tmp_path / "small.tsv"
    kraken_path.write_text(
        "C\tseq1\t(60)\t1000\t60:5 garbage 61:3\n"
        "bad\tline\n"
        "U\tseq3\t(50)\t500\t\n"
    )

    taxids = m.collect_unique_taxids(str(kraken_path))

    assert taxids == {60, 61}
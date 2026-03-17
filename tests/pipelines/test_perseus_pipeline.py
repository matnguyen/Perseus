# tests/pipelines/test_extract_filter_pipeline.py

from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest
import torch

def run_cmd(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"command failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    return res

@pytest.mark.pipeline
def test_extract_then_filter_end_to_end_auto(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    kraken_file = repo_root / "tests" / "test_data" / "test_kraken.txt"
    expected_filtered = repo_root / "tests" / "test_data" / "filtered.txt"

    extract_out = tmp_path / "extracted_output"
    filter_out = tmp_path / "filtered_output.txt"
    
    db_dir = tmp_path / "db"
    
    # Run setup
    setup_cmd = [
        sys.executable, "-m", "perseus.cli",
        "setup",
        str(db_dir),
    ]
    setup_res = run_cmd(setup_cmd)
    
    assert db_dir.exists(), f"Database directory {db_dir} was not created"
    assert any(db_dir.iterdir()), f"Database directory {db_dir} is empty after setup"
    assert (db_dir / "taxa.sqlite").exists(), "Expected taxa.sqlite not found in database directory"

    # Run extract
    extract_cmd = [
        sys.executable, "-m", "perseus.cli",
        "extract",
        str(kraken_file),
        str(extract_out),
        str(db_dir)
    ]

    extract_res = run_cmd(extract_cmd)

    # Verify extract output exists
    assert extract_out.exists(), f"Extract output directory {extract_out} was not created"

    entries = list(extract_out.iterdir())

    manifests = [p for p in entries if "manifest" in p.name and p.suffix == ".json"]
    assert manifests, f"No manifest found in {extract_out}"

    shard_files = [p for p in entries if p.suffix == ".pt"]
    assert shard_files, f"No shard files found in {extract_out}"
    
    # Check if shard file matches the one in test_data
    expected_shard = repo_root / "tests" / "test_data" / "test_shards" / "part-p4040034-000001.pt"

    assert len(shard_files) == 1, f"Expected 1 shard file, found {len(shard_files)}"

    expected_data = torch.load(expected_shard, map_location="cpu")
    observed_data = torch.load(shard_files[0], map_location="cpu")

    # Compare structure
    assert expected_data.keys() == observed_data.keys(), "Shard keys differ"
    
    for key in expected_data:
        e = expected_data[key]
        o = observed_data[key]

        if isinstance(e, torch.Tensor):
            assert e.shape == o.shape, f"Shape mismatch for {key}: {e.shape} != {o.shape}"
            assert e.dtype == o.dtype, f"Dtype mismatch for {key}: {e.dtype} != {o.dtype}"
        else:
            assert type(e) == type(o), f"Type mismatch for {key}"

    # Run filter
    filter_cmd = [
        sys.executable, "-m", "perseus.cli",
        "filter",
        str(extract_out),
        str(kraken_file),
        str(filter_out),
        str(db_dir),
    ]

    filter_res = run_cmd(filter_cmd)

    assert filter_out.exists(), f"Filter output file {filter_out} was not created"

    # Compare to ground truth output
    got = pd.read_csv(filter_out, sep="\t")
    exp = pd.read_csv(expected_filtered, sep="\t")

    # Normalize row order / column order
    got = got.sort_values(list(got.columns)).reset_index(drop=True)
    exp = exp.sort_values(list(exp.columns)).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        got,
        exp,
        check_dtype=False,
        check_exact=False,
        atol=1e-5,
        rtol=1e-4
    )

@pytest.mark.pipeline
def test_extract_then_filter_end_to_end_multithreaded(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    kraken_file = repo_root / "tests" / "test_data" / "test_kraken.txt"
    expected_filtered = repo_root / "tests" / "test_data" / "filtered.txt"

    extract_out = tmp_path / "extracted_output"
    filter_out = tmp_path / "filtered_output.txt"
    
    db_dir = tmp_path / "db"
    
    # Run setup
    setup_cmd = [
        sys.executable, "-m", "perseus.cli",
        "setup",
        str(db_dir),
    ]
    setup_res = run_cmd(setup_cmd)
    
    assert db_dir.exists(), f"Database directory {db_dir} was not created"
    assert any(db_dir.iterdir()), f"Database directory {db_dir} is empty after setup"
    assert (db_dir / "taxa.sqlite").exists(), "Expected taxa.sqlite not found in database directory"

    # Run extract
    extract_cmd = [
        sys.executable, "-m", "perseus.cli",
        "extract",
        "--threads", "4",
        str(kraken_file),
        str(extract_out),
        str(db_dir)
    ]

    extract_res = run_cmd(extract_cmd)

    # Verify extract output exists
    assert extract_out.exists(), f"Extract output directory {extract_out} was not created"

    entries = list(extract_out.iterdir())

    manifests = [p for p in entries if "manifest" in p.name and p.suffix == ".json"]
    assert manifests, f"No manifest found in {extract_out}"

    shard_files = [p for p in entries if p.suffix == ".pt"]
    assert shard_files, f"No shard files found in {extract_out}"
    
    # Check if shard file matches the one in test_data
    expected_shard = repo_root / "tests" / "test_data" / "test_shards" / "part-p4040034-000001.pt"

    assert len(shard_files) == 1, f"Expected 1 shard file, found {len(shard_files)}"

    expected_data = torch.load(expected_shard, map_location="cpu")
    observed_data = torch.load(shard_files[0], map_location="cpu")

    # Compare structure
    assert expected_data.keys() == observed_data.keys(), "Shard keys differ"
    
    for key in expected_data:
        e = expected_data[key]
        o = observed_data[key]

        if isinstance(e, torch.Tensor):
            assert e.shape == o.shape, f"Shape mismatch for {key}: {e.shape} != {o.shape}"
            assert e.dtype == o.dtype, f"Dtype mismatch for {key}: {e.dtype} != {o.dtype}"
        else:
            assert type(e) == type(o), f"Type mismatch for {key}"

    # Run filter
    filter_cmd = [
        sys.executable, "-m", "perseus.cli",
        "filter",
        str(extract_out),
        str(kraken_file),
        str(filter_out),
        str(db_dir),
    ]

    filter_res = run_cmd(filter_cmd)

    assert filter_out.exists(), f"Filter output file {filter_out} was not created"

    # Compare to ground truth output
    got = pd.read_csv(filter_out, sep="\t")
    exp = pd.read_csv(expected_filtered, sep="\t")

    # Normalize row order / column order
    got = got.sort_values(list(got.columns)).reset_index(drop=True)
    exp = exp.sort_values(list(exp.columns)).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        got,
        exp,
        check_dtype=False,
        check_exact=False,
        atol=1e-5,
        rtol=1e-4
    )

@pytest.mark.pipeline
def test_extract_then_filter_end_to_end_single_threaded(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    kraken_file = repo_root / "tests" / "test_data" / "test_kraken.txt"
    expected_filtered = repo_root / "tests" / "test_data" / "filtered.txt"

    extract_out = tmp_path / "extracted_output"
    filter_out = tmp_path / "filtered_output.txt"
    
    db_dir = tmp_path / "db"
    
    # Run setup
    setup_cmd = [
        sys.executable, "-m", "perseus.cli",
        "setup",
        str(db_dir),
    ]
    setup_res = run_cmd(setup_cmd)
    
    assert db_dir.exists(), f"Database directory {db_dir} was not created"
    assert any(db_dir.iterdir()), f"Database directory {db_dir} is empty after setup"
    assert (db_dir / "taxa.sqlite").exists(), "Expected taxa.sqlite not found in database directory"

    # Run extract
    extract_cmd = [
        sys.executable, "-m", "perseus.cli",
        "extract",
        "--threads", "1",
        str(kraken_file),
        str(extract_out),
        str(db_dir)
    ]

    extract_res = run_cmd(extract_cmd)

    # Verify extract output exists
    assert extract_out.exists(), f"Extract output directory {extract_out} was not created"

    entries = list(extract_out.iterdir())

    manifests = [p for p in entries if "manifest" in p.name and p.suffix == ".json"]
    assert manifests, f"No manifest found in {extract_out}"

    shard_files = [p for p in entries if p.suffix == ".pt"]
    assert shard_files, f"No shard files found in {extract_out}"
    
    # Check if shard file matches the one in test_data
    expected_shard = repo_root / "tests" / "test_data" / "test_shards" / "part-p4040034-000001.pt"

    assert len(shard_files) == 1, f"Expected 1 shard file, found {len(shard_files)}"

    expected_data = torch.load(expected_shard, map_location="cpu")
    observed_data = torch.load(shard_files[0], map_location="cpu")

    # Compare structure
    assert expected_data.keys() == observed_data.keys(), "Shard keys differ"
    
    for key in expected_data:
        e = expected_data[key]
        o = observed_data[key]

        if isinstance(e, torch.Tensor):
            assert e.shape == o.shape, f"Shape mismatch for {key}: {e.shape} != {o.shape}"
            assert e.dtype == o.dtype, f"Dtype mismatch for {key}: {e.dtype} != {o.dtype}"
        else:
            assert type(e) == type(o), f"Type mismatch for {key}"

    # Run filter
    filter_cmd = [
        sys.executable, "-m", "perseus.cli",
        "filter",
        str(extract_out),
        str(kraken_file),
        str(filter_out),
        str(db_dir),
    ]

    filter_res = run_cmd(filter_cmd)

    assert filter_out.exists(), f"Filter output file {filter_out} was not created"

    # Compare to ground truth output
    got = pd.read_csv(filter_out, sep="\t")
    exp = pd.read_csv(expected_filtered, sep="\t")

    # Normalize row order / column order
    got = got.sort_values(list(got.columns)).reset_index(drop=True)
    exp = exp.sort_values(list(exp.columns)).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        got,
        exp,
        check_dtype=False,
        check_exact=False,
        atol=1e-5,
        rtol=1e-4
    )
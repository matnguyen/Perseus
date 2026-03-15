# tests/pipelines/test_extract_filter_pipeline.py

from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest
import torch

@pytest.mark.pipeline
def test_extract_then_filter_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    kraken_file = repo_root / "tests" / "test_data" / "test_kraken.txt"
    expected_filtered = repo_root / "tests" / "test_data" / "filtered.txt"

    extract_out = tmp_path / "extracted_output"
    filter_out = tmp_path / "filtered_output.txt"

    # Run extract
    extract_cmd = [
        sys.executable, "-m", "perseus.cli",
        "extract",
        str(kraken_file),
        str(extract_out),
    ]

    extract_res = subprocess.run(extract_cmd, capture_output=True, text=True)

    assert extract_res.returncode == 0, (
        f"extract failed\nSTDOUT:\n{extract_res.stdout}\nSTDERR:\n{extract_res.stderr}"
    )

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

            if e.dtype.is_floating_point:
                diff = (e - o).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                num_bad = (~torch.isclose(e, o, atol=1e-6, rtol=1e-5)).sum().item()
                assert torch.allclose(e, o, atol=1e-6, rtol=1e-5), (
                    f"Tensor values differ for {key}; "
                    f"max_diff={max_diff:.10g}, mean_diff={mean_diff:.10g}, num_bad={num_bad}"
                )
            else:
                assert torch.equal(e, o), f"Tensor values differ for {key}"

        else:
            assert e == o, f"Value mismatch for {key}: {e} != {o}"

    # Run filter
    filter_cmd = [
        sys.executable, "-m", "perseus.cli",
        "filter",
        str(extract_out),
        str(kraken_file),
        str(filter_out),
    ]

    filter_res = subprocess.run(filter_cmd, capture_output=True, text=True)

    assert filter_res.returncode == 0, (
        f"filter failed\nSTDOUT:\n{filter_res.stdout}\nSTDERR:\n{filter_res.stderr}"
    )

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
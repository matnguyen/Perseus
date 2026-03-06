# tests/pipelines/test_extract_filter_pipeline.py

from pathlib import Path
import subprocess
import sys
import pandas as pd
import os


def test_extract_then_filter_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    kraken_file = os.path.join(repo_root, "tests", "test_data", "test_kraken.txt")
    expected_filtered = os.path.join(repo_root, "tests", "test_data", "filtered.txt")

    extract_out = os.path.join(tmp_path, "extracted_output")
    filter_out = os.path.join(tmp_path, "filtered_output.txt")

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
    entries = []
    extract_out_str = str(extract_out)
    assert os.path.exists(extract_out_str), f"Extract output directory {extract_out} was not created"
    if os.path.exists(extract_out_str):
        entries = os.listdir(extract_out_str)

    manifests = [os.path.join(extract_out_str, f) for f in entries if "manifest" in f and f.endswith(".json")]
    assert manifests, f"No manifest found in {extract_out}"

    shard_files = [os.path.join(extract_out_str, f) for f in entries if f.endswith(".pt")]
    assert shard_files, f"No shard files found in {extract_out}"

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

    assert os.path.exists(filter_out), f"Filter output file {filter_out} was not created"

    # Compare to ground truth output
    got = pd.read_csv(filter_out, sep="\t")
    exp = pd.read_csv(expected_filtered, sep="\t")

    # Normalize row order / column order
    got = got.sort_values(list(got.columns)).reset_index(drop=True)
    exp = exp.sort_values(list(exp.columns)).reset_index(drop=True)

    pd.testing.assert_frame_equal(got, exp, check_dtype=False)
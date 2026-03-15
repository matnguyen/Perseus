# tests/pipelines/test_filter_pipeline.py

import argparse
from pathlib import Path
import pandas as pd
import torch
import pytest

from perseus.commands import filter as m

class DummyModel:
    def __call__(self, x, mask=None, extra=None):
        batch_size = x.shape[0]
        out_dim = 7
        # logits chosen so sigmoid is high for all ranks
        return torch.ones((batch_size, out_dim), dtype=torch.float32)

    def eval(self):
        return self
class DummyBar:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return lambda *a, **k: None
    def __exit__(self, *args):
        pass

@pytest.mark.pipeline
def test_run_filter_small(monkeypatch, tmp_path):
    input_shards = tmp_path / "shards"
    input_shards.mkdir()
    (input_shards / "permute_manifest.json").write_text("{}")

    kraken_path = tmp_path / "kraken.tsv"
    kraken_path.write_text(
        "C\tseq1\tkraken:taxid|60 (taxid 60)\t1000\t60:5 61:3\n"
        "C\tseq2\tkraken:taxid|61 (taxid 61)\t900\t61:4 60:1\n"
    )

    output_path = tmp_path / "filtered.tsv"

    args = argparse.Namespace(
        input_shards=str(input_shards),
        input_kraken=str(kraken_path),
        output_path=str(output_path),
        batch_size=128,
        cache_shards=1,
        downcast="fp16",
        cpu_float32=False,
        num_workers=0,
        split_dir=None,
        seed=667,
        output_all=False,
        model_path=None,
    )

    fake_batch = {
        "x": torch.zeros((2, 3, 4), dtype=torch.float32),
        "mask": torch.ones((2, 4), dtype=torch.bool),
        "lengths": torch.tensor([1000, 900]),
        "seq_id": ["seq1", "seq2"],
        "taxon": [60, 61],
    }

    def fake_build_loader(args, manifest_path, batch_size, is_train, shuffle, rank_filter=None):
        return None, [fake_batch]

    monkeypatch.setattr(m, "load_default_model", lambda out_dim, device=None: DummyModel())
    monkeypatch.setattr(m, "build_loader", fake_build_loader)
    monkeypatch.setattr(m, "alive_bar", DummyBar)
    monkeypatch.setattr(m, "get_lineage", lambda taxid: [1, 2, int(taxid)])
    monkeypatch.setattr(m, "get_rank", lambda taxid: "species")
    monkeypatch.setattr(
        m,
        "select_one_row_per_seq",
        lambda df, **kwargs: df.sort_values("sequence_id").reset_index(drop=True)
    )

    out_df = m.run_filter(args)

    assert output_path.exists()
    assert len(out_df) == 2
    assert set(out_df["sequence_id"]) == {"seq1", "seq2"}
    assert "prob_species" in out_df.columns
    assert "perseus_in_lineage" in out_df.columns
    assert "perseus_predicted_rank" in out_df.columns

    saved = pd.read_csv(output_path, sep="\t")
    assert len(saved) == 2
    
@pytest.mark.pipeline
def test_run_filter_requires_manifest(tmp_path, monkeypatch):
    input_shards = tmp_path / "shards"
    input_shards.mkdir()

    kraken_path = tmp_path / "kraken.tsv"
    kraken_path.write_text("C\tseq1\tkraken:taxid|60 (taxid 60)\t1000\t60:1\n")

    args = argparse.Namespace(
        input_shards=str(input_shards),
        input_kraken=str(kraken_path),
        output_path=str(tmp_path / "out.tsv"),
        batch_size=128,
        cache_shards=1,
        downcast="fp16",
        cpu_float32=False,
        num_workers=0,
        split_dir=None,
        seed=667,
        output_all=False,
        model_path=None,
    )

    monkeypatch.setattr(m, "load_default_model", lambda out_dim, device=None: DummyModel())

    with pytest.raises(SystemExit) as exc:
        m.run_filter(args)

    assert exc.value.code == 1
        
def test_run_filter_uses_explicit_model_path(monkeypatch, tmp_path):
    input_shards = tmp_path / "shards"
    input_shards.mkdir()
    (input_shards / "permute_manifest.json").write_text("{}")

    kraken_path = tmp_path / "kraken.tsv"
    kraken_path.write_text("C\tseq1\tkraken:taxid|60 (taxid 60)\t1000\t60:1\n")

    output_path = tmp_path / "filtered.tsv"
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")

    args = argparse.Namespace(
        input_shards=str(input_shards),
        input_kraken=str(kraken_path),
        output_path=str(output_path),
        batch_size=128,
        cache_shards=1,
        downcast="fp16",
        cpu_float32=False,
        num_workers=0,
        split_dir=None,
        seed=667,
        output_all=False,
        model_path=str(model_path),
    )

    fake_batch = {
        "x": torch.zeros((1, 3, 4)),
        "mask": torch.ones((1, 4), dtype=torch.bool),
        "lengths": torch.tensor([1000]),
        "seq_id": ["seq1"],
        "taxon": [60],
    }

    calls = {"make": 0, "load": 0}

    monkeypatch.setattr(m, "make_model", lambda out_dim, device: calls.__setitem__("make", calls["make"] + 1) or DummyModel())
    monkeypatch.setattr(m, "load_model", lambda model, path, device: calls.__setitem__("load", calls["load"] + 1) or model)
    monkeypatch.setattr(m, "build_loader", lambda *a, **k: (None, [fake_batch]))
    monkeypatch.setattr(m, "alive_bar", DummyBar)
    monkeypatch.setattr(m, "get_lineage", lambda taxid: [1, 2, int(taxid)])
    monkeypatch.setattr(m, "get_rank", lambda taxid: "species")
    monkeypatch.setattr(m, "select_one_row_per_seq", lambda df, **kwargs: df)

    m.run_filter(args)

    assert calls["make"] == 1
    assert calls["load"] == 1
import importlib
import torch

MODULE = "taxoncnn.trainer.utils"

def test_normalize_y_per_rank_to7_with_names():
    m = importlib.import_module(MODULE)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    rank_names = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]
    out = m.normalize_y_per_rank_to7(y, rank_names)
    assert torch.allclose(out, y)


def test_normalize_y_per_rank_to7_missing_names():
    m = importlib.import_module(MODULE)
    y = torch.tensor([1.0, 2.0, 3.0])
    rank_names = ["superkingdom", "phylum", "class"]
    out = m.normalize_y_per_rank_to7(y, rank_names)
    assert out.shape[0] == 7
    assert torch.all(out[:3] == y)
    assert torch.all(out[3:] == -1)


def test_normalize_y_per_rank_to7_no_names():
    m = importlib.import_module(MODULE)
    y = torch.tensor([1.0, 2.0, 3.0])
    out = m.normalize_y_per_rank_to7(y, None)
    assert out.shape[0] == 7
    assert torch.all(out[:3] == y)
    assert torch.all(out[3:] == -1)


def test_remap_rank_index_to7_found():
    m = importlib.import_module(MODULE)
    rank_names = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]
    idx = m.remap_rank_index_to7(2, rank_names)
    assert idx == 2
    

def test_remap_rank_index_to7_found_sub_super():
    m = importlib.import_module(MODULE)
    rank_names = ["superkingdom", "phylum", "superclass", "order", "family", "genus", "subspecies"]
    idx = m.remap_rank_index_to7(6, rank_names)
    assert idx == 6
    idx = m.remap_rank_index_to7(2, rank_names)
    assert idx == 2


def test_remap_rank_index_to7_not_found():
    m = importlib.import_module(MODULE)
    rank_names = ["strain", "subspecies", "other"]
    idx = m.remap_rank_index_to7(2, rank_names)
    assert idx == -1


def test_remap_rank_index_to7_bad_index():
    m = importlib.import_module(MODULE)
    rank_names = ["superkingdom", "phylum"]
    idx = m.remap_rank_index_to7(-1, rank_names)
    assert idx == -1
    idx = m.remap_rank_index_to7(10, rank_names)
    assert idx == -1


def test_build_rank_filtered_index(tmp_path):
    m = importlib.import_module(MODULE)
    # Create a fake shard file
    shard = {
        "x": torch.randn(3, 22, 10),
        "rank_index": torch.tensor([0, 1, 2])
    }
    shard_path = tmp_path / "shard0.pt"
    torch.save(shard, shard_path)
    # Should find one sample for each canonical rank
    pairs, stats = m.build_rank_filtered_index(str(tmp_path), "superkingdom", None)
    assert isinstance(pairs, list)
    assert isinstance(stats, dict)
    # Should find at least one sample for "superkingdom"
    assert any(idx == 0 for _, idx in pairs)
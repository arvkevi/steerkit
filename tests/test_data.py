from pathlib import Path

from steerkit import ContrastPair, load_pairs_jsonl, save_pairs_jsonl


def test_pairs_roundtrip(tmp_path: Path):
    pairs = [
        ContrastPair(prompt="hi", positive_response="yes", negative_response="no"),
        ContrastPair(prompt="how are you", positive_response="fine", negative_response="bad"),
    ]
    path = tmp_path / "pairs.jsonl"
    save_pairs_jsonl(pairs, path)
    loaded = load_pairs_jsonl(path)
    assert loaded == pairs


def test_refusal_dataset_loads():
    repo_root = Path(__file__).parent.parent
    pairs = load_pairs_jsonl(repo_root / "examples" / "data" / "refusal_pairs.jsonl")
    assert len(pairs) >= 80  # teacher-generated set is ~100 pairs across 4 prompt buckets
    assert all(p.prompt and p.positive_response and p.negative_response for p in pairs)

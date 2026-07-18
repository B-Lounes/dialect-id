from pathlib import Path

from tools.big_run.prepare import infer_cc, stable_shard


def test_tar_sharding_is_deterministic_and_bounded() -> None:
    first = stable_shard("dataset/cc=ma/a.tar", 32)
    assert first == stable_shard("dataset/cc=ma/a.tar", 32)
    assignments = {stable_shard(f"dataset/sample-{idx}.tar", 32) for idx in range(200)}
    assert all(0 <= shard < 32 for shard in assignments)
    assert len(assignments) > 20


def test_country_code_is_inferred_from_partition_path() -> None:
    assert infer_cc(Path("dataset/split=train/cc=ma/audio.tar")) == "MA"
    assert infer_cc(Path("dataset/no-label/audio.tar")) == ""

import torch

from dialect_id.labels import CODE_TO_ID, num_dialects
from dialect_id.train import (
    aggregate_chunk_outputs,
    apply_active_dialect_mask,
    apply_active_soft_target_mask,
    mean_by_sample,
)


def test_active_mask_blocks_logits_and_renormalizes_soft_targets() -> None:
    active = torch.ones(num_dialects(), dtype=torch.bool)
    active[CODE_TO_ID["DJ"]] = False
    logits = torch.arange(num_dialects(), dtype=torch.float32).unsqueeze(0)
    masked = apply_active_dialect_mask(logits, active)

    assert masked is not None
    assert masked[0, CODE_TO_ID["DJ"]] == torch.finfo(torch.float32).min
    assert masked[0, CODE_TO_ID["AE"]] == logits[0, CODE_TO_ID["AE"]]

    targets = torch.full((1, num_dialects()), 1.0 / num_dialects())
    renormalized = apply_active_soft_target_mask(targets, torch.tensor([CODE_TO_ID["AE"]]), active)
    assert renormalized is not None
    assert renormalized[0, CODE_TO_ID["DJ"]] == 0
    assert torch.allclose(renormalized.sum(dim=1), torch.ones(1))


def test_chunk_outputs_are_mean_aggregated_per_sample() -> None:
    chunks = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 14.0]])
    mapping = torch.tensor([0, 0, 1])
    expected = torch.tensor([[2.0, 4.0], [10.0, 14.0]])

    assert torch.equal(mean_by_sample(chunks, mapping, 2), expected)

    outputs = {"dialect_logits": chunks, "specialists": {"gulf": chunks + 1}, "metadata": "kept"}
    aggregated = aggregate_chunk_outputs(
        outputs,
        {"chunk_to_sample": mapping, "num_samples": torch.tensor(2)},
    )
    assert torch.equal(aggregated["dialect_logits"], expected)
    assert torch.equal(aggregated["specialists"]["gulf"], expected + 1)
    assert aggregated["metadata"] == "kept"

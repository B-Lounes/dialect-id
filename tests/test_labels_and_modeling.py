import torch

from dialect_id.labels import (
    CODE_TO_ID,
    DIALECTS,
    DIALECT_ID_TO_REGION_ID,
    REGIONS,
    SPECIALIST_DIALECT_IDS,
)
from dialect_id.modeling import dialect_region_marginal_logits


def test_label_space_is_stable_and_complete() -> None:
    assert len(DIALECTS) == 23
    assert [label.id for label in DIALECTS] == list(range(23))
    assert len(CODE_TO_ID) == 23
    assert {DIALECTS[idx].region for idx in range(23)} == set(REGIONS)
    assert SPECIALIST_DIALECT_IDS["east_africa"] == (
        CODE_TO_ID["DJ"],
        CODE_TO_ID["KM"],
        CODE_TO_ID["SO"],
    )


def test_region_logits_are_exact_dialect_probability_marginals() -> None:
    generator = torch.Generator().manual_seed(7)
    dialect_logits = torch.randn(4, 23, generator=generator)
    region_logits = dialect_region_marginal_logits(dialect_logits)

    dialect_probs = dialect_logits.softmax(dim=1)
    expected = torch.zeros(4, len(REGIONS))
    for dialect_id, region_id in DIALECT_ID_TO_REGION_ID.items():
        expected[:, region_id] += dialect_probs[:, dialect_id]

    assert torch.allclose(region_logits.softmax(dim=1), expected, atol=1e-6)

import torch
from torch import nn

from dialect_id.checkpointing import load_compatible_state_dict
from dialect_id.labels import CODE_TO_ID, num_dialects


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.Module()
        self.heads.dialect = nn.Linear(2, num_dialects())


def test_checkpoint_expansion_maps_classifier_rows_by_code() -> None:
    model = TinyModel()
    with torch.no_grad():
        model.heads.dialect.weight.fill_(-1.0)
        model.heads.dialect.bias.fill_(-2.0)

    source_state = {
        "heads.dialect.weight": torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
        "heads.dialect.bias": torch.tensor([1.0, 2.0]),
    }
    source_labels = {
        "dialects": [
            {"id": 0, "code": "MA"},
            {"id": 1, "code": "AE"},
        ]
    }
    load_compatible_state_dict(model, source_state, strict=False, source_labels=source_labels)

    assert torch.equal(model.heads.dialect.weight[CODE_TO_ID["MA"]], torch.tensor([10.0, 11.0]))
    assert torch.equal(model.heads.dialect.weight[CODE_TO_ID["AE"]], torch.tensor([20.0, 21.0]))
    assert model.heads.dialect.bias[CODE_TO_ID["MA"]].item() == 1.0
    assert model.heads.dialect.bias[CODE_TO_ID["AE"]].item() == 2.0
    assert torch.equal(model.heads.dialect.weight[CODE_TO_ID["DZ"]], torch.tensor([-1.0, -1.0]))

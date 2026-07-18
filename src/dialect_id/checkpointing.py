from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .labels import CODE_TO_ID


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def load_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool,
    source_labels: dict[str, Any] | None = None,
) -> None:
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return
    current = model.state_dict()
    expanded = _label_aware_expansions(current, state_dict, source_labels)
    compatible = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current and tuple(current[name].shape) == tuple(tensor.shape)
    }
    compatible.update(expanded)
    model.load_state_dict(compatible, strict=False)


def _source_code_to_id(source_labels: dict[str, Any] | None) -> dict[str, int]:
    if not source_labels:
        return {}
    dialects = source_labels.get("dialects")
    if not isinstance(dialects, list):
        return {}
    result: dict[str, int] = {}
    for row in dialects:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "").upper()
        try:
            idx = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if code:
            result[code] = idx
    return result


def _copy_label_rows(
    target: torch.Tensor,
    source: torch.Tensor,
    source_code_to_id: dict[str, int],
) -> torch.Tensor:
    output = target.detach().clone()
    for code, dst_idx in CODE_TO_ID.items():
        src_idx = source_code_to_id.get(code)
        if src_idx is None:
            continue
        if 0 <= src_idx < source.shape[0] and 0 <= dst_idx < output.shape[0]:
            output[dst_idx].copy_(source[src_idx].to(dtype=output.dtype))
    return output


def _label_aware_expansions(
    current: dict[str, torch.Tensor],
    state_dict: dict[str, torch.Tensor],
    source_labels: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    source_code_to_id = _source_code_to_id(source_labels)
    if not source_code_to_id:
        return {}
    expanded: dict[str, torch.Tensor] = {}
    label_head_names = {"heads.dialect.weight", "heads.dialect.bias", "heads.accent.weight", "heads.accent.bias"}
    for name in label_head_names:
        if name not in current or name not in state_dict:
            continue
        target = current[name]
        source = state_dict[name]
        if target.ndim not in {1, 2} or source.ndim != target.ndim:
            continue
        if target.ndim == 2 and target.shape[1] != source.shape[1]:
            continue
        if target.shape[0] == source.shape[0]:
            continue
        expanded[name] = _copy_label_rows(target, source, source_code_to_id)
    return expanded


def load_checkpoint_weights(
    model: nn.Module,
    checkpoint: dict[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    raw_model = unwrap_model(model)
    train_args = checkpoint.get("args", {})
    init_path = train_args.get("init_from_checkpoint")
    if checkpoint.get("state_type") == "trainable_only" and init_path:
        path = Path(init_path)
        if path.exists():
            init_payload = torch.load(path, map_location=map_location)
            load_compatible_state_dict(
                raw_model,
                init_payload["model"],
                strict=False,
                source_labels=init_payload.get("labels"),
            )
    load_compatible_state_dict(
        raw_model,
        checkpoint["model"],
        strict=checkpoint.get("state_type") != "trainable_only",
        source_labels=checkpoint.get("labels"),
    )

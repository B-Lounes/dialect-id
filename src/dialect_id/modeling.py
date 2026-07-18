from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, WhisperModel

from .labels import DIALECT_ID_TO_REGION_ID, SPECIALIST_DIALECT_IDS, num_dialects, num_regions


class AttentiveStatsPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        mean = torch.sum(x * weights, dim=1)
        var = torch.sum(((x - mean.unsqueeze(1)) ** 2) * weights, dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=-1)


class ClassifierHeads(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        dialects: int = num_dialects(),
        regions: int = num_regions(),
        lang_types: int = 2,
        accents: int = num_dialects(),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dialect = nn.Linear(hidden_dim, dialects)
        self.region = nn.Linear(hidden_dim, regions)
        self.lang_type = nn.Linear(hidden_dim, lang_types)
        self.accent = nn.Linear(hidden_dim, accents)
        self.specialists = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, len(dialect_ids))
                for name, dialect_ids in SPECIALIST_DIALECT_IDS.items()
                if len(dialect_ids) > 1
            }
        )

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(pooled)
        return {
            "dialect_logits": self.dialect(hidden),
            "region_logits": self.region(hidden),
            "lang_type_logits": self.lang_type(hidden),
            "accent_logits": self.accent(hidden),
            "specialist_logits": {
                name: head(hidden) for name, head in self.specialists.items()
            },
            "embedding": hidden,
        }


def dialect_region_marginal_logits(dialect_logits: torch.Tensor) -> torch.Tensor:
    """Return region logits whose softmax is the sum of dialect probabilities."""
    region_rows = []
    for region_id in range(num_regions()):
        dialect_ids = [
            dialect_id
            for dialect_id in range(num_dialects())
            if DIALECT_ID_TO_REGION_ID[dialect_id] == region_id
        ]
        if not dialect_ids:
            region_rows.append(torch.full_like(dialect_logits[:, 0], torch.finfo(dialect_logits.dtype).min))
            continue
        index = torch.tensor(dialect_ids, dtype=torch.long, device=dialect_logits.device)
        region_rows.append(torch.logsumexp(dialect_logits.index_select(1, index), dim=1))
    return torch.stack(region_rows, dim=1)


class CleanTaxonomyHeads(nn.Module):
    """Primary dialect head plus task heads; region is derived from dialect mass."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        dialects: int = num_dialects(),
        lang_types: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dialect = nn.Linear(hidden_dim, dialects)
        self.lang_type = nn.Linear(hidden_dim, lang_types)
        self.specialists = nn.ModuleDict()

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(pooled)
        dialect_logits = self.dialect(hidden)
        return {
            "dialect_logits": dialect_logits,
            "region_logits": dialect_region_marginal_logits(dialect_logits),
            "region_logits_are_dialect_marginals": True,
            "lang_type_logits": self.lang_type(hidden),
            "specialist_logits": {},
            "embedding": hidden,
        }


def make_classifier_heads(
    head_type: str,
    input_dim: int,
    *,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if head_type == "legacy":
        return ClassifierHeads(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if head_type == "clean_taxonomy":
        return CleanTaxonomyHeads(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported classifier_head_type={head_type!r}")


def _set_trainable_layers(layers: Any, freeze_encoder_layers: int) -> None:
    if freeze_encoder_layers <= 0:
        return
    for layer in list(layers)[:freeze_encoder_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def _llm_layers(model: nn.Module) -> Any | None:
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return layers
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        layers = getattr(encoder, "layers", None)
        if layers is not None:
            return layers
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        return getattr(transformer, "h", None)
    return None


def _unfreeze_llm_tail(model: nn.Module, trainable_layers: int) -> int:
    if trainable_layers <= 0:
        return 0
    layers = _llm_layers(model)
    if layers is None:
        return 0
    layer_list = list(layers)
    for layer in layer_list[-trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    for attr in ("norm", "ln_f", "final_layernorm"):
        module = getattr(model, attr, None)
        if module is None and getattr(model, "model", None) is not None:
            module = getattr(model.model, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
    return min(trainable_layers, len(layer_list))


def _config_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, attr, 0)
        if value:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return _config_hidden_size(text_config)
    return 0


def _ensure_qwen_asr_package() -> None:
    try:
        __import__("qwen_asr")
    except ImportError as exc:
        raise ImportError(
            "The qwen3_asr_audio backend is optional. Install the pinned extra "
            "with `python -m pip install 'dialect-id[qwen]'`, or use another "
            "model_type. See docs/qwen-integration.md."
        ) from exc


def _load_qwen3_asr_audio_weights(encoder: nn.Module, model_dir: str | Path) -> None:
    from safetensors.torch import load_file

    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    prefix = "thinker.audio_tower."
    state: dict[str, torch.Tensor] = {}
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = payload["weight_map"]
        files = sorted({filename for key, filename in weight_map.items() if key.startswith(prefix)})
        for filename in files:
            tensors = load_file(str(model_path / filename), device="cpu")
            for key, tensor in tensors.items():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = tensor
    else:
        tensors = load_file(str(model_path / "model.safetensors"), device="cpu")
        for key, tensor in tensors.items():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = tensor
    if not state:
        raise ValueError(f"No Qwen3-ASR audio tower weights found in {model_path}")
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if key]
    missing = [key for key in missing if not key.startswith("positional_embedding.")]
    if missing or unexpected:
        raise ValueError(
            "Could not load Qwen3-ASR audio tower cleanly: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )


class WhisperDialectClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        freeze_encoder_layers: int = 24,
        freeze_conv: bool = True,
        gradient_checkpointing: bool = True,
        pool_hidden_dim: int = 256,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        base = WhisperModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.encoder = base.encoder
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if freeze_conv:
            for name, param in self.encoder.named_parameters():
                if name.startswith("conv"):
                    param.requires_grad = False
        if hasattr(self.encoder, "layers"):
            _set_trainable_layers(self.encoder.layers, freeze_encoder_layers)
        hidden_size = int(base.config.d_model)
        del base
        self.pool = AttentiveStatsPool(hidden_size, pool_hidden_dim)
        self.heads = ClassifierHeads(
            hidden_size * 2,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(self, input_features: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_features=input_features)
        pooled = self.pool(outputs.last_hidden_state)
        return self.heads(pooled)


class WhisperAudioLlmClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        llm_name_or_path: str,
        *,
        cache_dir: str | None = None,
        llm_cache_dir: str | None = None,
        local_files_only: bool = False,
        freeze_encoder_layers: int = 24,
        freeze_conv: bool = True,
        freeze_llm: bool = True,
        llm_trainable_layers: int = 0,
        gradient_checkpointing: bool = True,
        audio_token_count: int = 32,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        base = WhisperModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.encoder = base.encoder
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if freeze_conv:
            for name, param in self.encoder.named_parameters():
                if name.startswith("conv"):
                    param.requires_grad = False
        if hasattr(self.encoder, "layers"):
            _set_trainable_layers(self.encoder.layers, freeze_encoder_layers)
        speech_hidden = int(base.config.d_model)
        del base

        self.audio_queries = nn.Parameter(torch.empty(audio_token_count, speech_hidden))
        nn.init.normal_(self.audio_queries, mean=0.0, std=0.02)
        self.audio_resampler = nn.MultiheadAttention(
            speech_hidden,
            num_heads=8,
            batch_first=True,
        )
        self.audio_norm = nn.LayerNorm(speech_hidden)

        llm_load_kwargs: dict[str, Any] = {
            "cache_dir": llm_cache_dir or cache_dir,
            "local_files_only": local_files_only,
        }
        self.llm = AutoModel.from_pretrained(llm_name_or_path, **llm_load_kwargs)
        self.llm_frozen = freeze_llm and llm_trainable_layers <= 0
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            _unfreeze_llm_tail(self.llm, llm_trainable_layers)
        elif gradient_checkpointing and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        llm_hidden = _config_hidden_size(self.llm.config)
        if llm_hidden <= 0:
            raise ValueError(f"Could not infer LLM hidden size for {llm_name_or_path}")
        self.audio_to_llm = nn.Sequential(
            nn.LayerNorm(speech_hidden),
            nn.Linear(speech_hidden, llm_hidden),
        )
        self.heads = ClassifierHeads(
            llm_hidden,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(self, input_features: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        audio_hidden = self.encoder(input_features=input_features).last_hidden_state
        batch_size = audio_hidden.shape[0]
        queries = self.audio_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tokens, _ = self.audio_resampler(
            query=queries,
            key=audio_hidden,
            value=audio_hidden,
            need_weights=False,
        )
        tokens = self.audio_norm(tokens + queries)
        llm_inputs = self.audio_to_llm(tokens)
        attention_mask = torch.ones(
            llm_inputs.shape[:2],
            dtype=torch.long,
            device=llm_inputs.device,
        )
        if self.llm_frozen:
            self.llm.eval()
        try:
            outputs = self.llm(
                inputs_embeds=llm_inputs,
                attention_mask=attention_mask,
                use_cache=False,
            )
        except TypeError:
            outputs = self.llm(
                inputs_embeds=llm_inputs,
                attention_mask=attention_mask,
            )
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.heads(pooled)


class WhisperAudioLlmFusionClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        llm_name_or_path: str,
        *,
        cache_dir: str | None = None,
        llm_cache_dir: str | None = None,
        local_files_only: bool = False,
        freeze_encoder_layers: int = 24,
        freeze_conv: bool = True,
        freeze_llm: bool = True,
        llm_trainable_layers: int = 0,
        gradient_checkpointing: bool = True,
        audio_token_count: int = 32,
        pool_hidden_dim: int = 256,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        base = WhisperModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.encoder = base.encoder
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if freeze_conv:
            for name, param in self.encoder.named_parameters():
                if name.startswith("conv"):
                    param.requires_grad = False
        if hasattr(self.encoder, "layers"):
            _set_trainable_layers(self.encoder.layers, freeze_encoder_layers)
        speech_hidden = int(base.config.d_model)
        del base

        self.pool = AttentiveStatsPool(speech_hidden, pool_hidden_dim)
        self.audio_queries = nn.Parameter(torch.empty(audio_token_count, speech_hidden))
        nn.init.normal_(self.audio_queries, mean=0.0, std=0.02)
        self.audio_resampler = nn.MultiheadAttention(
            speech_hidden,
            num_heads=8,
            batch_first=True,
        )
        self.audio_norm = nn.LayerNorm(speech_hidden)

        llm_load_kwargs: dict[str, Any] = {
            "cache_dir": llm_cache_dir or cache_dir,
            "local_files_only": local_files_only,
        }
        self.llm = AutoModel.from_pretrained(llm_name_or_path, **llm_load_kwargs)
        self.llm_frozen = freeze_llm and llm_trainable_layers <= 0
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            _unfreeze_llm_tail(self.llm, llm_trainable_layers)
        elif gradient_checkpointing and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        llm_hidden = _config_hidden_size(self.llm.config)
        if llm_hidden <= 0:
            raise ValueError(f"Could not infer LLM hidden size for {llm_name_or_path}")
        self.audio_to_llm = nn.Sequential(
            nn.LayerNorm(speech_hidden),
            nn.Linear(speech_hidden, llm_hidden),
        )
        self.heads = ClassifierHeads(
            llm_hidden + speech_hidden * 2,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(self, input_features: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        audio_hidden = self.encoder(input_features=input_features).last_hidden_state
        speech_pooled = self.pool(audio_hidden)
        batch_size = audio_hidden.shape[0]
        queries = self.audio_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tokens, _ = self.audio_resampler(
            query=queries,
            key=audio_hidden,
            value=audio_hidden,
            need_weights=False,
        )
        tokens = self.audio_norm(tokens + queries)
        llm_inputs = self.audio_to_llm(tokens)
        attention_mask = torch.ones(
            llm_inputs.shape[:2],
            dtype=torch.long,
            device=llm_inputs.device,
        )
        if self.llm_frozen:
            self.llm.eval()
        try:
            outputs = self.llm(
                inputs_embeds=llm_inputs,
                attention_mask=attention_mask,
                use_cache=False,
            )
        except TypeError:
            outputs = self.llm(
                inputs_embeds=llm_inputs,
                attention_mask=attention_mask,
            )
        llm_pooled = outputs.last_hidden_state.mean(dim=1)
        return self.heads(torch.cat([llm_pooled, speech_pooled], dim=-1))


class SSLDialectClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        freeze_feature_encoder: bool = True,
        freeze_encoder_layers: int = 12,
        gradient_checkpointing: bool = True,
        pool_hidden_dim: int = 256,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        load_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if local_files_only:
            load_kwargs["use_safetensors"] = False
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if freeze_feature_encoder and hasattr(self.encoder, "feature_extractor"):
            for param in self.encoder.feature_extractor.parameters():
                param.requires_grad = False
        layers = getattr(getattr(self.encoder, "encoder", None), "layers", None)
        if layers is not None:
            _set_trainable_layers(layers, freeze_encoder_layers)
        hidden_size = int(self.encoder.config.hidden_size)
        self.pool = AttentiveStatsPool(hidden_size, pool_hidden_dim)
        self.heads = ClassifierHeads(
            hidden_size * 2,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = None
        if attention_mask is not None:
            try:
                lengths = self.encoder._get_feat_extract_output_lengths(attention_mask.sum(-1))
                max_len = hidden.shape[1]
                mask = torch.arange(max_len, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
            except Exception:
                mask = None
        pooled = self.pool(hidden, mask=mask)
        return self.heads(pooled)


class Wav2Vec2BertDialectClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        freeze_encoder_layers: int = 12,
        gradient_checkpointing: bool = True,
        pool_hidden_dim: int = 256,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        load_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        layers = getattr(getattr(self.encoder, "encoder", None), "layers", None)
        if layers is not None:
            _set_trainable_layers(layers, freeze_encoder_layers)
        hidden_size = int(self.encoder.config.hidden_size)
        self.pool = AttentiveStatsPool(hidden_size, pool_hidden_dim)
        self.heads = ClassifierHeads(
            hidden_size * 2,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = None
        if attention_mask is not None:
            if attention_mask.shape[-1] == hidden.shape[1]:
                mask = attention_mask
            elif hasattr(self.encoder, "_get_feat_extract_output_lengths"):
                try:
                    lengths = self.encoder._get_feat_extract_output_lengths(attention_mask.sum(-1))
                    max_len = hidden.shape[1]
                    mask = torch.arange(max_len, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
                except Exception:
                    mask = None
        pooled = self.pool(hidden, mask=mask)
        return self.heads(pooled)


class Qwen3ASRAudioDialectClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        freeze_encoder_layers: int = 12,
        gradient_checkpointing: bool = True,
        pool_hidden_dim: int = 256,
        classifier_hidden_dim: int = 512,
        classifier_head_type: str = "legacy",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _ensure_qwen_asr_package()
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

        if not model_name_or_path:
            raise ValueError("qwen3_asr_audio requires model_name_or_path")
        model_path = Path(model_name_or_path)
        if not model_path.exists():
            raise ValueError(
                "qwen3_asr_audio currently expects a local snapshot path so it can "
                "load only thinker.audio_tower weights."
            )
        config_payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        audio_config = Qwen3ASRAudioEncoderConfig(**config_payload["thinker_config"]["audio_config"])
        self.encoder = Qwen3ASRAudioEncoder(audio_config)
        _load_qwen3_asr_audio_weights(self.encoder, model_path)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing = True
        if freeze_encoder_layers > 0:
            layers = list(self.encoder.layers)
            if freeze_encoder_layers >= len(layers):
                for param in self.encoder.parameters():
                    param.requires_grad = False
            else:
                for layer in layers[:freeze_encoder_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
        hidden_size = int(audio_config.output_dim)
        self.pool = AttentiveStatsPool(hidden_size, pool_hidden_dim)
        self.heads = make_classifier_heads(
            classifier_head_type,
            hidden_size * 2,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        feature_mask = feature_attention_mask if feature_attention_mask is not None else attention_mask
        if feature_mask is None:
            feature_lens = torch.full(
                (input_features.shape[0],),
                input_features.shape[-1],
                dtype=torch.long,
                device=input_features.device,
            )
        else:
            feature_lens = feature_mask.sum(-1).to(device=input_features.device, dtype=torch.long)
        pieces = []
        lengths = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            feature_len = feature_len.clamp_min(1)
            output = self.encoder(
                input_feature[:, : int(feature_len.item())],
                feature_lens=feature_len.unsqueeze(0),
            )
            piece = output.last_hidden_state
            pieces.append(piece)
            lengths.append(piece.shape[0])
        hidden = pad_sequence(pieces, batch_first=True)
        max_len = hidden.shape[1]
        mask = torch.arange(max_len, device=hidden.device).unsqueeze(0) < torch.tensor(
            lengths,
            dtype=torch.long,
            device=hidden.device,
        ).unsqueeze(1)
        pooled = self.pool(hidden, mask=mask)
        return self.heads(pooled)


class SERes2Block(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.tdnn1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.tdnn2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=8,
        )
        self.tdnn3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(8, channels // 8), kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(max(8, channels // 8), channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.tdnn1(x)))
        out = self.relu(self.bn2(self.tdnn2(out)))
        out = self.bn3(self.tdnn3(out))
        out = out * self.se(out)
        return self.relu(out + residual)


class EcapaDialectClassifier(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 512,
        n_mels: int = 80,
        sample_rate: int = 16000,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=20,
            f_max=7600,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.input = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )
        self.blocks = nn.ModuleList(
            [
                SERes2Block(channels, dilation=2),
                SERes2Block(channels, dilation=3),
                SERes2Block(channels, dilation=4),
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )
        self.pool = AttentiveStatsPool(channels)
        self.heads = ClassifierHeads(
            channels * 2,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(self, input_values: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        feats = self.to_db(self.mel(input_values)).clamp(min=-80.0, max=80.0)
        feats = feats - feats.mean(dim=-1, keepdim=True)
        x = self.input(feats)
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        x = self.mix(torch.cat(block_outputs, dim=1))
        pooled = self.pool(x.transpose(1, 2))
        return self.heads(pooled)


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    model_name_or_path: str | None


def build_model(
    model_type: str,
    model_name_or_path: str | None,
    *,
    cache_dir: str | None = None,
    llm_name_or_path: str | None = None,
    llm_cache_dir: str | None = None,
    local_files_only: bool = False,
    freeze_encoder_layers: int = 0,
    freeze_feature_encoder: bool = True,
    freeze_llm: bool = True,
    llm_trainable_layers: int = 0,
    gradient_checkpointing: bool = True,
    audio_llm_tokens: int = 32,
    classifier_hidden_dim: int = 512,
    classifier_head_type: str = "legacy",
    dropout: float = 0.1,
) -> nn.Module:
    if model_type == "whisper":
        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required for whisper")
        return WhisperDialectClassifier(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            freeze_encoder_layers=freeze_encoder_layers,
            gradient_checkpointing=gradient_checkpointing,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
    if model_type in {"whisper_llm", "whisper_llm_fusion"}:
        if model_name_or_path is None:
            raise ValueError(f"model_name_or_path is required for {model_type}")
        if llm_name_or_path is None:
            raise ValueError(f"llm_name_or_path is required for {model_type}")
        model_cls = WhisperAudioLlmFusionClassifier if model_type == "whisper_llm_fusion" else WhisperAudioLlmClassifier
        return model_cls(
            model_name_or_path,
            llm_name_or_path,
            cache_dir=cache_dir,
            llm_cache_dir=llm_cache_dir,
            local_files_only=local_files_only,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_llm=freeze_llm,
            llm_trainable_layers=llm_trainable_layers,
            gradient_checkpointing=gradient_checkpointing,
            audio_token_count=audio_llm_tokens,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
    if model_type == "w2vbert":
        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required for w2vbert")
        return Wav2Vec2BertDialectClassifier(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            freeze_encoder_layers=freeze_encoder_layers,
            gradient_checkpointing=gradient_checkpointing,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
    if model_type == "qwen3_asr_audio":
        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required for qwen3_asr_audio")
        return Qwen3ASRAudioDialectClassifier(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            freeze_encoder_layers=freeze_encoder_layers,
            gradient_checkpointing=gradient_checkpointing,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_head_type=classifier_head_type,
            dropout=dropout,
        )
    if model_type in {"wav2vec", "wavlm", "xlsr"}:
        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required for wav2vec/wavlm/xlsr")
        return SSLDialectClassifier(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            freeze_feature_encoder=freeze_feature_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            gradient_checkpointing=gradient_checkpointing,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
    if model_type == "ecapa":
        return EcapaDialectClassifier(
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")

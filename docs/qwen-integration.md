# Optional Qwen3-ASR Integration

`model_type=qwen3_asr_audio` uses only the audio tower and expects a separately
installed compatible `qwen_asr` implementation. The official
[`qwen-asr==0.0.6`](https://pypi.org/project/qwen-asr/0.0.6/) wheel contains the
two modules imported below and is declared as the pinned `qwen` optional extra:

```bash
python -m pip install -e '.[qwen]'
```

At runtime the backend imports:

```text
qwen_asr.core.transformers_backend.configuration_qwen3_asr.Qwen3ASRAudioEncoderConfig
qwen_asr.core.transformers_backend.modeling_qwen3_asr.Qwen3ASRAudioEncoder
```

The model directory must contain the compatible configuration and either a
`model.safetensors` file or `model.safetensors.index.json` plus referenced
shards. Only audio-tower tensors are loaded.

The extra can pull substantial ASR dependencies, so it remains opt-in. Verify
the two imports in the target environment before launching training. Do not add
absolute package paths or `sys.path` injection to the repository. The code
raises an actionable `ImportError` only when this backend is selected, so all
other backends remain usable without Qwen3-ASR.

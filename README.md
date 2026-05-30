**SUARA**

SUARA is an autoregressive language modeling implementation focused on efficient causal convolution-based sequence modeling and flexible FFN choices (MoE / dense). The codebase provides training, evaluation, and generation utilities together with configuration-driven experiments and checkpointing.

**Features**
- **Architecture**: token embedding + stacked `Block` modules combining a CausalWaveConv attention-like convolution and a SwiGLU or MoE feed-forward.
- **Flexible FFN**: supports dense SwiGLU or Mixture-of-Experts (MoE) routing with auxiliary load-balancing loss.
- **Normalizers**: multiple normalization options (RMS, Layer, Dual, Group, Power, etc.).
- **Training utilities**: configurable optimizer/scheduler, mixed precision (bf16/fp16), gradient checkpointing, wandb support.
- **Inference**: bundled generation helpers and support for inference bundles.

**Repository Structure (key files)**
- **Main entry**: [main.py](main.py) — training entrypoint and dataset loader.
- **Model**: [pipeline/model.py](pipeline/model.py) — `SuaRA` model definition.
- **Blocks & primitives**: [arc/block.py](arc/block.py), [arc/waveconv.py](arc/waveconv.py), [arc/ffn.py](arc/ffn.py), [arc/normalizer.py](arc/normalizer.py).
- **Training loop**: [pipeline/training.py](pipeline/training.py).
- **Inference**: [pipeline/inference.py](pipeline/inference.py) — tokenizer wrapper and generation utilities.
- **Config**: [data/configuration/config.yaml](data/configuration/config.yaml) (defaults) and loader [data/configuration/config.py](data/configuration/config.py).
- **Artifacts**: `artifacts/checkpoints/` — saved `best.pt`, `last.pt` and exported bundles.

**Installation**
Install Python requirements and optional extras (tokenizers, sentencepiece, transformers):

```bash
python -m pip install -r requirements.txt
# If using HF tokenizers or sentencepiece:
python -m pip install sentencepiece tokenizers transformers
```

**Quick Start**

- Train a model (uses `data/configuration/config.yaml` by default):

```bash
python main.py --config data/configuration/config.yaml --label suara_experiment
```

- Generate text from a checkpoint:

```bash
python pipeline/inference.py "Your prompt here" --config data/configuration/config.yaml
```

- Count model parameters from config:

```bash
python utils/calculate_params.py --config data/configuration/config.yaml
```

**Configuration**
- Project configuration is read with `load_config()` from [data/configuration/config.py](data/configuration/config.py). The main YAML (defaults) is [data/configuration/config.yaml](data/configuration/config.yaml).
- Important config sections: `model`, `training`, `training_data`, `tokenizer`, `generation`, `checkpoint`, `wandb`.

**Model architecture (brief)**
- `SuaRA` embeds token ids and applies a stack of `Block` modules. Each `Block` combines:
	- `CausalWaveConv` — a learned, causal convolution implemented via FFT kernels for long-range context.
	- A normalization layer (configurable via `norm_type`).
	- A `FeedForward` which is either `SwiGLU` (dense) or `MoE` (mixture-of-experts) with auxiliary load-balancing loss.

**Research status**
- This architecture is experimental and under active research. The implementation and design choices are intended for investigation and development; it is not guaranteed to be production-ready. Use for experimentation, benchmarking, and further research; evaluate stability and performance before any production deployment.

**Data & Tokenization**
- Tokenizer config lives in `data/configuration/config.yaml` and is loaded by `InferenceTokenizer` in [pipeline/inference.py](pipeline/inference.py).
- Training data expects tokenized parquet files (see `TrainingDataConfig.train_tokens_path` in [data/configuration/config.py](data/configuration/config.py)).

**Checkpoints & Bundles**
- Training saves `best.pt` and `last.pt` under the configured checkpoint `output_directory` (default `artifacts/checkpoints`).
- Inference can also load exported bundles with the internal format `suara_inference_bundle` containing `model_state_dict`, `model_config`, and `tokenizer_config`.

**Development & Contributing**
- Run linters and tests (not included in repo) before opening PRs.
- When adding features, keep `config.yaml` defaults updated and add example runs to `artifacts/plots`.

**License**
- See the `LICENSE` file in the repository root.

If you'd like, I can add example commands, training tips, or a short model card summarizing benchmark results — tell me which you'd prefer.

# SUARA

An experimental language model built around **CausalWaveConv** — a frequency-domain sequence mixer
that replaces self-attention with learned wavelet convolutions. The core research question: can a
model learn long-range dependencies through spectral filtering instead of pairwise token comparisons?

> **Research status.** This architecture is under active investigation. It is not production-ready.
> Use it for experimentation, benchmarking, and further research.

---

## What is CausalWaveConv?

Standard attention asks *"how much should token A attend to token B?"* for every pair — powerful,
but O(L²) in time and memory. CausalWaveConv takes a different route: it learns **wave-shaped
filters** that sweep across the entire sequence and pick out different temporal patterns. Think of
it as a bank of tunable radio receivers — each one resonates at a different frequency and timescale,
and the model learns which frequencies matter.

### Signal flow

```
  Input sequence  x  [B, L, D]
         │
         ├─────────────────────────────────────────────┐
         │                                             │
         ▼                                             ▼
   ┌─────────────┐                             ┌─────────────┐
   │   W_v proj  │                             │   W_q proj  │
   │  (values)   │                             │   (gates)   │
   └──────┬──────┘                             └──────┬──────┘
          │  v [B, L, H, Dh]                          │  sigmoid
          │                                           │  gate [B, L, H, K]
          ▼                                           │
   ┌──────────────────────────────────┐               │
   │         FFT convolution          │               │
   │                                  │               │
   │  1. pad v  →  FFT(v)             │               │
   │  2. build kernel  (H × K scales) │               │
   │     ┌──────────────────────┐     │               │
   │     │  static Morlet basis │     │               │
   │     │  + dynamic δ(input)  │     │               │
   │     │  + causal enforcement│     │               │
   │     └──────────────────────┘     │               │
   │  3. FFT(v) × kernel              │               │
   │  4. IFFT  →  per-scale output    │               │
   └──────────────┬───────────────────┘               │
                  │  [B, H, K, L, Dh]                 │
                  ▼                                    │
          ┌───────────────┐                            │
          │ scale_interact│  (mix across K scales)     │
          └───────┬───────┘                            │
                  │                                    │
                  └──────────────┬─────────────────────┘
                                 │  weighted sum over K
                                 ▼
                        ┌─────────────────┐
                        │    out_proj      │
                        └────────┬────────┘
                                 │
                          output [B, L, D]
```

### Stage 1 — building the kernel

Each convolution kernel is a superposition of **Morlet wavelets** (oscillating waves with a Gaussian
envelope), composed of two parts:

```
  kernel = static_component + dynamic_correction
               │                      │
               │                      └─ small projection from mean(x)
               │                         lets the filter adapt to context
               │
               └─ learned ω₀, amplitude, phase shift
                  the model's prior on what patterns to look for

  Then: enforce causality in the frequency domain
        (zero out "future" side of the spectrum)
```

### Stage 2 — FFT convolution

Convolving over the full sequence length via the FFT:

```
  output = IFFT( FFT(v_padded) × kernel )

  Complexity:  O(L log L)   vs   O(L²) for attention
```

Mathematically equivalent to a causal convolution over every position, but long sequences stay
tractable.

### Stage 3 — multi-scale gating

The model runs `K` wavelet scales in parallel (short, medium, long range) across `H` heads:

```
  scales:   [──── short ────]   captures local syntax
            [─────── mid ────────]   captures phrase structure
            [────────────── long ──────────────]   captures document-level patterns

  gate (from W_q):  per-position sigmoid weight for each scale
  scale_interact:   small MLP mixes information across scales
  → weighted sum → out_proj → output
```

### Why not attention?

|                  | Self-attention              | CausalWaveConv              |
|------------------|-----------------------------|-----------------------------|
| Complexity       | O(L²)                       | O(L log L)                  |
| Long-range       | Direct token comparison     | Global spectral filtering   |
| Adaptivity       | Per-token Q/K/V             | Per-input kernel modulation |
| Interpretability | Attention weight matrix     | Kernel frequency spectrum   |

The two mechanisms are **complementary** — attention excels at sparse, content-driven retrieval
while wavelet convolution excels at structured temporal patterns. Combining them is an open research
direction; currently the architecture uses CausalWaveConv only.

---

## Architecture

Each `Block` stacks a CausalWaveConv mixer with a feed-forward network:

```
  Token IDs
      │
      ▼
 ┌─────────────────────────────────────────┐
 │              Embedding                  │
 └───────────────────┬─────────────────────┘
                     │
         ┌───────────▼───────────┐
         │        Block × N      │
         │  ┌─────────────────┐  │
         │  │  CausalWaveConv │  │  ← sequence mixing
         │  └────────┬────────┘  │
         │  ┌────────▼────────┐  │
         │  │      Norm       │  │  ← RMS / Layer / Dual / Group / Power
         │  └────────┬────────┘  │
         │  ┌────────▼────────┐  │
         │  │  FeedForward    │  │  ← dense SwiGLU  or  MoE
         │  └─────────────────┘  │
         └───────────┬───────────┘
                     │
      ┌──────────────▼──────────────┐
      │          LM Head            │
      └─────────────────────────────┘
                     │
                  logits
```

**MoE option:** a router dispatches each token to a subset of expert FFNs. An auxiliary
load-balancing loss encourages uniform expert utilization during training.

---

## Installation

```bash
pip install -r requirements.txt

# Optional: for HuggingFace tokenizers / sentencepiece
pip install sentencepiece tokenizers transformers
```

---

## Quick Start

**Train a model**

```bash
python main.py --config data/configuration/config.yaml --label my_experiment
```

**Generate text from a checkpoint**

```bash
python pipeline/inference.py "Your prompt here" --config data/configuration/config.yaml
```

**Count parameters**

```bash
python utils/calculate_params.py --config data/configuration/config.yaml
```

---

## Configuration

All configuration lives in `data/configuration/config.yaml`, loaded by `load_config()` from
`data/configuration/config.py`. Key sections:

| Section         | Controls                                                  |
|-----------------|-----------------------------------------------------------|
| `model`         | depth, dim, heads, scales, norm type, FFN type            |
| `training`      | optimizer, scheduler, mixed precision, grad checkpointing |
| `training_data` | path to tokenized parquet files                           |
| `tokenizer`     | tokenizer type and vocabulary                             |
| `generation`    | sampling parameters (temperature, top-k, top-p)           |
| `checkpoint`    | output directory, save frequency                          |
| `wandb`         | experiment tracking                                       |

---

## Repository Structure

```
suara/
├── main.py                          # Training entrypoint
├── pipeline/
│   ├── model.py                     # SuaRA model definition
│   ├── training.py                  # Training loop
│   └── inference.py                 # Tokenizer wrapper + generation
├── arc/
│   ├── block.py                     # Block (WaveConv + FFN)
│   ├── waveconv.py                  # CausalWaveConv implementation
│   ├── ffn.py                       # SwiGLU and MoE feed-forwards
│   └── normalizer.py                # Normalization variants
├── data/configuration/
│   ├── config.yaml                  # Default configuration
│   └── config.py                    # Config loader
├── utils/
│   └── calculate_params.py          # Parameter counter
└── artifacts/
    └── checkpoints/                 # best.pt, last.pt, exported bundles
```

---

## Checkpoints & Inference Bundles

Training saves two checkpoints under `artifacts/checkpoints/` (or the configured
`output_directory`):

- `best.pt` — lowest validation loss checkpoint
- `last.pt` — most recent checkpoint

Inference bundles are an exportable format containing `model_state_dict`, `model_config`, and
`tokenizer_config` in a single file, loadable by `InferenceTokenizer` in `pipeline/inference.py`.

---

## Data & Tokenization

Training data is expected as tokenized parquet files. Set the path via
`TrainingDataConfig.train_tokens_path` in `config.yaml`. Tokenizer configuration is co-located in
the same config file and loaded automatically at training and inference time.

---

## Contributing

- Keep `config.yaml` defaults updated when adding new options.
- Add example runs and loss curves to `artifacts/plots/`.
- Run linters and tests before opening PRs.

---

## License

See the `LICENSE` file in the repository root.
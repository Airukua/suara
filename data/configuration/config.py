from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "configuration" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"isi config harus berupa mapping/dict: {path}")
    return data


def _resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


@dataclass(slots=True)
class ModelConfig:
    dim: int = 512
    n_layers: int = 6
    n_wave_heads: int = 4
    n_scales: int = 4
    sigma_scales: list[float] | None = None
    ff_mult: float = 8 / 3
    dropout: float = 0.1
    max_seq: int = 512
    ffn_mode: str = "moe"
    norm_type: str = "rms"
    norm_kwargs: dict[str, Any] = field(default_factory=dict)
    gradient_checkpointing: bool = False
    num_experts: int = 8
    active_experts: int = 2
    aux_loss_coef: float = 0.01
    compile_mode: str | None = None

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_wave_heads": self.n_wave_heads,
            "n_scales": self.n_scales,
            "sigma_scales": self.sigma_scales,
            "ff_mult": self.ff_mult,
            "dropout": self.dropout,
            "max_seq": self.max_seq,
            "ffn_mode": self.ffn_mode,
            "norm_type": self.norm_type,
            "norm_kwargs": self.norm_kwargs,
            "gradient_checkpointing": self.gradient_checkpointing,
            "num_experts": self.num_experts,
            "active_experts": self.active_experts,
            "aux_loss_coef": self.aux_loss_coef,
            "compile_mode": self.compile_mode,
        }


@dataclass(slots=True)
class TrainingConfig:
    optimizer_name: str = field(default="adamw", init=False)
    max_steps: int = 5_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 250
    decay_start_step: int | None = None
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"
    log_every_steps: int = 20
    eval_every_steps: int = 200
    eval_max_batches: int | None = 50
    sample_every_steps: int = 200
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    restore_best_model: bool = True


@dataclass(slots=True)
class TrainingDataConfig:
    train_tokens_path: str = "data/tokens/train/tokens/tokenized.parquet"
    validation_tokens_path: str = "data/tokens/validation/tokens/tokenized.parquet"
    test_tokens_path: str = "data/tokens/test/tokens/tokenized.parquet"
    batch_size: int = 8
    eval_batch_size: int = 8
    shuffle_train: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    seed: int = 42


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 120
    temperature: float = 0.9
    top_k: int | None = 40
    top_p: float | None = 0.95
    repetition_penalty: float = 1.05


@dataclass(slots=True)
class TokenizerConfig:
    tokenizer_type: str = "sentencepiece"
    model_path: str = "data/tokens/train/tokenizer/sp_tokenizer.model"
    add_bos: bool = True
    add_eos: bool = True


@dataclass(slots=True)
class CheckpointConfig:
    enabled: bool = True
    output_directory: str = "artifacts/checkpoints"
    save_best: bool = True
    save_last: bool = True


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = False
    project: str = "suara"
    name: str | None = None
    entity: str | None = None
    mode: str = "online"
    log_model: bool = False


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    training_data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(path: str | None = None) -> AppConfig:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    raw = _read_yaml(config_path)

    model_data = dict(raw.get("model", {}))
    training_data = dict(raw.get("training", {}))
    training_input_data = dict(raw.get("training_data", {}))
    tokenizer_data = dict(raw.get("tokenizer", {}))
    checkpoint_data = dict(raw.get("checkpoint", {}))

    optimizer_name = training_data.pop("optimizer", None)
    if optimizer_name is not None and str(optimizer_name).lower() != "adamw":
        raise ValueError("optimizer yang didukung hanya AdamW")

    for key in ("train_tokens_path", "validation_tokens_path", "test_tokens_path"):
        if key in training_input_data:
            training_input_data[key] = _resolve_path(training_input_data[key])
    if "model_path" in tokenizer_data:
        tokenizer_data["model_path"] = _resolve_path(tokenizer_data["model_path"])
    if "output_directory" in checkpoint_data:
        checkpoint_data["output_directory"] = _resolve_path(checkpoint_data["output_directory"])

    return AppConfig(
        model=ModelConfig(**model_data),
        training=TrainingConfig(**training_data),
        training_data=TrainingDataConfig(**training_input_data),
        generation=GenerationConfig(**raw.get("generation", {})),
        tokenizer=TokenizerConfig(**tokenizer_data),
        checkpoint=CheckpointConfig(**checkpoint_data),
        wandb=WandbConfig(**raw.get("wandb", {})),
    )

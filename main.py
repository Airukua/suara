from __future__ import annotations
import argparse
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import pyarrow.parquet as pq
import torch
from data.configuration.config import load_config
from pipeline.inference import InferenceTokenizer
from pipeline.model import SuaRA
from pipeline.training import train
from utils.calculate_params import count_params


PROJECT_ROOT = Path(__file__).resolve().parent


class PackedTokenBatchLoader:
    def __init__(
        self,
        parquet_dir: str,
        batch_size: int,
        sequence_length: int,
        pad_token_id: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        read_batch_size: int = 128,
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.read_batch_size = read_batch_size
        self._epoch = 0
        self._part_files = sorted(self.parquet_dir.glob("part-*.parquet"))
        if not self._part_files:
            raise FileNotFoundError(f"part parquet tidak ditemukan di: {self.parquet_dir}")
        self._total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in self._part_files)
        self._source_sequence_length = self._infer_source_sequence_length()

    def _infer_source_sequence_length(self) -> int:
        first_file = pq.ParquetFile(self._part_files[0])
        first_batch = next(first_file.iter_batches(batch_size=1, columns=["input_ids"]))
        return len(first_batch.column(0)[0].as_py())

    def __len__(self) -> int:
        chunk_factor = max(1, self._source_sequence_length // self.sequence_length)
        estimated_samples = self._total_rows * chunk_factor
        return math.ceil(estimated_samples / self.batch_size)

    def _iter_chunks(self):
        rng = random.Random(self.seed + self._epoch)
        part_files = list(self._part_files)
        if self.shuffle:
            rng.shuffle(part_files)

        for part_file in part_files:
            parquet_file = pq.ParquetFile(part_file)
            for record_batch in parquet_file.iter_batches(
                batch_size=self.read_batch_size,
                columns=["input_ids", "attention_mask"],
            ):
                token_rows = record_batch.column(0).to_pylist()
                mask_rows = record_batch.column(1).to_pylist()
                rows = list(zip(token_rows, mask_rows))
                if self.shuffle:
                    rng.shuffle(rows)

                for token_ids, attention_mask in rows:
                    valid_tokens = int(sum(attention_mask))
                    if valid_tokens < 2:
                        continue
                    active_ids = token_ids[:valid_tokens]
                    max_start = len(active_ids) - 1
                    for start in range(0, max_start, self.sequence_length):
                        chunk = active_ids[start:start + self.sequence_length]
                        if len(chunk) < 2:
                            continue
                        yield chunk

        self._epoch += 1

    def _collate(self, chunks: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        padded_ids = []
        padded_mask = []
        for chunk in chunks:
            trimmed = chunk[:self.sequence_length]
            pad_size = self.sequence_length - len(trimmed)
            padded_ids.append(trimmed + [self.pad_token_id] * pad_size)
            padded_mask.append([1] * len(trimmed) + [0] * pad_size)

        token_tensor = torch.tensor(padded_ids, dtype=torch.long)
        mask_tensor = torch.tensor(padded_mask, dtype=torch.long)
        inputs = token_tensor[:, :-1]
        labels = token_tensor[:, 1:].clone()
        labels[mask_tensor[:, 1:] == 0] = -100
        return inputs, labels

    def __iter__(self):
        batch_buffer: list[list[int]] = []
        for chunk in self._iter_chunks():
            batch_buffer.append(chunk)
            if len(batch_buffer) == self.batch_size:
                yield self._collate(batch_buffer)
                batch_buffer = []

        if batch_buffer and not self.drop_last:
            yield self._collate(batch_buffer)


def _parse_args():
    parser = argparse.ArgumentParser(description="Training entrypoint untuk model SUARA.")
    parser.add_argument("--config", type=str, default=None, help="Path config YAML.")
    parser.add_argument("--label", type=str, default=None, help="Label run training.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
        help=(
            "Resume training dari checkpoint. "
            "Bisa berupa path file, atau keyword: last, best, auto. "
            "Jika --resume dipanggil tanpa nilai, default ke last."
        ),
    )
    return parser.parse_args()


def _resolve_run_label(label: str | None) -> str:
    if label:
        return label
    return f"suara_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_resume_checkpoint_path(resume_arg: str | None, checkpoint_dir: str) -> Path | None:
    if resume_arg is None:
        return None

    normalized = resume_arg.strip().lower()
    base_dir = Path(checkpoint_dir)
    default_candidates = {
        "last": [base_dir / "last.pt"],
        "best": [base_dir / "best.pt"],
        "auto": [base_dir / "last.pt", base_dir / "best.pt"],
    }
    if normalized in default_candidates:
        for candidate in default_candidates[normalized]:
            if candidate.exists():
                return candidate
        candidate_list = ", ".join(str(path) for path in default_candidates[normalized])
        raise FileNotFoundError(
            f"checkpoint resume mode '{resume_arg}' tidak ditemukan. "
            f"Sudah cek: {candidate_list}"
        )

    checkpoint_path = Path(resume_arg).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint resume tidak ditemukan: {checkpoint_path}")
    return checkpoint_path


def _load_resume_checkpoint(
    resume_arg: str | None,
    checkpoint_dir: str,
    device: torch.device,
) -> tuple[dict | None, Path | None]:
    checkpoint_path = _resolve_resume_checkpoint_path(resume_arg, checkpoint_dir)
    if checkpoint_path is None:
        return None, None
    return torch.load(checkpoint_path, map_location=device), checkpoint_path


def _build_sample_prompt_from_val(
    val_loader: PackedTokenBatchLoader,
    tokenizer: InferenceTokenizer,
    max_prompt_tokens: int = 32,
) -> str | None:
    for part_file in val_loader._part_files:
        parquet_file = pq.ParquetFile(part_file)
        for record_batch in parquet_file.iter_batches(
            batch_size=1,
            columns=["input_ids", "attention_mask"],
        ):
            token_ids = record_batch.column(0)[0].as_py()
            attention_mask = record_batch.column(1)[0].as_py()
            valid_tokens = int(sum(attention_mask))
            if valid_tokens < 2:
                continue

            active_ids = token_ids[:valid_tokens]
            prompt_len = min(max_prompt_tokens, max(8, len(active_ids) // 2))
            prompt_text = tokenizer.decode(active_ids[:prompt_len], skip_special_tokens=True).strip()
            if prompt_text:
                return prompt_text
    return None


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    run_label = _resolve_run_label(args.label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_float32_matmul_precision("high")

    tokenizer = InferenceTokenizer.from_config(cfg.tokenizer)
    model = SuaRA(
        vocab_size=len(tokenizer),
        **cfg.model.to_kwargs(),
    )

    train_loader = PackedTokenBatchLoader(
        parquet_dir=cfg.training_data.train_tokens_path,
        batch_size=cfg.training_data.batch_size,
        sequence_length=cfg.model.max_seq + 1,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=cfg.training_data.shuffle_train,
        drop_last=cfg.training_data.drop_last,
        seed=cfg.training_data.seed,
    )
    val_loader = PackedTokenBatchLoader(
        parquet_dir=cfg.training_data.validation_tokens_path,
        batch_size=cfg.training_data.eval_batch_size,
        sequence_length=cfg.model.max_seq + 1,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        drop_last=False,
        seed=cfg.training_data.seed,
    )

    resume_checkpoint, resume_checkpoint_path = _load_resume_checkpoint(
        args.resume,
        cfg.checkpoint.output_directory,
        device,
    )
    total_params, trainable_params = count_params(model)

    print(f"Training run: {run_label}")
    print(f"Device      : {device}")
    print(f"Vocab size  : {len(tokenizer):,}")
    print(f"Train batch : {cfg.training_data.batch_size}")
    print(f"Eval batch  : {cfg.training_data.eval_batch_size}")
    print(f"Max seq     : {cfg.model.max_seq}")
    print(f"Params      : total={total_params:,} trainable={trainable_params:,}")
    print(f"Train data  : {cfg.training_data.train_tokens_path}")
    print(f"Val data    : {cfg.training_data.validation_tokens_path}")
    if resume_checkpoint_path is not None:
        resumed_step = int(resume_checkpoint.get("step", 0))
        resumed_label = resume_checkpoint.get("label", "-")
        print(f"Resume ckpt : {resume_checkpoint_path}")
        print(f"Resume meta : label={resumed_label} step={resumed_step:,}")

    sample_prompt = _build_sample_prompt_from_val(val_loader, tokenizer)
    if sample_prompt:
        print(f"Sample prompt (from val): {sample_prompt}")
    else:
        print("Sample prompt (from val): tidak ditemukan, sample generation dinonaktifkan")

    history = train(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        device=device,
        label=run_label,
        train_config=cfg.training,
        generation_config=cfg.generation,
        tokenizer=tokenizer,
        sample_prompt=sample_prompt,
        wandb_config=cfg.wandb,
        model_config=asdict(cfg.model),
        checkpoint_config=cfg.checkpoint,
        resume_checkpoint=resume_checkpoint,
    )

    print(
        "Training selesai. "
        f"best_val_loss={history.get('best_val_loss')} "
        f"best_step={history.get('best_step')}"
    )


if __name__ == "__main__":
    sys.exit(main())

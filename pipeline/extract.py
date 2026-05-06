from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.configuration.config import load_config


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ekstrak checkpoint training menjadi paket inferensi yang ringan."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path ke file config YAML. Default pakai data/configuration/config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path checkpoint sumber. Default: best.pt lalu fallback ke last.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Folder output paket inferensi. Default: artifacts/exported/<nama-checkpoint>/",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(checkpoint_arg: str | None, checkpoint_dir: str) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint tidak ditemukan: {checkpoint_path}")
        return checkpoint_path

    base_dir = Path(checkpoint_dir)
    for candidate in (base_dir / "best.pt", base_dir / "last.pt"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"checkpoint default tidak ditemukan di {base_dir}. "
        "Gunakan --checkpoint untuk menentukan file checkpoint."
    )


def _prepare_output_dir(output_arg: str | None, checkpoint_path: Path) -> Path:
    if output_arg is not None:
        output_dir = Path(output_arg).expanduser()
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
    else:
        output_dir = (PROJECT_ROOT / "artifacts" / "exported" / checkpoint_path.stem).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_tree(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _bundle_tokenizer(tokenizer_config, output_dir: Path) -> dict[str, Any]:
    tokenizer_type = tokenizer_config.tokenizer_type
    source_path = Path(tokenizer_config.model_path).expanduser()
    bundle_path: str

    if tokenizer_type == "sentencepiece":
        if not source_path.exists():
            raise FileNotFoundError(f"model tokenizer tidak ditemukan: {source_path}")
        target_path = _copy_file(source_path, output_dir / "tokenizer" / source_path.name)
        bundle_path = str(target_path.relative_to(output_dir))
    elif tokenizer_type in {"hf", "pretrained"}:
        if source_path.exists():
            if source_path.is_dir():
                target_dir = _copy_tree(source_path, output_dir / "tokenizer")
                bundle_path = str(target_dir.relative_to(output_dir))
            else:
                target_path = _copy_file(source_path, output_dir / "tokenizer" / source_path.name)
                bundle_path = str(target_path.relative_to(output_dir))
        else:
            if tokenizer_type == "pretrained":
                # Jika model_path berupa nama repo HF, simpan identifier apa adanya.
                bundle_path = tokenizer_config.model_path
            else:
                raise FileNotFoundError(f"asset tokenizer tidak ditemukan: {source_path}")
    else:
        raise ValueError(f"tokenizer_type tidak didukung: {tokenizer_type}")

    return {
        "tokenizer_type": tokenizer_type,
        "model_path": bundle_path,
        "add_bos": tokenizer_config.add_bos,
        "add_eos": tokenizer_config.add_eos,
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, cfg.checkpoint.output_directory)
    output_dir = _prepare_output_dir(args.output, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint tidak memiliki key 'model_state_dict'")

    bundled_tokenizer_config = _bundle_tokenizer(cfg.tokenizer, output_dir)

    bundle = {
        "format": "suara_inference_bundle",
        "source_checkpoint": str(checkpoint_path),
        "label": checkpoint.get("label"),
        "step": int(checkpoint.get("step", 0)),
        "best_step": int(checkpoint.get("best_step", 0)),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "model_config": asdict(cfg.model),
        "tokenizer_config": bundled_tokenizer_config,
        "generation_config": asdict(cfg.generation),
        "model_state_dict": checkpoint["model_state_dict"],
    }

    bundle_path = output_dir / "bundle.pt"
    torch.save(bundle, bundle_path)

    manifest = {
        "format": bundle["format"],
        "bundle_file": bundle_path.name,
        "source_checkpoint": bundle["source_checkpoint"],
        "label": bundle["label"],
        "step": bundle["step"],
        "best_step": bundle["best_step"],
        "best_val_loss": bundle["best_val_loss"],
        "tokenizer_type": bundled_tokenizer_config["tokenizer_type"],
        "tokenizer_model_path": bundled_tokenizer_config["model_path"],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"Checkpoint sumber : {checkpoint_path}")
    print(f"Paket inferensi   : {bundle_path}")
    print(f"Manifest          : {output_dir / 'manifest.json'}")
    print(f"Tokenizer bundled : {bundled_tokenizer_config['model_path']}")
    print("Selesai. Paket ini hanya menyimpan komponen penting untuk inferensi.")


if __name__ == "__main__":
    main()

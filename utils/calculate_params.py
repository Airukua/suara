from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.configuration.config import AppConfig, load_config
from pipeline.inference import InferenceTokenizer
from pipeline.model import SuaRA


def count_params(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_model_from_config(config: AppConfig) -> tuple[SuaRA, InferenceTokenizer]:
    tokenizer = InferenceTokenizer.from_config(config.tokenizer)
    model = SuaRA(
        vocab_size=len(tokenizer),
        **config.model.to_kwargs(),
    )
    return model, tokenizer


def get_parameter_report(config_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    model, tokenizer = build_model_from_config(config)
    total_params, trainable_params = count_params(model)
    non_trainable_params = total_params - trainable_params

    return {
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "model_class": model.__class__.__name__,
        "vocab_size": len(tokenizer),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "model_config": config.model.to_kwargs(),
    }


def format_parameter_report(report: dict[str, Any]) -> str:
    return (
        f"Model           : {report['model_class']}\n"
        f"Vocab size      : {report['vocab_size']:,}\n"
        f"Total params    : {report['total_params']:,}\n"
        f"Trainable params: {report['trainable_params']:,}\n"
        f"Frozen params   : {report['non_trainable_params']:,}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Hitung jumlah parameter model SUARA dari config.")
    parser.add_argument("--config", type=str, default=None, help="Path config YAML.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report dalam format JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = get_parameter_report(args.config)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(format_parameter_report(report))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    return list(values)


def _ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _safe_run_name(run_name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name.strip())
    return cleaned or "training_run"


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.25)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_train_loss(history: dict[str, Any], output_dir: str | Path) -> Path | None:
    steps = _as_list(history.get("train_steps"))
    losses = _as_list(history.get("train_loss"))
    if not steps or not losses:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#1f77b4", linewidth=2, label="Train Loss")
    _style_axis(ax, "Train Loss", "Step", "Loss")
    ax.legend()

    path = _ensure_dir(output_dir) / "train_loss.png"
    _save_figure(fig, path)
    return path


def plot_grad_norm(history: dict[str, Any], output_dir: str | Path) -> Path | None:
    steps = _as_list(history.get("grad_norm_steps"))
    values = _as_list(history.get("grad_norm"))
    if not steps or not values:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, color="#ff7f0e", linewidth=2, label="Grad Norm")
    _style_axis(ax, "Gradient Norm", "Step", "Grad Norm")
    ax.legend()

    path = _ensure_dir(output_dir) / "grad_norm.png"
    _save_figure(fig, path)
    return path


def plot_validation_metrics(history: dict[str, Any], output_dir: str | Path) -> Path | None:
    steps = _as_list(history.get("steps"))
    val_loss = _as_list(history.get("val_loss"))
    val_ppl = _as_list(history.get("val_ppl"))
    if not steps or (not val_loss and not val_ppl):
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    if val_loss:
        axes[0].plot(steps, val_loss, color="#d62728", linewidth=2, marker="o", label="Validation Loss")
        best_val_loss = history.get("best_val_loss")
        best_step = history.get("best_step")
        if best_val_loss is not None and best_step:
            axes[0].scatter([best_step], [best_val_loss], color="#2ca02c", s=70, label="Best")
        axes[0].legend()
    _style_axis(axes[0], "Validation Loss", "Step", "Loss")

    if val_ppl:
        axes[1].plot(steps, val_ppl, color="#9467bd", linewidth=2, marker="o", label="Validation Perplexity")
        axes[1].legend()
    _style_axis(axes[1], "Validation Perplexity", "Step", "Perplexity")

    path = _ensure_dir(output_dir) / "validation_metrics.png"
    _save_figure(fig, path)
    return path


def plot_learning_rate(history: dict[str, Any], output_dir: str | Path) -> Path | None:
    steps = _as_list(history.get("learning_rate_steps"))
    values = _as_list(history.get("learning_rates"))
    if not steps or not values:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, color="#8c564b", linewidth=2, label="Learning Rate")
    _style_axis(ax, "Learning Rate Schedule", "Step", "LR")
    ax.legend()

    path = _ensure_dir(output_dir) / "learning_rate.png"
    _save_figure(fig, path)
    return path


def plot_training_overview(history: dict[str, Any], output_dir: str | Path) -> Path | None:
    train_steps = _as_list(history.get("train_steps"))
    train_loss = _as_list(history.get("train_loss"))
    grad_steps = _as_list(history.get("grad_norm_steps"))
    grad_norm = _as_list(history.get("grad_norm"))
    eval_steps = _as_list(history.get("steps"))
    val_loss = _as_list(history.get("val_loss"))
    val_ppl = _as_list(history.get("val_ppl"))
    if not any([train_steps, grad_steps, eval_steps]):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if train_steps and train_loss:
        axes[0, 0].plot(train_steps, train_loss, color="#1f77b4", linewidth=2)
    _style_axis(axes[0, 0], "Train Loss", "Step", "Loss")

    if grad_steps and grad_norm:
        axes[0, 1].plot(grad_steps, grad_norm, color="#ff7f0e", linewidth=2)
    _style_axis(axes[0, 1], "Gradient Norm", "Step", "Grad Norm")

    if eval_steps and val_loss:
        axes[1, 0].plot(eval_steps, val_loss, color="#d62728", linewidth=2, marker="o")
    _style_axis(axes[1, 0], "Validation Loss", "Step", "Loss")

    if eval_steps and val_ppl:
        axes[1, 1].plot(eval_steps, val_ppl, color="#9467bd", linewidth=2, marker="o")
    _style_axis(axes[1, 1], "Validation Perplexity", "Step", "Perplexity")

    path = _ensure_dir(output_dir) / "training_overview.png"
    _save_figure(fig, path)
    return path


def plot_elapsed_time(history: dict[str, Any], output_dir: str | Path) -> Path | None:
    steps = _as_list(history.get("steps"))
    elapsed = _as_list(history.get("elapsed_times"))
    if not steps or not elapsed:
        return None

    elapsed_minutes = [value / 60.0 for value in elapsed]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, elapsed_minutes, color="#2ca02c", linewidth=2, marker="o", label="Elapsed Time")
    _style_axis(ax, "Elapsed Time", "Step", "Minutes")
    ax.legend()

    path = _ensure_dir(output_dir) / "elapsed_time.png"
    _save_figure(fig, path)
    return path


def save_training_history(history: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = _ensure_dir(output_dir) / "history.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    return output_path


def save_training_plots(
    history: dict[str, Any],
    output_dir: str | Path,
    run_name: str = "training_run",
) -> dict[str, str]:
    base_dir = _ensure_dir(output_dir) / _safe_run_name(run_name)
    base_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "history": str(save_training_history(history, base_dir)),
    }

    for name, builder in (
        ("train_loss", plot_train_loss),
        ("grad_norm", plot_grad_norm),
        ("validation_metrics", plot_validation_metrics),
        ("learning_rate", plot_learning_rate),
        ("training_overview", plot_training_overview),
        ("elapsed_time", plot_elapsed_time),
    ):
        plot_path = builder(history, base_dir)
        if plot_path is not None:
            outputs[name] = str(plot_path)

    return outputs

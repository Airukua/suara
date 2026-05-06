from __future__ import annotations
from typing import Iterable, Sequence
import torch

def validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value

def validate_non_negative_float(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value

def validate_probability(name: str, value: float, *, upper_open: bool = True) -> float:
    if upper_open:
        valid = 0.0 <= value < 1.0
        bound = "[0, 1)"
    else:
        valid = 0.0 <= value <= 1.0
        bound = "[0, 1]"
    if not valid:
        raise ValueError(f"{name} must be in {bound}, got {value}")
    return value

def validate_choice(name: str, value: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    return value

def validate_divisible(name: str, value: int, divisor: int, detail: str | None = None) -> int:
    if value % divisor != 0:
        suffix = f" ({detail})" if detail else ""
        raise ValueError(f"{name} must be divisible by {divisor}, got {value}{suffix}")
    return value

def validate_sequence_length(name: str, values: Sequence[object], expected: int) -> Sequence[object]:
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(values)}")
    return values


def validate_less_equal(name: str, value: int, upper: int) -> int:
    if value > upper:
        raise ValueError(f"{name} must be <= {upper}, got {value}")
    return value


def validate_tensor_rank(x: torch.Tensor, expected_rank: int, name: str = "tensor") -> torch.Tensor:
    if x.dim() != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got shape {tuple(x.shape)}")
    return x

def validate_tensor_last_dim(x: torch.Tensor, expected_dim: int, name: str = "tensor") -> torch.Tensor:
    if x.size(-1) != expected_dim:
        raise ValueError(
            f"{name} must have trailing dim {expected_dim}, got shape {tuple(x.shape)}"
        )
    return x

def validate_tensor_shape(x: torch.Tensor, expected: Sequence[int], name: str = "tensor") -> torch.Tensor:
    if tuple(x.shape) != tuple(expected):
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(x.shape)}")
    return x

def validate_finite_tensor(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return x

def validate_optional_dict(name: str, value: dict | None) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict or None, got {type(value).__name__}")
    return value

def validate_all_finite_tensors(pairs: Iterable[tuple[str, torch.Tensor]]) -> None:
    for name, tensor in pairs:
        validate_finite_tensor(tensor, name)

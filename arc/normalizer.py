import torch
import torch.nn as nn

from utils.sanity_check import (
    validate_choice,
    validate_divisible,
    validate_finite_tensor,
    validate_positive_int,
    validate_tensor_last_dim,
    validate_tensor_rank,
)

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        validate_positive_int("dim", dim)
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.norm.normalized_shape[0], "x")
        return validate_finite_tensor(self.norm(x), "LayerNorm output")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        validate_positive_int("dim", dim)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.scale.numel(), "x")
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return validate_finite_tensor((x / rms) * self.scale, "RMSNorm output")

class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        validate_positive_int("dim", dim)
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(dim ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return validate_finite_tensor(x * (self.scale / norm), "ScaleNorm output")

class CRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        validate_positive_int("dim", dim)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.scale.numel(), "x")
        x = x - x.mean(dim=-1, keepdim=True)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return validate_finite_tensor((x / rms) * self.scale, "CRMSNorm output")

class GroupNorm(nn.Module):
    def __init__(self, dim: int, n_groups: int = 4, eps: float = 1e-6):
        super().__init__()
        validate_positive_int("dim", dim)
        validate_positive_int("n_groups", n_groups)
        validate_divisible("dim", dim, n_groups, "for GroupNorm")
        self.norm = nn.GroupNorm(n_groups, dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.norm.num_channels, "x")
        B, L, D = x.shape
        x = x.transpose(1, 2)      
        x = self.norm(x)
        return validate_finite_tensor(x.transpose(1, 2), "GroupNorm output")

class PowerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, momentum: float = 0.1):
        super().__init__()
        validate_positive_int("dim", dim)
        self.eps = eps
        self.momentum = momentum
        self.scale = nn.Parameter(torch.ones(dim))
        self.register_buffer("running_qmean", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.scale.numel(), "x")
        if self.training:
            qmean = x.pow(2).mean(dim=[0, 1])  # (D,)
            self.running_qmean = (
                (1 - self.momentum) * self.running_qmean
                + self.momentum * qmean.detach()
            )
            norm = qmean.add(self.eps).sqrt()
        else:
            norm = self.running_qmean.add(self.eps).sqrt()
        return validate_finite_tensor((x / norm) * self.scale, "PowerNorm output")

class NormFree(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        validate_positive_int("dim", dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        return validate_finite_tensor(x, "NormFree output")

class DualNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        validate_positive_int("dim", dim)
        self.pre  = RMSNorm(dim, eps)
        self.post = LayerNorm(dim, eps)

    def forward(self, x: torch.Tensor,
                post: bool = False) -> torch.Tensor:
        validate_tensor_rank(x, 3, "x")
        return self.post(x) if post else self.pre(x)

NORM_REGISTRY = {
    "layer":    LayerNorm,
    "rms":      RMSNorm,
    "scale":    ScaleNorm,
    "crms":     CRMSNorm,
    "group":    GroupNorm,
    "power":    PowerNorm,
    "none":     NormFree,
    "dual":     DualNorm,
}


def get_norm(name: str, dim: int, **kwargs) -> nn.Module:
    name = name.lower()
    validate_choice("name", name, tuple(NORM_REGISTRY.keys()))
    return NORM_REGISTRY[name](dim, **kwargs)


def list_norms() -> list[str]:
    return list(NORM_REGISTRY.keys())


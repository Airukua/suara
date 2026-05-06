import torch
import torch.nn as nn
from typing import Literal, Optional
from .ffn import FeedForward
from .normalizer import get_norm
from .waveconv import CausalWaveConv
from utils.sanity_check import (
    validate_choice,
    validate_finite_tensor,
    validate_optional_dict,
    validate_probability,
    validate_tensor_last_dim,
    validate_tensor_rank,
)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_type: str,
        ffn_mode: Literal["dense", "moe"],
        n_wave_heads: int = 4,
        n_scales: int = 4,
        sigma_scales: Optional[list[float]] = None,
        dropout: float = 0.0,
        compile_mode: Optional[str] = None,
        max_seq: Optional[int] = None,
        norm_kwargs: Optional[dict] = None,
        ff_mult: float = 8 / 3,
        num_experts: int = 8,
        active_experts: int = 2,
        aux_loss_coef: float = 0.01,
        enable_runtime_checks: bool = True,
    ):
        super().__init__()
        norm_kwargs = validate_optional_dict("norm_kwargs", norm_kwargs)
        validate_choice("ffn_mode", ffn_mode, ("dense", "moe"))
        validate_probability("dropout", dropout)

        self.norm_type = norm_type.lower()
        self.ffn_mode = ffn_mode
        self.enable_runtime_checks = enable_runtime_checks

        self.waveconv = CausalWaveConv(
            dim=dim,
            n_wave_heads=n_wave_heads,
            n_scales=n_scales,
            sigma_scales=sigma_scales,
            dropout=dropout,
            compile_mode=compile_mode,
            max_seq=max_seq,
            enable_runtime_checks=enable_runtime_checks,
        )
        self.norm1 = get_norm(self.norm_type, dim, **norm_kwargs)
        self.norm2 = get_norm(self.norm_type, dim, **norm_kwargs)
        self.ffn = FeedForward(
            dim=dim,
            mode=ffn_mode,
            ff_mult=ff_mult,
            dropout=dropout,
            num_experts=num_experts,
            active_experts=active_experts,
            aux_loss_coef=aux_loss_coef,
        )
        self.dropout = nn.Dropout(dropout)

    def _apply_norm(self, norm: nn.Module, x: torch.Tensor, post: bool = False) -> torch.Tensor:
        if self.norm_type == "dual":
            out = norm(x, post=post)
        else:
            out = norm(x)
        if self.enable_runtime_checks:
            validate_finite_tensor(out, f"{self.norm_type} norm output")
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.enable_runtime_checks:
            validate_tensor_rank(x, 3, "x")
            validate_tensor_last_dim(x, self.waveconv.dim, "x")
        wave_in = self._apply_norm(self.norm1, x, post=False)
        x = x + self.dropout(self.waveconv(wave_in))
        if self.enable_runtime_checks:
            validate_finite_tensor(x, "Block residual after waveconv")
        if self.norm_type == "dual":
            x = self._apply_norm(self.norm1, x, post=True)

        ffn_in = self._apply_norm(self.norm2, x, post=False)
        ffn_out, aux_loss = self.ffn(ffn_in)
        x = x + self.dropout(ffn_out)
        if self.enable_runtime_checks:
            validate_finite_tensor(x, "Block residual after ffn")
        if self.norm_type == "dual":
            x = self._apply_norm(self.norm2, x, post=True)

        if self.enable_runtime_checks and aux_loss is not None:
            validate_finite_tensor(aux_loss, "Block aux_loss")
        return x, aux_loss

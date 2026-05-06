import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional, Literal
from utils.sanity_check import (
    validate_choice,
    validate_finite_tensor,
    validate_less_equal,
    validate_positive_int,
    validate_probability,
    validate_tensor_last_dim,
    validate_tensor_rank,
)

class SwiGLU(nn.Module):
    def __init__(self, dim, ff_mult=8/3, dropout=0.0):
        super().__init__()
        validate_positive_int("dim", dim)
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")
        validate_probability("dropout", dropout)

        hidden = int(dim * ff_mult)
        hidden = (hidden + 63) // 64 * 64
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() < 2:
            raise ValueError(f"x must have rank >= 2, got shape {tuple(x.shape)}")
        validate_tensor_last_dim(x, self.down.out_features, "x")
        out = self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))
        return validate_finite_tensor(out, "SwiGLU output")

class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        active_experts: int = 2, 
        ff_mult: float = 8/3,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        validate_positive_int("dim", dim)
        validate_positive_int("num_experts", num_experts)
        validate_positive_int("active_experts", active_experts)
        validate_less_equal("active_experts", active_experts, num_experts)
        validate_probability("dropout", dropout)
        self.dim = dim
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.aux_loss_coef = aux_loss_coef
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(dim, ff_mult=ff_mult, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.dim, "x")
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        router_logits = self.router(x_flat) 
        weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1), 
            self.active_experts, 
            dim=-1
        ) 

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i).any(dim=-1) 
            if mask.any():
                expert_input = x_flat[mask]
                expert_out = expert(expert_input)
                expert_idx_in_topk = (selected_experts[mask] == i).nonzero(as_tuple=True)[1]
                w = weights[mask].gather(1, expert_idx_in_topk.unsqueeze(1)).squeeze(1)
                output[mask] += w.unsqueeze(1) * expert_out

        router_prob = F.softmax(router_logits, dim=-1).mean(dim=0)  # [num_experts]
        expert_fraction = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )
        for i in range(self.num_experts):
            mask = (selected_experts == i).any(dim=-1).float()
            expert_fraction[i] = mask.mean()
        aux_loss = self.num_experts * (router_prob * expert_fraction).sum()
        aux_loss = self.aux_loss_coef * aux_loss
        output = output.view(B, S, D)
        validate_finite_tensor(output, "MoE output")
        validate_finite_tensor(aux_loss, "MoE aux_loss")
        return output, aux_loss
    

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: Literal["dense", "moe"] = "moe",
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        num_experts: int = 8,
        active_experts: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        validate_positive_int("dim", dim)
        validate_choice("mode", mode, ("dense", "moe"))
        validate_probability("dropout", dropout)
        self.mode = mode

        if mode == "dense":
            self.layer = SwiGLU(dim, ff_mult=ff_mult, dropout=dropout)
            self.num_experts = 1
            self.active_experts = 1
        elif mode == "moe":
            self.layer = MoE(
                dim=dim,
                num_experts=num_experts,
                active_experts=active_experts,
                ff_mult=ff_mult,
                dropout=dropout,
                aux_loss_coef=aux_loss_coef,
            )
            self.num_experts = num_experts
            self.active_experts = active_experts
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        validate_tensor_rank(x, 3, "x")
        if self.mode == "dense":
            out = self.layer(x)
            return validate_finite_tensor(out, "FeedForward[dense] output"), None
        else:
            out, aux_loss = self.layer(x)
            validate_finite_tensor(out, "FeedForward[moe] output")
            if aux_loss is not None:
                validate_finite_tensor(aux_loss, "FeedForward[moe] aux_loss")
            return out, aux_loss

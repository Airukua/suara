import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from arc.block import Block
from arc.normalizer import get_norm
from utils.sanity_check import (
    validate_choice,
    validate_finite_tensor,
    validate_optional_dict,
    validate_positive_int,
    validate_probability,
    validate_tensor_rank,
)

class SuaRA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 6,
        n_wave_heads: int = 4,
        n_scales: int = 4,
        sigma_scales=None,
        ff_mult: float = 8 / 3,
        dropout: float = 0.1,
        max_seq: int = 512,
        ffn_mode: str = "moe",
        norm_type: str = "rms",
        norm_kwargs: dict | None = None,
        gradient_checkpointing: bool = False,
        num_experts: int = 8,
        active_experts: int = 2,
        aux_loss_coef: float = 0.01,
        compile_mode: str | None = None,
        enable_runtime_checks: bool = True,
    ):
        super().__init__()
        validate_positive_int("vocab_size", vocab_size)
        validate_positive_int("dim", dim)
        validate_positive_int("n_layers", n_layers)
        validate_positive_int("n_wave_heads", n_wave_heads)
        validate_positive_int("n_scales", n_scales)
        validate_positive_int("max_seq", max_seq)
        validate_probability("dropout", dropout)
        validate_choice("ffn_mode", ffn_mode, ("dense", "moe"))
        norm_kwargs = validate_optional_dict("norm_kwargs", norm_kwargs)

        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.gradient_checkpointing = gradient_checkpointing
        self.norm_type = norm_type
        self.enable_runtime_checks = enable_runtime_checks
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    norm_type=norm_type,
                    ffn_mode=ffn_mode,
                    n_wave_heads=n_wave_heads,
                    n_scales=n_scales,
                    sigma_scales=sigma_scales,
                    compile_mode=compile_mode,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    max_seq=max_seq,
                    norm_kwargs=norm_kwargs,
                    num_experts=num_experts,
                    active_experts=active_experts,
                    aux_loss_coef=aux_loss_coef,
                    enable_runtime_checks=enable_runtime_checks,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_final = get_norm(norm_type, dim, **norm_kwargs)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _forward_block(self, block, x):
        if self.training and self.gradient_checkpointing:
            def custom_forward(hidden_states):
                return block(hidden_states)

            return checkpoint(custom_forward, x, use_reentrant=True)
        return block(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _maybe_validate_finite(self, x, name: str):
        if self.enable_runtime_checks:
            validate_finite_tensor(x, name)
        return x

    def forward(self, token_ids, labels=None):
        if self.enable_runtime_checks:
            validate_tensor_rank(token_ids, 2, "token_ids")
        if token_ids.size(1) > self.max_seq:
            raise ValueError(
                f"sequence length {token_ids.size(1)} exceeds max_seq {self.max_seq}"
            )
        if self.enable_runtime_checks and (
            token_ids.min().item() < 0 or token_ids.max().item() >= self.vocab_size
        ):
            raise ValueError("token_ids contain values outside the embedding vocabulary range")

        if labels is not None:
            if labels.shape != token_ids.shape:
                raise ValueError(
                    f"labels shape must match token_ids shape, got {tuple(labels.shape)} "
                    f"and {tuple(token_ids.shape)}"
                )
            invalid_mask = (labels != -100) & ((labels < 0) | (labels >= self.vocab_size))
            if self.enable_runtime_checks and invalid_mask.any().item():
                raise ValueError(
                    "labels contain values outside the valid class range or ignore_index=-100"
                )

        x = self.embedding(token_ids)
        self._maybe_validate_finite(x, "embedding output")
        aux_loss = x.new_zeros(())
        for block in self.blocks:
            x, block_aux_loss = self._forward_block(block, x)
            if block_aux_loss is not None:
                aux_loss = aux_loss + block_aux_loss
            self._maybe_validate_finite(x, "hidden states after block")

        if self.norm_type == "dual":
            x = self.norm_final(x, post=True)
        else:
            x = self.norm_final(x)
        self._maybe_validate_finite(x, "final norm output")
        logits = self.lm_head(x)
        self._maybe_validate_finite(logits, "logits")
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + aux_loss
        return logits, loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from utils.sanity_check import (
    validate_finite_tensor, 
    validate_positive_int,
    validate_probability,
    validate_sequence_length,
    validate_tensor_last_dim,
    validate_tensor_rank,
)


def _causal_shift_spectrum(H: torch.Tensor, L_padded: int) -> torch.Tensor:
    h_time = torch.fft.irfft(H, n=L_padded, dim=-1)
    causal = torch.zeros(L_padded, device=H.device, dtype=h_time.dtype)
    causal[0] = 1.0
    causal[1 : L_padded // 2 + 1] = 2.0
    if L_padded % 2 == 0:
        causal[L_padded // 2] = 1.0
    h_causal = h_time * causal[: h_time.shape[-1]]
    return torch.fft.rfft(h_causal, n=L_padded, dim=-1)


_BUCKET_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _bucket_length(L: int) -> int:
    doubled = L * 2
    for b in _BUCKET_SIZES:
        if b >= doubled:
            return b
    return 2 ** math.ceil(math.log2(doubled))


class CausalWaveConv(nn.Module):
    def __init__(
        self,
        dim: int,
        n_wave_heads: int = 4,
        n_scales: int = 4,
        sigma_scales: Optional[list[float]] = None,
        dropout: float = 0.0,
        compile_mode: Optional[str] = None,
        max_seq: Optional[int] = None,
        enable_runtime_checks: bool = True,
        use_fixed_fft_in_eval: bool = False,
    ):
        super().__init__()
        validate_positive_int("dim", dim)
        validate_positive_int("n_wave_heads", n_wave_heads)
        validate_positive_int("n_scales", n_scales)
        validate_probability("dropout", dropout)
        if dim % n_wave_heads != 0:
            raise ValueError(f"dim must be divisible by n_wave_heads, got dim={dim}, n_wave_heads={n_wave_heads}")

        self.dim = dim
        self.H = n_wave_heads
        self.K = n_scales
        self.head_dim = dim // n_wave_heads
        self.enable_runtime_checks = enable_runtime_checks
        self.use_fixed_fft_in_eval = use_fixed_fft_in_eval
        self._fixed_L_total: Optional[int] = None
        if max_seq is not None:
            validate_positive_int("max_seq", max_seq)
            self._fixed_L_total = _bucket_length(max_seq)

        if sigma_scales is None:
            sigma_scales = [1.0 * (4.0 ** k) for k in range(n_scales)]
        validate_sequence_length("sigma_scales", sigma_scales, n_scales)

        self.register_buffer(
            "sigma_scales", torch.tensor(sigma_scales, dtype=torch.float32)
        )

        self.log_amp = nn.Parameter(torch.zeros(n_wave_heads, n_scales))
        self.mu_shift = nn.Parameter(
            torch.stack(
                [torch.full((n_wave_heads,), float(k) * 0.5) for k in range(n_scales)],
                dim=1,
            )
        )
        self.head_freq_bias = nn.Parameter(
            torch.randn(n_wave_heads, n_scales) * 0.01
        )

        self.W_q = nn.Linear(dim, n_wave_heads * n_scales, bias=False)
        nn.init.normal_(self.W_q.weight, std=0.02)

        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self._kernel_cache: dict[tuple, torch.Tensor] = {}

        self._compiled_conv = None
        if compile_mode is not None:
            self._compiled_conv = torch.compile(
                self._fft_conv_forward, mode=compile_mode, fullgraph=True
            )

    def _build_kernel(self, L_total: int, device: torch.device) -> torch.Tensor:
        if not self.training:
            cache_key = (L_total, str(device))
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]

        n_freq = L_total // 2 + 1
        dtype = torch.float32

        omega = (
            torch.arange(n_freq, device=device, dtype=dtype)
            * (2 * math.pi / L_total)
        ).view(1, 1, n_freq)

        sigma = self.sigma_scales.view(1, self.K, 1)
        gauss = torch.exp(-0.5 * sigma**2 * omega**2)
        amp = torch.exp(self.log_amp).unsqueeze(-1)
        mu = (self.mu_shift + self.head_freq_bias).unsqueeze(-1)
        phase = -mu * omega.expand(self.H, self.K, n_freq)
        H_freq = amp * gauss * torch.polar(
            torch.ones(self.H, self.K, n_freq, device=device, dtype=dtype),
            phase,
        )

        H_flat = H_freq.reshape(self.H * self.K, n_freq)
        H_causal_flat = _causal_shift_spectrum(H_flat, L_total)
        H_causal = H_causal_flat.reshape(self.H, self.K, n_freq)

        if not self.training:
            cache_key = (L_total, str(device))
            self._kernel_cache[cache_key] = H_causal.detach()

        if self.enable_runtime_checks:
            validate_finite_tensor(H_causal, "waveconv kernel")
        return H_causal

    def _fft_conv_forward(
        self,
        v: torch.Tensor,
        kernel: torch.Tensor,
        gate: torch.Tensor,
        L_pad: int,
        L_total: int,
    ) -> torch.Tensor:
        B, H, L, Dh = v.shape
        v_t = v.permute(0, 1, 3, 2).contiguous()
        v_padded = F.pad(v_t, (L_pad, 0))
        v_freq = torch.fft.rfft(v_padded, n=L_total, dim=-1)
        k = kernel.to(dtype=v_freq.dtype)
        k_exp = k.unsqueeze(0).unsqueeze(3)
        v_exp = v_freq.unsqueeze(2)

        per_scale_freq = v_exp * k_exp
        per_scale_time = torch.fft.irfft(
            per_scale_freq, n=L_total, dim=-1
        )[..., L_pad:]

        gate_exp = gate.unsqueeze(3)
        weighted = (per_scale_time * gate_exp).sum(dim=2)

        return weighted.permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_runtime_checks:
            validate_tensor_rank(x, 3, "x")
            validate_tensor_last_dim(x, self.dim, "x")
        B, L, D = x.shape
        gate = torch.sigmoid(self.W_q(x))
        gate = gate.view(B, L, self.H, self.K)
        gate = gate.permute(0, 2, 3, 1)
        v = self.W_v(x).view(B, L, self.H, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        if self._fixed_L_total is not None and (self.training or self.use_fixed_fft_in_eval):
            L_total = self._fixed_L_total
        else:
            L_total = _bucket_length(L)
        L_pad = L_total - L
        kernel = self._build_kernel(L_total, x.device)
        conv_fn = self._compiled_conv if self._compiled_conv is not None else self._fft_conv_forward
        out = conv_fn(v.float(), kernel, gate.float(), L_pad, L_total)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, L, D).to(x.dtype)
        out = self.drop(out)
        out = self.out_proj(out)
        if self.enable_runtime_checks:
            validate_finite_tensor(out, "CausalWaveConv output")
        return out

    def invalidate_kernel_cache(self):
        self._kernel_cache.clear()

    @torch.no_grad()
    def get_superposition_coherence(self, L: int = 512) -> float:
        L_total = self._fixed_L_total if self._fixed_L_total is not None else _bucket_length(L)
        device = next(self.parameters()).device
        kernel = self._build_kernel(L_total, device)
        sum_k = kernel.sum(dim=1).abs() ** 2
        sum_sq = (kernel.abs() ** 2).sum(dim=1)
        coherence = sum_k / (self.K * sum_sq + 1e-8)
        return coherence.mean().item()

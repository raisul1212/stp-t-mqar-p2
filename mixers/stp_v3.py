"""
STP-Transformer v3 (White Paper v3.0) — Zoology-compatible mixers
RISE Lab, Purdue University  |  March 2026

Two variants as specified in the white paper:
  - STPTLight  : scalar-per-head two-factor retention (Section 2.4)
  - STPT       : per-column  two-factor retention    (Section 2.5)

════════════════════════════════════════════════════════════════════════════════
ZOOLOGY CALLING CONVENTION
════════════════════════════════════════════════════════════════════════════════

Zoology's ModelConfig/ModuleConfig instantiates sequence mixers as:

    Mixer(d_model, **kwargs)

where d_model is passed as the FIRST POSITIONAL ARGUMENT.
Any extra kwargs from ModuleConfig.kwargs are merged in.
The forward() method receives:

    output = mixer(x)   # x: Tensor[B, T, d_model]

and must return a raw Tensor[B, T, d_model] — NOT a tuple.

════════════════════════════════════════════════════════════════════════════════
RECURRENCE (per-column STP-T, Eq. 14 in white paper)
════════════════════════════════════════════════════════════════════════════════

    gamma_j(t)  = sigmoid(W_gamma_j * k_j(t) + b_gamma_j)
    lambda_j(t) = sigmoid(W_lambda_j * k_j(t) + b_lambda_j)
    rho_j(t)    = (1 - lambda_j(t)) * gamma_j(t)     <- two-factor product
    S_ij(t)     = rho_j * S_ij(t-1) + (1-rho_j) * v_i(t) * k_j(t)  <- convex
    y(t)        = q(t) . S(t)

For STP-T-Light (scalar, Section 2.4):
    k_bar_h(t)  = mean_j( k_j(t) )
    gamma_h, lambda_h computed from k_bar_h
    rho_h broadcast to full (dk, dk) block

════════════════════════════════════════════════════════════════════════════════
INITIALIZATION (from Phase-1 learnings)
════════════════════════════════════════════════════════════════════════════════

W_LTM = 0; biases set so sigmoid outputs → gamma~0.9, lambda~0.1 at init.
This gives rho ~ 0.81 as prior: strong context retention, not saturated.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class STPTLight(nn.Module):
    """
    STP-T-Light: scalar-per-head two-factor retention (WP Section 2.4).

    Retention overhead: 4H scalars.
    Input to gates: mean(k_h(t)) — scalar per head.

    Zoology calling convention: STPTLight(d_model, num_heads=2, ...)
    Returns: Tensor[B, T, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        gamma_init: float = 0.9,
        lambda_init: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        logit_g = math.log(gamma_init  / (1.0 - gamma_init))
        logit_l = math.log(lambda_init / (1.0 - lambda_init))

        H = num_heads
        self.W_gamma  = nn.Parameter(torch.zeros(H))
        self.b_gamma  = nn.Parameter(torch.full((H,), logit_g))
        self.W_lambda = nn.Parameter(torch.zeros(H))
        self.b_lambda = nn.Parameter(torch.full((H,), logit_l))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def extra_repr(self):
        return (f"d_model={self.d_model}, num_heads={self.num_heads}, "
                f"d_k={self.d_k}, retention=scalar_per_head")

    def forward(self, x, *args, **kwargs):
        B, T, _ = x.shape
        H, dk   = self.num_heads, self.d_k

        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=H)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=H)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=H)
        k = F.normalize(k, dim=-1)

        k_bar = k.mean(dim=-1)                              # (B, H, T)
        Wg = self.W_gamma .view(1, H, 1)
        bg = self.b_gamma .view(1, H, 1)
        Wl = self.W_lambda.view(1, H, 1)
        bl = self.b_lambda.view(1, H, 1)
        gamma = torch.sigmoid(Wg * k_bar + bg)              # (B, H, T)
        lam   = torch.sigmoid(Wl * k_bar + bl)
        rho   = (1.0 - lam) * gamma

        S = x.new_zeros(B, H, dk, dk)
        outputs = []
        for t in range(T):
            q_t   = q[:, :, t, :]
            k_t   = k[:, :, t, :]
            v_t   = v[:, :, t, :]
            rho_t = rho[:, :, t].unsqueeze(-1).unsqueeze(-1)   # (B,H,1,1)
            outer = torch.einsum("bhv,bhk->bhvk", v_t, k_t)
            S     = rho_t * S + (1.0 - rho_t) * outer
            y_t   = torch.einsum("bhq,bhqk->bhk", q_t, S)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)


class STPT(nn.Module):
    """
    STP-T: per-column two-factor retention (WP Section 2.5).

    Retention overhead: 4 * d_k * H = 4 * d_model scalars (~0.39% at d=256).
    Input to gates: k_j(t) — element-wise, no extra projection.

    Zoology calling convention: STPT(d_model, num_heads=2, ...)
    Returns: Tensor[B, T, d_model]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        gamma_init: float = 0.9,
        lambda_init: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        dk             = self.d_k

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        logit_g = math.log(gamma_init  / (1.0 - gamma_init))
        logit_l = math.log(lambda_init / (1.0 - lambda_init))

        H = num_heads
        self.W_gamma  = nn.Parameter(torch.zeros(H, dk))
        self.b_gamma  = nn.Parameter(torch.full((H, dk), logit_g))
        self.W_lambda = nn.Parameter(torch.zeros(H, dk))
        self.b_lambda = nn.Parameter(torch.full((H, dk), logit_l))

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def extra_repr(self):
        return (f"d_model={self.d_model}, num_heads={self.num_heads}, "
                f"d_k={self.d_k}, retention=per_column")

    def forward(self, x, *args, **kwargs):
        B, T, _ = x.shape
        H, dk   = self.num_heads, self.d_k

        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=H)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=H)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=H)
        k = F.normalize(k, dim=-1)

        # Per-column LTM weights: (H,dk) → (1,H,1,dk)
        Wg = self.W_gamma .unsqueeze(0).unsqueeze(2)
        bg = self.b_gamma .unsqueeze(0).unsqueeze(2)
        Wl = self.W_lambda.unsqueeze(0).unsqueeze(2)
        bl = self.b_lambda.unsqueeze(0).unsqueeze(2)

        gamma = torch.sigmoid(Wg * k + bg)   # (B, H, T, dk)
        lam   = torch.sigmoid(Wl * k + bl)
        rho   = (1.0 - lam) * gamma

        S = x.new_zeros(B, H, dk, dk)
        outputs = []
        for t in range(T):
            q_t     = q[:, :, t, :]
            k_t     = k[:, :, t, :]
            v_t     = v[:, :, t, :]
            rho_col = rho[:, :, t, :].unsqueeze(-2)     # (B,H,1,dk)
            outer   = torch.einsum("bhv,bhk->bhvk", v_t, k_t)
            S       = rho_col * S + (1.0 - rho_col) * outer
            y_t     = torch.einsum("bhq,bhqk->bhk", q_t, S)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)

"""Diagonal Linear Recurrent Unit (LRU) core — drop-in replacement for nn.LSTM in TD3.
(full rationale: docs/decision_history.md#--legacy-lru_core-py-1)"""

import torch
import torch.nn as nn


class LRUCore(nn.Module):
    """Input-independent diagonal recurrence h_t = lam*h_{t-1} + gamma*(W_in x_t).
    (full rationale: docs/decision_history.md#--legacy-lru_core-decay-matrix-scan)"""

    def __init__(self, d_model, r_min=0.60, r_max=0.999):
        super().__init__()
        self.d_model = d_model
        # lam = exp(-exp(nu_log)) is in (0,1) for any real nu_log, so the recurrence is
        # unconditionally stable and |h| is bounded — no cell-state saturation is possible.
        lam = r_min + (r_max - r_min) * torch.rand(d_model)
        self.nu_log = nn.Parameter(torch.log(-torch.log(lam)))
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _decay(self):
        a = torch.exp(self.nu_log)                       # a > 0, lam = exp(-a)
        lam = torch.exp(-a)
        gamma = torch.sqrt(1.0 - lam * lam + 1e-8)       # unit-variance input gain (LRU norm)
        return a, lam, gamma

    def initial_hidden(self, batch, device=None):
        return torch.zeros(batch, self.d_model, device=device)

    def forward(self, x, hidden=None):
        """x: [B, T, D] -> out [B, T, D], h_last [B, D]. Matches step() to ~1e-7."""
        B, T, _ = x.shape
        a, lam, gamma = self._decay()
        if hidden is None:
            hidden = self.initial_hidden(B, device=x.device)

        # Materialise the causal decay matrix K[d,t,i] = lam_d^(t-i) for t>=i and contract
        # in one einsum. Costs (D,T,T) = 460k floats at our shapes, vs (B,T,D,N) for an
        # S4/Mamba scan — and needs no exp(+cumsum), so it cannot overflow.
        idx = torch.arange(T, device=x.device, dtype=x.dtype)
        pw = idx[:, None] - idx[None, :]                 # (T,T)
        K = torch.exp(-a[:, None, None] * pw.clamp(min=0)) * (pw >= 0)
        u = self.in_proj(x) * gamma
        hs = torch.einsum('bih,hti->bth', u, K)
        if hidden is not None:
            hs = hs + torch.exp(-a[None, None, :] * (idx[None, :, None] + 1.0)) * hidden[:, None, :]

        out = self.norm(self.out_proj(hs) + x)           # residual keeps the block near-identity at init
        return out, hs[:, -1]

    def step(self, x_t, hidden):
        """Single-step O(1) rollout path. x_t: [B, D], hidden: [B, D]."""
        _, lam, gamma = self._decay()
        h = lam * hidden + gamma * self.in_proj(x_t)
        return self.norm(self.out_proj(h) + x_t), h

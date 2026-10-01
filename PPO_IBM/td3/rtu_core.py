"""Recurrent Trace Unit (RTU) and GRU cores with the LRUCore interface, for TD3_cores.py.

RTU (Elelimy et al., NeurIPS 2024, "Real-Time Recurrent Learning using Trace Units in
Reinforcement Learning"): a complex-valued diagonal recurrence written as two real vectors,
    h1_t = r cos(th) h1_{t-1} - r sin(th) h2_{t-1} + gamma W1 x_t
    h2_t = r cos(th) h2_{t-1} + r sin(th) h1_{t-1} + gamma W2 x_t
with r = exp(-exp(nu_log)), th = exp(theta_log), gamma = sqrt(1 - r^2), and the nonlinearity
applied after the recurrence (the paper's "Linear RTU"). lru_core.LRUCore is the special case
th = 0: a real diagonal, which cannot represent oscillating (complex-eigenvalue) memory. The
paper trains RTUs with RTRL; here they are trained by BPTT over the replay window like every
other core, so only the architecture is being compared.
"""
import math

import torch
import torch.nn as nn


class RTUCore(nn.Module):
    def __init__(self, d_model, r_min=0.60, r_max=0.999, theta_max=math.pi / 10):
        super().__init__()
        assert d_model % 2 == 0, "RTU splits the state into two real halves"
        self.d_model, self.n = d_model, d_model // 2
        r = r_min + (r_max - r_min) * torch.rand(self.n)
        self.nu_log = nn.Parameter(torch.log(-torch.log(r)))
        # Small initial phases, as in the LRU paper's ring initialisation.
        self.theta_log = nn.Parameter(torch.log(theta_max * torch.rand(self.n).clamp(min=1e-3)))
        self.in_proj = nn.Linear(d_model, d_model, bias=False)     # rows [W1; W2]
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _params(self):
        a = torch.exp(self.nu_log)                      # r = exp(-a) in (0, 1): always stable
        th = torch.exp(self.theta_log)
        gamma = torch.sqrt(1.0 - torch.exp(-2.0 * a) + 1e-8)
        return a, th, gamma

    def initial_hidden(self, batch, device=None):
        return torch.zeros(batch, self.d_model, device=device)

    def _read(self, h, x):
        return self.norm(self.out_proj(torch.relu(h)) + x)

    def forward(self, x, hidden=None):
        """x: [B, T, D] -> out [B, T, D], h_last [B, D] (state = [h1; h2])."""
        B, T, _ = x.shape
        if hidden is None:
            hidden = self.initial_hidden(B, device=x.device)
        if T == 1:
            y, h = self.step(x[:, 0], hidden)
            return y.unsqueeze(1), h
        a, th, gamma = self._params()
        n = self.n
        idx = torch.arange(T, device=x.device, dtype=x.dtype)
        pw = idx[:, None] - idx[None, :]                               # (T, T), t - i
        mag = torch.exp(-a[:, None, None] * pw.clamp(min=0)) * (pw >= 0)
        Kr = mag * torch.cos(th[:, None, None] * pw)                   # Re(lam^(t-i))
        Ki = mag * torch.sin(th[:, None, None] * pw)                   # Im(lam^(t-i))
        u = self.in_proj(x)
        u1, u2 = u[..., :n] * gamma, u[..., n:] * gamma
        h1 = torch.einsum('bih,hti->bth', u1, Kr) - torch.einsum('bih,hti->bth', u2, Ki)
        h2 = torch.einsum('bih,hti->bth', u2, Kr) + torch.einsum('bih,hti->bth', u1, Ki)
        k = idx[None, :, None] + 1.0                                   # carry-in: lam^(t+1) h0
        m0 = torch.exp(-a[None, None, :] * k)
        c0, s0 = m0 * torch.cos(th[None, None, :] * k), m0 * torch.sin(th[None, None, :] * k)
        g1, g2 = hidden[:, None, :n], hidden[:, None, n:]
        hs = torch.cat([h1 + c0 * g1 - s0 * g2, h2 + c0 * g2 + s0 * g1], dim=-1)
        return self._read(hs, x), hs[:, -1]

    def step(self, x_t, hidden):
        a, th, gamma = self._params()
        n = self.n
        r = torch.exp(-a)
        g, phi = r * torch.cos(th), r * torch.sin(th)
        u = self.in_proj(x_t)
        h1 = g * hidden[:, :n] - phi * hidden[:, n:] + gamma * u[:, :n]
        h2 = g * hidden[:, n:] + phi * hidden[:, :n] + gamma * u[:, n:]
        h = torch.cat([h1, h2], dim=-1)
        return self._read(h, x_t), h


class GRUCore(nn.Module):
    """nn.GRU behind the same interface (hidden is [B, D])."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRU(d_model, d_model, 1, batch_first=True)

    def initial_hidden(self, batch, device=None):
        return torch.zeros(batch, self.d_model, device=device)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.initial_hidden(x.shape[0], device=x.device)
        out, h = self.gru(x, hidden.unsqueeze(0).contiguous())
        return out, h[0]

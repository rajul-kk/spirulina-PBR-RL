"""LayerNorm LSTM (Ba, Kiros & Hinton 2016) behind nn.LSTM's call interface, for TD3_CORE_LN=1.

Layer norm on the input-to-gate and hidden-to-gate pre-activations (separately, each over the
4H gate vector) and on the cell state before the output tanh:
    gates = LN_ih(W_ih x_t) + LN_hh(W_hh h_{t-1})
    c_t   = sig(f) * c_{t-1} + sig(i) * tanh(g)
    h_t   = sig(o) * tanh(LN_c(c_t))
so |h_t| < 1 whatever the cell state does, and the gate pre-activations are re-centred every step.
That targets the LSTM cell-state saturation failure (decision_history #--legacy-TD3-py-107):
a saturated c cannot push the gates, because their inputs are normalised.
Single layer, batch_first, hidden = (h, c) each [1, B, H] -- exactly nn.LSTM(…, 1, batch_first=True).
"""
import torch
import torch.nn as nn


class LNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        assert num_layers == 1 and batch_first, "LNLSTM is single-layer, batch_first"
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        with torch.no_grad():
            self.ln_ih.bias[hidden_size:2 * hidden_size] = 1.0     # forget-gate bias, gate order i,f,g,o

    def forward(self, x, hidden=None):
        B, T, _ = x.shape
        H = self.hidden_size
        if hidden is None:
            z = torch.zeros(1, B, H, device=x.device, dtype=x.dtype)
            hidden = (z, z.clone())
        h, c = hidden[0][0], hidden[1][0]
        gi = self.ln_ih(self.ih(x))                       # input half for all T at once
        outs = []
        for t in range(T):
            gates = gi[:, t] + self.ln_hh(self.hh(h))
            i, f, g, o = gates.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(self.ln_c(c))
            outs.append(h)
        return torch.stack(outs, dim=1), (h.unsqueeze(0), c.unsqueeze(0))

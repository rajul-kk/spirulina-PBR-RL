import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearSSMBlock(nn.Module):
    """
    A simplified, pure-PyTorch Linear State Space Model (SSM) block.
    Inspired by S4/Mamba foundations, this module provides infinite receptive 
    field memory without the fixed-window constraint of 1D CNNs, while remaining
    highly parallelizable for training.
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 'auto'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        if dt_rank == 'auto':
            dt_rank = max(d_model // 16, 1)
        self.dt_rank = dt_rank

        # Discretization timescale parameters (∆)
        self.dt_proj = nn.Linear(d_model, dt_rank, bias=False)
        self.dt_expand = nn.Linear(dt_rank, d_model, bias=True)
        
        # State matrices: A (decay/transition), B (input projection), C (output projection)
        # Initialize A to be stable (negative real parts)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A)) # (d_model, d_state) - softplus applied later for positivity
        
        self.x_proj = nn.Linear(d_model, d_state + d_state, bias=False) # B and C
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_sequence(self, x: torch.Tensor, h_init: torch.Tensor = None) -> torch.Tensor:
        """
        Process a full sequence in bulk during training.
        Uses a parallelized scan alias for fast GPU training.
        
        Args:
            x: (Batch, SeqLen, d_model)
            h_init: Optional initial hidden state (Batch, d_model, d_state)
            
        Returns:
            out: (Batch, SeqLen, d_model)
            h_last: Final hidden state (Batch, d_model, d_state) to pass to next chunk
        """
        batch, seq_len, d_model = x.shape
        
        # 1. Compute state parameters
        dt = F.softplus(self.dt_expand(self.dt_proj(x))) # (B, L, D) Delta timescales
        
        # Ensure A is strictly positive (meaning -A is strictly negative/stable)
        A = -torch.exp(self.A_log) # (D, N) -> (d_model, d_state)
        
        # Project inputs to B and C
        bc = self.x_proj(x) # (B, L, d_state * 2)
        B, C = torch.split(bc, self.d_state, dim=-1) # (B, L, N)
        
        # 2. Continuous -> Discrete transition (Zero-order hold approximation)
        # ∆A = exp(∆ * A)
        dt_A = torch.einsum('bld,dn->bldn', dt, A) # (B, L, D, N)
        dA = torch.exp(dt_A) # (B, L, D, N)
        
        # ∆B = ∆ * B
        dB = torch.einsum('bld,bln->bldn', dt, B) # (B, L, D, N)
        
        # 3. Parallel Vectorized Scan (replaces the slow Python for-loop)
        # We want to compute: h_t = dA_t * h_{t-1} + dB_t * x_t
        # This is a first-order linear recurrence. We can solve it in parallel 
        # using the general formula:
        # h_t = \sum_{i=0}^t ( \prod_{j=i+1}^t dA_j ) * (dB_i * x_i)
        
        # Multiply dB and x to get the inputs at each step
        inputs = dB * x.unsqueeze(-1) # (B, L, D, N)
        
        # We need to compute the cumulative product of dA. 
        # Since dA is strictly positive (it's exp(something)), we can use cumsum in log-space 
        # for numerical stability and massive speedup.
        
        # log(dA) = dt_A
        # cumulative sum of log(dA)
        log_dA_cumsum = torch.cumsum(dt_A, dim=1) # (B, L, D, N)
        
        # The term \prod_{j=i+1}^t dA_j is equivalent to exp(log_dA_cumsum[t] - log_dA_cumsum[i])
        # So h_t = exp(log_dA_cumsum[t]) * \sum_{i=0}^t exp(-log_dA_cumsum[i]) * inputs_i
        
        # To avoid exp() exploding, we factor it out:
        # h_t = \sum_{i=0}^t exp( log_dA_cumsum[t] - log_dA_cumsum[i] + log(inputs_i) ) -> wait, inputs can be negative.
        # Better: h_t = exp(log_dA_cumsum[t]) * \cumsum( exp(-log_dA_cumsum_i) * inputs_i )
        
        # Shift log_dA_cumsum right by 1 to represent the product starting from index 0
        log_dA_cumsum_shifted = F.pad(log_dA_cumsum[:, :-1], (0, 0, 0, 0, 1, 0)) # (B, L, D, N)
        
        # Multiply inputs by the decay factor to that point
        decay_factor = torch.exp(-log_dA_cumsum_shifted) # (B, L, D, N)
        decayed_inputs = decay_factor * inputs
        
        # Cumulative sum of the decayed inputs
        cumsum_decayed_inputs = torch.cumsum(decayed_inputs, dim=1) # (B, L, D, N)
        
        # Multiply back by the total decay to get the final state at each step
        hs = torch.exp(log_dA_cumsum_shifted) * cumsum_decayed_inputs # (B, L, D, N)
        
        # Incorporate h_init if it exists
        if h_init is not None:
            # h_init decays by the full cumprod up to step t
            h_init_decayed = torch.exp(log_dA_cumsum) * h_init.unsqueeze(1)
            hs = hs + h_init_decayed
            
        h = hs[:, -1] # The final hidden state to pass to the next chunk
        
        # 4. Output projection
        # y(t) = C(t) * h(t)
        y = torch.einsum('bldn,bln->bld', hs, C) # (B, L, D)
        out = self.out_proj(y) # (B, L, D)
        
        return out, h
        
    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) Step function for environmental rollout inference.
        
        Args:
            x_t: (Batch, d_model) current observation embedding
            h_prev: (Batch, d_model, d_state) previous hidden state
            
        Returns:
            out_t: (Batch, d_model) output
            h_new: (Batch, d_model, d_state) updated hidden state
        """
        # 1. Compute state parameters for this single step
        dt = F.softplus(self.dt_expand(self.dt_proj(x_t))) # (B, D)
        A = -torch.exp(self.A_log) # (D, N)
        
        bc = self.x_proj(x_t) # (B, N * 2)
        B, C = torch.split(bc, self.d_state, dim=-1) # (B, N)
        
        # 2. Discretize
        dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A)) # (B, D, N)
        dB = torch.einsum('bd,bn->bdn', dt, B) # (B, D, N)
        
        # 3. State update
        h_new = dA * h_prev + dB * x_t.unsqueeze(-1) # (B, D, N)
        
        # 4. Output
        y = torch.einsum('bdn,bn->bd', h_new, C) # (B, D)
        out_t = self.out_proj(y) # (B, D)
        
        return out_t, h_new

"""Core-agnostic loading for TD3 actor checkpoints (LSTM or diagonal LRU).
(full rationale: docs/decision_history.md#--legacy-actor_io-py-1)"""

import torch


def detect_core(state_dict):
    """Identify the recurrent core from parameter names alone.

    Provenance is derived rather than stored, so this works on every checkpoint already
    written -- no format change and nothing to keep in sync.
    """
    keys = list(state_dict.keys())
    if any(k.endswith("nu_log") for k in keys):
        return "lru"
    if any("weight_ih_l0" in k for k in keys):
        return "lstm"
    raise ValueError(
        "Cannot identify the recurrent core of this checkpoint. Expected either "
        "'lstm.nu_log' (diagonal LRU) or 'lstm.weight_ih_l0' (nn.LSTM) among its keys.\n"
        f"Got: {keys[:12]}{' ...' if len(keys) > 12 else ''}"
    )


def load_actor(path, obs_dim=None, action_dim=None, device=None):
    """Build the actor class matching the checkpoint, load it, and return (actor, core)."""
    from TD3 import RecurrentActor, OBS_DIM, ACTION_DIM, DEVICE

    device = device or DEVICE
    state_dict = torch.load(path, map_location=device)
    core = detect_core(state_dict)

    if core == "lru":
        from TD3_lru import LRUActor as Cls
    else:
        Cls = RecurrentActor

    actor = Cls(obs_dim or OBS_DIM, action_dim or ACTION_DIM).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, core

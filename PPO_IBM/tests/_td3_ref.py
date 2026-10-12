"""Deterministic short td3_update sequence used to pin the default (no context-ablation flags)
training path. `run(core)` returns the actor's parameters and per-tensor critic sums after a
seeded run of N_UPDATES updates on synthetic episodes. The committed fixture
(fixtures/td3_default_ref.npz) was generated from the code at tag rl-protocol-v1's td3/TD3.py
(before the context-ablation flags were added); test_context_ablation.py compares against it.

  python tests/_td3_ref.py   # regenerate (only ever from a pre-ablation checkout)
"""
import os
import sys

import numpy as np
import torch

N_UPDATES = 8


def synthetic_episode(rng, n):
    return {"obs": rng.randn(n, 6).astype(np.float32), "action": np.clip(rng.randn(n, 3), -1, 1).astype(np.float32),
            "reward": rng.randn(n).astype(np.float32), "next_obs": rng.randn(n, 6).astype(np.float32),
            "done": np.zeros(n, dtype=np.float32)}


def run(core, td3_mod, lru_mod, n_updates=N_UPDATES, buffer_cls=None):
    import torch.optim as optim
    base = td3_mod
    torch.manual_seed(1234)
    np.random.seed(1234)
    rng = np.random.RandomState(99)
    Actor, Critic = (base.RecurrentActor, base.RecurrentCritic) if core == "lstm" else (lru_mod.LRUActor, lru_mod.LRUCritic)
    update = base.td3_update if core == "lstm" else lru_mod.td3_update
    actor, actor_t = Actor(base.OBS_DIM, base.ACTION_DIM), Actor(base.OBS_DIM, base.ACTION_DIM)
    actor_t.load_state_dict(actor.state_dict())
    critic, critic_t = Critic(base.OBS_DIM, base.ACTION_DIM), Critic(base.OBS_DIM, base.ACTION_DIM)
    critic_t.load_state_dict(critic.state_dict())
    for p in list(actor_t.parameters()) + list(critic_t.parameters()):
        p.requires_grad = False
    a_opt = optim.Adam(actor.parameters(), lr=base.LR_ACTOR)
    c_opt = optim.Adam(critic.parameters(), lr=base.LR_CRITIC)
    demo = base.SequenceReplayBuffer(8, 60)
    online = base.SequenceReplayBuffer(20, 60)
    for _ in range(8):
        demo.add(synthetic_episode(rng, 700))
    for _ in range(6):
        online.add(synthetic_episode(rng, 700))
    for i in range(n_updates):
        update(actor, actor_t, critic, critic_t, a_opt, c_opt, demo, online, i)
    out = {f"actor/{k}": v.detach().numpy().copy() for k, v in actor.state_dict().items()}
    for k, v in critic.state_dict().items():
        out[f"critic_sum/{k}"] = np.array([v.double().sum().item(), v.double().abs().sum().item()])
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for p in (os.path.join(root, "td3"), os.path.join(root, "environments"), os.path.join(root, "training")):
        sys.path.insert(0, p)
    os.environ.setdefault("TD3_THREADS", "2")
    import TD3
    import TD3_lru
    ref = {}
    for core in ("lstm", "lru"):
        for k, v in run(core, TD3, TD3_lru).items():
            ref[f"{core}/{k}"] = v
    os.makedirs(os.path.join(here, "fixtures"), exist_ok=True)
    np.savez_compressed(os.path.join(here, "fixtures", "td3_default_ref.npz"), **ref)
    print("wrote", len(ref), "arrays")

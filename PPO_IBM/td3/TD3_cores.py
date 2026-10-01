"""TD3_cores.py — TD3 with a GRU or RTU recurrent core (exploratory arms; see
docs/reports/comparison_protocol.md section 6). Everything except the core is TD3_lru.py's:
the same networks, update, seeding and train loop.

  python td3/TD3_cores.py --core gru [--resume]
  python td3/TD3_cores.py --core rtu [--resume]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import TD3_lru as lru          # sets the thread count and imports TD3 as lru.base
from rtu_core import GRUCore, RTUCore

base = lru.base
CORES = {"gru": GRUCore, "rtu": RTUCore}


def make_classes(core):
    Core = CORES[core]

    class Actor(lru.LRUActor):
        def __init__(self, obs_dim, action_dim, hidden_dim=base.HIDDEN_DIM, lstm_layers=base.LSTM_LAYERS):
            super().__init__(obs_dim, action_dim, hidden_dim, lstm_layers)
            self.lstm = Core(hidden_dim)

    class Critic(lru.LRUCritic):
        def __init__(self, obs_dim, action_dim, hidden_dim=base.HIDDEN_DIM, lstm_layers=base.LSTM_LAYERS):
            super().__init__(obs_dim, action_dim, hidden_dim, lstm_layers)
            self.q1_lstm = Core(hidden_dim)
            self.q2_lstm = Core(hidden_dim)

    Actor.__name__, Critic.__name__ = f"{core.upper()}Actor", f"{core.upper()}Critic"
    return Actor, Critic


def install(core):
    lru.install()                                   # fused update, foreach soft update
    base.RecurrentActor, base.RecurrentCritic = make_classes(core)
    sfx = base.RUN_SUFFIX
    base.CHECKPOINT_DIR = f"model_data/td3_{core}_checkpoints{sfx}"
    base.STATE_PATH = f"model_data/td3_{core}_training_state{sfx}.pkl"
    base.BUFFER_PATH = f"model_data/td3_{core}_checkpoints{sfx}/online_buffer.pkl"
    base.BEST_CHECKPOINT_DIR = f"model_data/td3_{core}_checkpoints_best{sfx}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 + GRU or RTU core")
    parser.add_argument("--core", required=True, choices=sorted(CORES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    install(args.core)
    print(f"--- TD3+{args.core.upper()} | threads={torch.get_num_threads()} ---")
    base.train(resume=args.resume)

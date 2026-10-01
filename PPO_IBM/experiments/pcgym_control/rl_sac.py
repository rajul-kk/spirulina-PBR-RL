"""RL baseline on the PC-Gym task: Stable-Baselines3 SAC with default hyperparameters (the
algorithm family PC-Gym itself benchmarks), trained on fresh scenarios and scored on the final
split through the same episode code as every program.

The policy sees what a program sees (noisy Ca, T, setpoint) plus its previous action, stacked
over the last 4 samples so it can infer the unmeasured feed disturbances.

  python rl_sac.py --seed 1 --steps 100000 --out results/sac/s1
"""
import argparse
import json
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import SPLITS, summarise  # noqa: E402
from tasks import TASKS  # noqa: E402

STACK = 4
TRAIN_SEED_BASE = 7_000_000     # disjoint from search (3M), pilot (5M) and final (20M)


def features(obs, u_prev, task):
    return np.array([(obs["Ca"] - 0.88) / 0.02, (obs["T"] - 323.0) / 5.0, (obs["Ca_sp"] - 0.88) / 0.02,
                     (obs["Ca"] - obs["Ca_sp"]) / task.ERR_SCALE, (u_prev - 298.5) / 3.5], dtype=np.float32)


class Stacker:
    def __init__(self, task):
        self.task, self.buf, self.u_prev = task, deque(maxlen=STACK), 298.5

    def push(self, obs):
        f = features(obs, self.u_prev, self.task)
        while len(self.buf) < STACK - 1:
            self.buf.append(f)
        self.buf.append(f)
        return np.concatenate(self.buf)

    def to_u(self, a):
        self.u_prev = float(298.5 + 3.5 * np.clip(a, -1, 1).reshape(-1)[0])
        return self.u_prev


class TaskEnv(gym.Env):
    def __init__(self, task, seed):
        self.task, self.rng = task, np.random.RandomState(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (5 * STACK,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, seed=None, options=None):
        self.ep = self.task.start(TRAIN_SEED_BASE + int(self.rng.randint(10_000_000)))
        self.st = Stacker(self.task)
        return self.st.push(self.ep.obs()), {}

    def step(self, a):
        self.ep.step(self.st.to_u(a))
        r = -min(self.ep.last_sq, 100.0) / 10.0
        if self.ep.done:
            return np.zeros(5 * STACK, np.float32), r, False, True, {}
        return self.st.push(self.ep.obs()), r, False, False, {}


class SacController:
    def __init__(self, model, task):
        self.model, self.st = model, Stacker(task)

    def act(self, obs):
        a, _ = self.model.predict(self.st.push(obs), deterministic=True)
        return self.st.to_u(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cstr")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)
    task = TASKS[args.task]
    model_path = os.path.join(args.out, "sac.zip")
    if os.path.exists(model_path):
        model = SAC.load(model_path)
    else:
        model = SAC("MlpPolicy", TaskEnv(task, args.seed), seed=args.seed, verbose=0)
        model.learn(total_timesteps=args.steps)
        model.save(model_path)
    base, n = SPLITS["final"]
    eps = []
    for i in range(n):
        s, _ = task.run(SacController(model, task), base + i)
        eps.append(s)
    s = summarise(eps)
    print(f"sac seed {args.seed}: final mean cost {s['mean_cost']:.3f}, median {s['median_cost']:.3f}, "
          f"runaway {100 * s['runaway_rate']:.0f}%", flush=True)
    with open(os.path.join(args.out, "final.json"), "w") as f:
        json.dump({"summary": s, "episodes": eps, "train_steps": args.steps}, f, indent=1)


if __name__ == "__main__":
    main()

"""pytest bootstrap for PPO_IBM.

Puts the code directories on sys.path the way the scripts themselves do (td3/, environments/,
training/), and keeps everything CPU-only and offline. Run from PPO_IBM/:

    pytest                # fast suite (default, -m "not slow")
    pytest -m slow        # slow regressions only
    pytest -m ""          # everything
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # tests never touch a GPU
os.environ.setdefault("TD3_THREADS", "2")
os.environ.pop("TD3_SEED", None)                       # seeding at TD3 import would leak into other tests
os.environ.pop("TD3_HIDDEN_RESET_INTERVAL", None)

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)
for _p in (_TESTS, os.path.join(_ROOT, "td3"), os.path.join(_ROOT, "environments"),
           os.path.join(_ROOT, "training"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

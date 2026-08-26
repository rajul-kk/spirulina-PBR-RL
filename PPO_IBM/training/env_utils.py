"""Small shared helpers for unwrapping the VecEnv/Wrapper stack down to the raw env."""

# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-env_utils-py-3)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------


def unwrap_raw_env(env):
    """Unwrap VecNormalize/DummyVecEnv/Monitor/Wrapper chain down to the raw env."""
    raw = env
    if hasattr(raw, "venv"):
        raw = raw.venv
    if hasattr(raw, "envs"):
        raw = raw.envs[0]
    while hasattr(raw, "env"):
        raw = raw.env
    return raw

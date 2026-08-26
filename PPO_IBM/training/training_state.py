
# --- path bootstrap (added by _refactor_layout.py) -------------------------------------
# (full rationale: docs/decision_history.md#--training-training_state-py-2)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "environments")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------------------
import os
import pickle
from typing import Iterable, Optional, Tuple


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_state(file_path: str, state: dict) -> None:
    ensure_parent_dir(file_path)
    with open(file_path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(file_path: str) -> Optional[dict]:
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


def find_latest_checkpoint(checkpoint_dir: str, prefix: str, suffix: str) -> Tuple[Optional[str], Optional[int]]:
    if not os.path.isdir(checkpoint_dir):
        return None, None

    candidates = []
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        step_part = name[len(prefix):-len(suffix)]
        try:
            step = int(step_part)
        except ValueError:
            continue
        candidates.append((step, os.path.join(checkpoint_dir, name)))

    if not candidates:
        return None, None

    step, path = max(candidates, key=lambda item: item[0])
    return path, step


def replay_buffer_state(buffer, field_names: Iterable[str]) -> dict:
    return {
        "capacity": buffer.capacity,
        "idx": buffer.idx,
        "size": buffer.size,
        "fields": {name: getattr(buffer, name) for name in field_names},
    }


def restore_replay_buffer(buffer, state: Optional[dict]) -> None:
    if not state:
        return
    buffer.idx = int(state.get("idx", 0))
    buffer.size = int(state.get("size", 0))
    for name, value in state.get("fields", {}).items():
        setattr(buffer, name, value)

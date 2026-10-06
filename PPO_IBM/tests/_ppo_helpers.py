"""Shared helpers for the test suite (plain module so tests can import it directly)."""
import importlib.util
import os

PPO_IBM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(PPO_IBM, "experiments")
PC = os.path.join(EXP, "program_control")
PG = os.path.join(EXP, "pcgym_control")
PC_FINAL = os.path.join(PC, "results", "final")
PG_FINAL = os.path.join(PG, "results", "final")
TD3_DIR = os.path.join(PPO_IBM, "td3")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runner.py")


def load_module(unique_name, path):
    """Import a script by path under a unique name (several scripts share names like
    'harness' or 'compare', so they must not go through sys.path / sys.modules)."""
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

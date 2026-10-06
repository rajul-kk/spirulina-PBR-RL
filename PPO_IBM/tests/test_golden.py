"""Golden regressions: re-run each committed analysis script on the committed result jsons/logs
and compare with the committed compare.txt / td3_secondary.txt.

The scripts run in a child process (tests/_runner.py), their stdout is captured, and
td3_secondary.py's unconditional write to results/final/td3_secondary.txt is redirected to tmp_path,
so nothing under results/ is ever written.

Two layers:
  * fast (default): bootstrap size cut to 300, so everything that does not depend on the bootstrap
    draws (tables, point estimates, "better in x% of episodes", exact permutation p) must match
    byte for byte and the bootstrap CI / p_holm fields are masked;
  * slow: full N_BOOT=20000, byte-exact, including the CIs and Holm p-values.

The committed program_control/compare.txt was written before the exploratory td3_gru / td3_rtu arms
had results; those rows are the only expected difference and are filtered out of the fresh output
(generically: any arm absent from the committed file). Likewise td3_secondary.txt predates the
exploratory gru/rtu cores, so only the lru/lstm part is compared (it contains the whole
pre-registered lru-vs-lstm section).
"""
import os
import re
import subprocess
import sys

import pytest

from _ppo_helpers import PC_FINAL, PG_FINAL, RUNNER

BOOT_FIELDS = re.compile(r"\[[^\]]*\]\s+p_holm=\S+")


def run_script(script, *flags):
    r = subprocess.run([sys.executable, RUNNER, script, *flags], capture_output=True, text=True,
                       encoding="utf-8", timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def committed(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def mask(text):
    return BOOT_FIELDS.sub("<boot>", text)


def drop_uncommitted_arms(fresh, old, arm_rows):
    """Drop table rows of arms that the committed file does not have (results that arrived later)."""
    old_arms = {ln.split()[0] for ln in old.splitlines() if ln.split() and ln.split()[0] in arm_rows}
    keep = []
    for ln in fresh.splitlines():
        tok = ln.split()[0] if ln.split() else ""
        if tok in arm_rows and tok not in old_arms:
            continue
        keep.append(ln)
    return "\n".join(keep) + "\n"


PC_ARMS = {"whitebox", "blackbox", "cmaes", "ref-oracle", "ref-expert", "td3_lru", "td3_lstm", "td3_gru", "td3_rtu"}
PG_ARMS = {"whitebox", "blackbox", "sac", "cmaes-pid", "ref-pid"}


def assert_same(fresh, old):
    if fresh != old:
        import difflib
        diff = "\n".join(difflib.unified_diff(old.splitlines(), fresh.splitlines(), "committed", "fresh", lineterm=""))
        pytest.fail("output differs from the committed text:\n" + diff[:3000])


# ------------------------------------------------------------------ program_control compare.py ----
def test_program_control_compare_matches_committed_fast_layer():
    old = committed(os.path.join(PC_FINAL, "compare.txt"))
    fresh = run_script(os.path.join(PC_FINAL, "compare.py"), "--nboot", "300")
    assert_same(mask(drop_uncommitted_arms(fresh, old, PC_ARMS)), mask(old))


@pytest.mark.slow
def test_program_control_compare_matches_committed_exact():
    old = committed(os.path.join(PC_FINAL, "compare.txt"))
    fresh = run_script(os.path.join(PC_FINAL, "compare.py"))
    assert_same(drop_uncommitted_arms(fresh, old, PC_ARMS), old)


# ------------------------------------------------------------------ pcgym_control compare.py ------
def test_pcgym_compare_matches_committed_fast_layer():
    old = committed(os.path.join(PG_FINAL, "compare.txt"))
    fresh = run_script(os.path.join(PG_FINAL, "compare.py"), "--nboot", "300")
    assert_same(mask(drop_uncommitted_arms(fresh, old, PG_ARMS)), mask(old))


@pytest.mark.slow
def test_pcgym_compare_matches_committed_exact():
    old = committed(os.path.join(PG_FINAL, "compare.txt"))
    fresh = run_script(os.path.join(PG_FINAL, "compare.py"))
    assert_same(drop_uncommitted_arms(fresh, old, PG_ARMS), old)


# ------------------------------------------------------------------ td3_secondary.py --------------
CORES = {"lru", "lstm", "gru", "rtu"}


def restrict_to_committed_cores(fresh, old):
    old_cores = {ln.split()[0] for ln in old.splitlines() if ln.split() and ln.split()[0] in CORES}
    extra = {ln.split()[0] for ln in fresh.splitlines() if ln.split() and ln.split()[0] in CORES} - old_cores
    blocks = fresh.rstrip("\n").split("\n\n")
    out = []
    for b in blocks:
        head = b.lstrip("\n").splitlines()[0] if b.strip() else ""
        m = re.match(r"(\w+) vs (\w+) ", head)
        if m and (m.group(1) in extra or m.group(2) in extra):
            continue                                  # an exploratory pair involving a newer core
        out.append("\n".join(ln for ln in b.splitlines() if not (ln.split() and ln.split()[0] in extra)))
    return "\n\n".join(out) + "\n"


def test_td3_secondary_matches_committed_lru_lstm_section(tmp_path):
    script = os.path.join(PC_FINAL, "td3_secondary.py")
    sink = tmp_path / "td3_secondary.txt"
    committed_path = os.path.join(PC_FINAL, "td3_secondary.txt")
    before = os.path.getmtime(committed_path)
    fresh = run_script(script, "--redirect-write", f"td3_secondary.txt={sink}")
    old = committed(committed_path)
    assert os.path.getmtime(committed_path) == before, "the golden run must not touch the committed file"
    assert sink.read_text(encoding="utf-8") == fresh          # the script's own file output == its stdout
    assert_same(restrict_to_committed_cores(fresh, old), old)

"""Offline tests for results/final/followup/compare_followup.py on synthetic results and logs."""
import importlib.util
import json
import os
import re

import numpy as np
import pytest

PPO_IBM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PPO_IBM, "experiments", "program_control", "results", "final",
                      "followup", "compare_followup.py")
spec = importlib.util.spec_from_file_location("compare_followup_under_test", SCRIPT)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

N_EP = 40          # episodes per synthetic run; the first 10 have init <= 80 cells (not yield-scored)
N_YIELD = 30


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(cf.cmp, "N_BOOT", 200)


def write_result(path, base, crashed=0):
    """A synthetic harness json: harvest = base + small per-episode pattern (same across runs)."""
    eps = []
    for i in range(N_EP):
        eps.append(dict(seed=20_000_000 + i, init_cells=50 if i < 10 else 100 + i, steps=7200,
                        crashed=i < crashed, error=None,
                        harvested_mg=1000 * (base + 0.01 * (i % 5)), time_avg_od=0.3))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"summary": {}, "episodes": eps}, open(path, "w"))


def put(final, rel, base, crashed=0):
    write_result(os.path.join(final, rel), base, crashed)


def expected_median(base):
    return float(np.median([base + 0.01 * (i % 5) for i in range(10, N_EP)]))


def write_log(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def chunk(n=100_000):
    return f"[Chunk] train_diff=D0 | mastery_diff=D0 | init=149 | steps={n:,}"


def adv(a, b):
    return f"  Curriculum ADVANCED: D{a} -> D{b} | chunk_eps=20"


def make_logs(logs, core, seed, d1, d2, abort=False):
    """rl_v3 first 2M (or the lstm60 fresh run), then the resumed rl_4m session to 4M."""
    def span(start, end, events):
        out = []
        for st in range(start + 500_000, end + 1, 500_000):
            out.append(chunk(500_000))
            if st in events:
                out.append(adv(*events[st]))
        return out
    ev = {}
    if d1:
        ev[d1] = (0, 1)
    if d2:
        ev[d2] = (1, 2)
    first = {k: v for k, v in ev.items() if k <= 2_000_000}
    second = {k: v for k, v in ev.items() if k > 2_000_000}
    if abort:
        write_log(os.path.join(logs, "rl_v3", f"rl-v3-{core}-s{seed}", "session1", "train_summary.log"),
                  span(0, 1_000_000, {}) + ["[CAPABILITY ABORT] 12 failed", "--- Training Complete ---"])
        return
    head = ["header"] + span(0, 2_000_000, first) + ["--- Training Complete ---"]
    if core == "lstm60":
        write_log(os.path.join(logs, "rl_4m", "lstm60-s%d" % seed, "session1", "train_summary.log"),
                  head[:-1] + ["[RESUME] step=2,000,000 D0 online_buffer=1 episodes"]
                  + span(2_000_000, 4_000_000, second) + ["--- Training Complete ---"])
        return
    write_log(os.path.join(logs, "rl_v3", f"rl-v3-{core}-s{seed}", "session1", "train_summary.log"), head)
    write_log(os.path.join(logs, "rl_4m", f"{core}-s{seed}", "session1", "train_summary.log"),
              ["[RESUME] step=2,000,000 D2 online_buffer=1 episodes"]
              + span(2_000_000, 4_000_000, second) + ["--- Training Complete ---"])


def build_tree(tmp_path, lstm2_decoy=True, skip=()):
    """A complete synthetic follow-up tree. LRU-600 high, LSTM-600 low, LSTM-60 in between, all
    fully separated seed by seed. `skip` lists result-path prefixes to leave out."""
    final, logs = str(tmp_path / "final"), str(tmp_path / "results")
    vals = {"lru600_2m": 2.0, "lru600_4m": 10.0, "lstm600_2m": 1.0, "lstm600_4m": 1.5,
            "lstm60_2m": 5.0, "lstm60_4m": 6.0}
    for arm, v in vals.items():
        for s in cf.SEEDS:
            rel, brel, carried = cf.result_paths(arm, s)
            if arm == "lstm600_4m" and s == cf.CARRY_SEED:
                continue
            if any(rel.startswith(p) for p in skip):
                continue
            put(final, rel, v + 0.1 * s, crashed=2 if arm.startswith("lstm600") else 0)
            put(final, brel, v + 0.5 + 0.1 * s)
    if lstm2_decoy:  # must be ignored: protocol says the carried-over 2M result counts
        put(final, "followup/td3_lstm_4m__s2.json", 99.0)
    for s in cf.SEEDS:
        make_logs(logs, "lru", s, 1_000_000, 3_500_000 if s == 1 else 1_500_000)
        make_logs(logs, "lstm", s, None, None, abort=(s == 2))
        make_logs(logs, "lstm60", s, 1_000_000, None)
    return final, logs


def report(final, logs):
    lines, complete = cf.build_report(final, logs)
    return "\n".join(lines), complete


# --------------------------------------------------------------------------------------------
def test_complete_tree_runs_family_with_carry_over_and_separation(tmp_path):
    final, logs = build_tree(tmp_path)
    text, complete = report(final, logs)
    assert complete
    assert "INCOMPLETE" not in text.split("Follow-up family")[0]
    assert "Carry-over applied: LSTM-600 seed 2" in text
    # all three comparisons ran and are reported
    for tag in ("F1", "F2", "F3"):
        assert re.search(rf"^  {tag}  .*p_holm=", text, re.M), tag
    # fully separated 5 v 5: smallest possible seed-level exact permutation p
    perms = re.findall(r"seed-level exact permutation test .*: p=([\d.]+)", text)
    assert perms == ["0.0079", "0.0079", "0.0079"]
    # lstm s2 per-run median is its 2M value (1.0 + 0.2), not the 99 decoy
    row = [l for l in text.splitlines() if l.startswith("LSTM-600@4M")][0]
    assert str(round(expected_median(1.0 + 0.2), 2)) in row
    assert "99" not in row
    assert "PARTIAL" not in row


def test_holm_is_over_three(tmp_path, monkeypatch):
    seen = []
    orig = cf.cmp.holm
    monkeypatch.setattr(cf.cmp, "holm", lambda ps: (seen.append(np.array(ps)), orig(ps))[1])
    final, logs = build_tree(tmp_path)
    report(final, logs)
    assert len(seen[-1]) == 3
    # with F3's inputs missing, the family is still size 3 (missing counted as p=1)
    final2, logs2 = build_tree(tmp_path / "b", skip=("followup/td3_lstm60_2m",))
    text, complete = report(final2, logs2)
    assert not complete
    assert len(seen[-1]) == 3 and seen[-1][-1] == 1.0
    assert cf.cmp.holm(np.array([0.01, 0.04, 1.0])) == pytest.approx([0.03, 0.08, 1.0])


def test_missing_inputs_listed_and_incomplete_families_skipped(tmp_path):
    final, logs = build_tree(tmp_path, skip=("followup/td3_lru_4m__s3.json",))
    text, complete = report(final, logs)
    assert not complete
    assert text.splitlines()[0] == "INCOMPLETE - not the pre-registered result"
    assert "  missing: followup/td3_lru_4m__s3.json" in text
    assert re.search(r"F1 .*not run \(arm incomplete\)", text)
    assert re.search(r"F2 .*not run \(arm incomplete\)", text)
    assert re.search(r"^  F3  .*p_holm=", text, re.M)       # F3's arms are complete
    assert "PARTIAL (seeds [1, 2, 4, 5])" in text


def test_empty_tree_does_not_crash(tmp_path):
    final, logs = str(tmp_path / "final"), str(tmp_path / "results")
    os.makedirs(final)
    text, complete = report(final, logs)
    assert not complete
    assert text.startswith("INCOMPLETE")
    assert "missing: followup/td3_lstm60_2m__s5.json" in text


def test_two_to_four_m_changes(tmp_path):
    final, logs = build_tree(tmp_path)
    text, _ = report(final, logs)
    sec = text.split("2M -> 4M change")[1].split("Best-det")[0]
    d = expected_median(10.0 + 0.1 * 3) - expected_median(2.0 + 0.1 * 3)
    assert re.search(rf"LRU-600\s+3\s+{expected_median(2.3):.2f}\s+{expected_median(10.3):.2f}\s+{re.escape(f'{d:+.2f}')}", sec)
    # lstm s2 is carried over: change is exactly zero
    assert re.search(r"LSTM-600\s+2\s+\S+\s+\S+\s+\+0.00\s+\(carried over", sec)
    d60 = expected_median(6.1) - expected_median(5.1)
    assert re.search(rf"LSTM-60\s+1\s+\S+\s+\S+\s+{re.escape(f'{d60:+.2f}')}", sec)


def test_log_replay_across_rl_v3_and_rl_4m_with_resume(tmp_path):
    logs = str(tmp_path)
    # rl_v3 ran to 2.1M and logged D2 there; the rl_4m session resumes at 2,000,000, so that D2
    # event is rewound away and the D2 reached after the resume (3.5M) is the one that counts.
    write_log(os.path.join(logs, "rl_v3", "rl-v3-lru-s1", "session1", "train_summary.log"),
              [chunk(1_000_000), adv(0, 1), chunk(1_000_000), chunk(100_000), adv(1, 2), "--- Training Complete ---"])
    write_log(os.path.join(logs, "rl_4m", "lru-s1", "session1", "train_summary.log"),
              ["[RESUME] step=2,000,000 D1 online_buffer=3 episodes", chunk(1_000_000),
               chunk(500_000), adv(1, 2), chunk(500_000), "--- Training Complete ---"])
    r = cf.parse_followup_run(logs, "lru", 1)
    assert r["d1"] == 1_000_000
    assert r["d2"] == 4_000_000 - 500_000
    assert r["steps"] == 4_000_000 and r["complete"] and r["finished"] and not r["abort"]
    # unfinished: resumed session never completed
    write_log(os.path.join(logs, "rl_4m", "lru-s2", "session1", "train_summary.log"),
              ["[RESUME] step=2,000,000 D1", chunk(1_000_000)])
    write_log(os.path.join(logs, "rl_v3", "rl-v3-lru-s2", "session1", "train_summary.log"),
              [chunk(2_000_000), "--- Training Complete ---"])
    r2 = cf.parse_followup_run(logs, "lru", 2)
    assert r2["steps"] == 3_000_000 and not r2["finished"]
    assert cf.parse_followup_run(logs, "lru", 3) is None


def test_replay_matches_td3_secondary_parse_run_on_single_dir(tmp_path):
    d = tmp_path / "rl-v3-lru-s9"
    write_log(str(d / "session1" / "train_summary.log"),
              [chunk(500_000), adv(0, 1), chunk(500_000), "  Curriculum DEMOTED: D1 -> D0", chunk(1_000_000)])
    write_log(str(d / "session2" / "train_summary.log"),
              ["[RESUME] step=1,000,000 D0", chunk(500_000), adv(0, 1), chunk(500_000), adv(1, 2),
               "[CAPABILITY ABORT] x", "--- Training Complete ---"])
    # session10 must sort after session2 (numeric order)
    write_log(str(d / "session10" / "train_summary.log"), ["[RESUME] step=2,000,000 D2", chunk(100_000)])
    assert cf.replay_logs(cf.session_files(str(d))) == cf.sec.parse_run(str(d))


def test_cli_writes_only_with_flag(tmp_path, capsys):
    final, logs = build_tree(tmp_path)
    out = os.path.join(final, "followup", cf.OUT_NAME)
    assert cf.main(["--final-dir", final, "--logs-dir", logs]) == 0
    assert not os.path.exists(out)
    assert "Pre-registered comparisons" in capsys.readouterr().out
    assert cf.main(["--final-dir", final, "--logs-dir", logs, "--write"]) == 0
    assert "Pre-registered comparisons" in open(out, encoding="utf-8").read()


def test_secondary_tests_use_logs(tmp_path):
    final, logs = build_tree(tmp_path)
    text, _ = report(final, logs)
    assert re.search(r"LRU-600\s+1\s+1,000,000\s+3,500,000\s+D2", text)
    assert re.search(r"LSTM-600\s+2\s+-\s+-\s+D0\s+0\s+yes\s+1,000,000", text)  # aborted, still finished
    assert "F1 pair LRU-600 vs LSTM-600 at 4M (5 v 5 seeds)" in text
    assert "F2 pair LRU-600 vs LSTM-60 at 4M" in text
    assert "seeds reaching D2: LRU-600 5/5, LSTM-600 0/5" in text

"""Offline tests for results/final/ctx/selection.py and compare_ctx.py on synthetic results."""
import importlib.util
import json
import os
import re

import numpy as np
import pytest

PPO_IBM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = os.path.join(PPO_IBM, "experiments", "program_control", "results", "final", "ctx")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CTX, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


selmod = _load("selection_under_test", "selection.py")
cc = _load("compare_ctx_under_test", "compare_ctx.py")

N_EP = 40
SEEDS = [1, 2, 3, 4, 5]


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(cc.cmp, "N_BOOT", 400)


# ---------------------------------------------------------------- selection rule
def test_select_higher_best_wins():
    assert selmod.select(100.0, 101.0) == "best"


def test_select_final_when_higher():
    assert selmod.select(101.0, 100.0) == "final"


def test_select_tie_goes_to_final():
    assert selmod.select(9744.4044192, 9744.4044192) == "final"


def test_select_missing_best_goes_to_final():
    assert selmod.select(5.0, None) == "final"


def write_val(directory, arm, ck, seed, median):
    with open(os.path.join(directory, f"td3_{arm}_{ck}__s{seed}.val.json"), "w") as fh:
        json.dump({"summary": {"median_mg": median}, "episodes": []}, fh)


def test_select_arm_from_files(tmp_path):
    d = str(tmp_path)
    vals = {1: (1.0, 2.0), 2: (3.0, 3.0), 3: (4.0, 1.0), 4: (2.0, None), 5: (1.0, 9.0)}
    for s, (f, b) in vals.items():
        write_val(d, "x", "final", s, f)
        if b is not None:
            write_val(d, "x", "best", s, b)
    got = selmod.select_arm("x", d)
    assert {s: v["selected"] for s, v in got.items()} == \
        {"1": "best", "2": "final", "3": "final", "4": "final", "5": "best"}
    assert got["4"]["best_val_median"] is None


def test_select_arm_missing_final_raises(tmp_path):
    write_val(str(tmp_path), "x", "best", 1, 1.0)
    with pytest.raises(FileNotFoundError):
        selmod.select_arm("x", str(tmp_path), seeds=[1])


def test_recorded_control_selection_matches_val_files():
    sel = json.load(open(os.path.join(CTX, "selection.json")))
    for arm in ("lstmctl", "lructl"):
        assert selmod.select_arm(arm) == sel[arm]


# ---------------------------------------------------------------- compare_ctx
def write_result(path, base, crashed=0):
    eps = [dict(seed=20_000_000 + i, init_cells=50 if i < 10 else 100 + i, steps=7200,
                crashed=i < crashed, error=None,
                harvested_mg=1000 * (base + 0.01 * (i % 5)), time_avg_od=0.3) for i in range(N_EP)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"summary": {}, "episodes": eps}, open(path, "w"))


def build(final, bases, arms=("lstmctl", "lructl", "bi", "w150"), selected="final"):
    """bases[arm] = list of 5 per-seed base harvests (g); every selection is `selected`."""
    sel = {}
    for arm in arms:
        sel[arm] = {}
        for s, b in zip(SEEDS, bases[arm]):
            write_result(os.path.join(final, cc.ckpt_path(arm, "final", s)), b)
            if selected == "best":
                write_result(os.path.join(final, cc.ckpt_path(arm, "best", s)), b + 1)
            sel[arm][str(s)] = dict(final_val_median=1.0, best_val_median=None, selected=selected)
    os.makedirs(os.path.join(final, "ctx"), exist_ok=True)
    json.dump(sel, open(os.path.join(final, "ctx", "selection.json"), "w"))


BASES = {"lstmctl": [5.0, 5.2, 4.8, 5.1, 4.9], "lructl": [8.0, 8.2, 7.8, 8.1, 7.9],
         "bi": [9.0, 9.2, 8.8, 9.1, 8.9],      # C1: large effect
         "w150": [5.05, 5.15, 4.85, 5.0, 4.95]}  # C2: near zero


def holm_p(lines, tag):
    for ln in lines:
        m = re.match(rf"\s+{tag}\s+.*p_holm=<?([\d.e-]+)", ln)
        if m:
            return float(m.group(1))
    raise AssertionError(f"no {tag} line in report")


def test_missing_inputs_refuse(tmp_path, capsys):
    rc = cc.main(["--final-dir", str(tmp_path), "--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "refusing to compute partial results" in out
    assert "missing: ctx/td3_bi_final__s1.final.json" in out
    assert "C1" not in out.split("missing")[-1]  # nothing computed


def test_missing_one_file_refuses_even_when_rest_present(tmp_path):
    build(str(tmp_path), BASES)
    os.remove(tmp_path / "ctx" / "td3_w150_final__s3.final.json")
    lines, complete = cc.build_report(str(tmp_path), str(tmp_path))
    assert not complete and any("td3_w150_final__s3" in ln for ln in lines)
    assert not any("p_holm" in ln for ln in lines)
    assert cc.main(["--final-dir", str(tmp_path), "--logs-dir", str(tmp_path)]) == 1


def test_allow_partial_runs_complete_comparisons(tmp_path):
    build(str(tmp_path), BASES, arms=("lstmctl", "lructl", "bi"))
    lines, complete = cc.build_report(str(tmp_path), str(tmp_path), allow_partial=True)
    text = "\n".join(lines)
    assert not complete and "not the pre-registered result" in text
    assert "C2  W150-LSTM - LSTM-ctl: not run" in text
    assert "C1  BI-LSTM - LSTM-ctl:" in text and "EXPLORATORY SANITY CHECK" in text


def test_full_run_holm_ordering(tmp_path):
    build(str(tmp_path), BASES)
    lines, complete = cc.build_report(str(tmp_path), str(tmp_path))
    assert complete
    p1, p2 = holm_p(lines, "C1"), holm_p(lines, "C2")
    assert p1 <= p2          # the larger effect has the smaller adjusted p
    assert p2 <= 1.0
    text = "\n".join(lines)
    assert "C1  BI-LSTM - LSTM-ctl: +4.0" in text
    assert "seed-level exact permutation test BI-LSTM vs LSTM-ctl (5 v 5 runs" in text


def test_holm_multiplier_is_family_of_two(tmp_path, monkeypatch):
    build(str(tmp_path), BASES, arms=("lstmctl", "lructl", "bi"))  # w150 missing -> counted as p=1
    seen = []
    real = cc.cmp.holm
    monkeypatch.setattr(cc.cmp, "holm", lambda ps: (seen.append(list(ps)), real(ps))[1])
    cc.build_report(str(tmp_path), str(tmp_path), allow_partial=True)
    assert len(seen) == 1 and len(seen[0]) == 2 and seen[0][1] == 1.0


def test_selected_best_checkpoint_is_the_one_read(tmp_path):
    build(str(tmp_path), BASES, selected="best")
    a = cc.load_arm("bi", str(tmp_path), json.load(open(tmp_path / "ctx" / "selection.json")))
    g = np.mean(list(a["sel"][1].values()))
    assert g == pytest.approx(BASES["bi"][0] + 1 + 0.02, abs=0.01)
    assert not a["missing"]


def test_control_paths_map_to_section4_files():
    assert cc.ckpt_path("lstmctl", "final", 3) == "td3_lstm__s3.json"
    assert cc.ckpt_path("lructl", "best", 2) == "secondary/td3_lru_best__s2.json"
    assert cc.ckpt_path("bi", "best", 1) == "ctx/td3_bi_best__s1.final.json"

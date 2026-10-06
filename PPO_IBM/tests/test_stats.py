"""Statistics used by the confirmatory comparisons: Holm, exact permutation, Fisher, the
hierarchical bootstrap, and td3_secondary.parse_run on synthetic training logs."""
import itertools
import os

import numpy as np
import pytest

from _ppo_helpers import PC_FINAL, load_module

cmp_ = load_module("pc_compare_under_test", os.path.join(PC_FINAL, "compare.py"))
sec = load_module("td3_secondary_under_test", os.path.join(PC_FINAL, "td3_secondary.py"))


# ---------------------------------------------------------------- Holm ---------------------------
def test_holm_known_values():
    adj = cmp_.holm(np.array([0.01, 0.04, 0.03, 0.005]))
    # sorted p: .005 .01 .03 .04 -> x4 x3 x2 x1 = .02 .03 .06 .04 -> running max .02 .03 .06 .06
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_holm_single_p_unchanged_and_capped_at_one():
    np.testing.assert_allclose(cmp_.holm(np.array([0.2])), [0.2])
    np.testing.assert_allclose(cmp_.holm(np.array([0.6, 0.7])), [1.0, 1.0])


def test_holm_properties_on_random_input():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.uniform(0, 1, rng.integers(1, 9))
        adj = cmp_.holm(p)
        assert np.all(adj >= p - 1e-12) and np.all(adj <= 1.0)
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-12)                 # monotone in the raw-p order
        assert np.all(adj <= np.minimum(1.0, len(p) * p) + 1e-12)    # never worse than Bonferroni


# ---------------------------------------------------------------- permutation --------------------
@pytest.mark.parametrize("fn", [cmp_.seed_permutation_p, sec.perm_p])
def test_exact_permutation_fully_separated_5v5(fn):
    a, b = np.array([10, 11, 12, 13, 14.0]), np.array([1, 2, 3, 4, 5.0])
    assert fn(a, b) == pytest.approx(2 / 252)       # 0.0079: the two extreme splits of C(10,5) = 252
    assert fn(b, a) == pytest.approx(2 / 252)       # two-sided, so symmetric


@pytest.mark.parametrize("fn", [cmp_.seed_permutation_p, sec.perm_p])
def test_exact_permutation_other_cases(fn):
    arr = np.array
    assert fn(arr([5, 6, 7.0]), arr([1, 2, 3.0])) == pytest.approx(2 / 20)
    assert fn(arr([1, 2, 3, 4.0]), arr([1, 2, 3, 4.0])) == pytest.approx(1.0)      # identical samples
    assert fn(arr([1, 2, 3, 4, 5.0]), arr([1.5, 2.5, 3.5, 4.5, 5.5])) > 0.5      # heavily overlapping


def test_permutation_p_matches_brute_force():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 4), rng.normal(0.5, 1, 5)
    pooled, obs = np.concatenate([a, b]), abs(a.mean() - b.mean())
    hits = 0
    for c in itertools.combinations(range(9), 4):
        rest = [i for i in range(9) if i not in c]
        hits += abs(pooled[list(c)].mean() - pooled[rest].mean()) >= obs - 1e-12
    assert cmp_.seed_permutation_p(a, b) == pytest.approx(hits / 126)


# ---------------------------------------------------------------- Fisher -------------------------
def test_fisher_known_value_from_the_report():
    # 5/5 seeds vs 1/5 seeds reaching D1 (td3_secondary.txt: p=0.0476)
    assert sec.fisher_two_sided(5, 5, 1, 5) == pytest.approx(0.0476, abs=5e-5)
    assert sec.fisher_two_sided(4, 5, 0, 5) == pytest.approx(0.0476, abs=5e-5)
    assert sec.fisher_two_sided(2, 4, 2, 4) == pytest.approx(1.0)


def test_fisher_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    for n1 in range(1, 7):
        for n2 in range(1, 7):
            for k1 in range(n1 + 1):
                for k2 in range(n2 + 1):
                    ref = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
                    assert sec.fisher_two_sided(k1, n1, k2, n2) == pytest.approx(ref, abs=1e-9), (k1, n1, k2, n2)


# ---------------------------------------------------------------- hierarchical bootstrap ---------
def _synthetic(rng, shift, n_runs=5, n_ep=60, run_sd=0.3, ep_sd=1.0):
    """Arms share the per-episode difficulty (paired design); each run adds its own offset."""
    ep_effect = rng.normal(0, ep_sd, n_ep)
    A = shift + ep_effect + rng.normal(0, run_sd, (n_runs, 1)) + rng.normal(0, 0.2, (n_runs, n_ep))
    B = ep_effect + rng.normal(0, run_sd, (n_runs, 1)) + rng.normal(0, 0.2, (n_runs, n_ep))
    return A, B


def test_boot_diff_ci_covers_known_shift(monkeypatch):
    monkeypatch.setattr(cmp_, "N_BOOT", 2000)
    A, B = _synthetic(np.random.default_rng(3), shift=1.0)
    d = cmp_.boot_diff(A, B, np.random.default_rng(0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert len(d) == 2000
    assert lo < 1.0 < hi
    assert lo > 0.0                                          # and clearly excludes "no difference"
    assert abs(d.mean() - (A.mean() - B.mean())) < 0.05      # centred on the point estimate


def test_boot_diff_no_shift_ci_contains_zero(monkeypatch):
    monkeypatch.setattr(cmp_, "N_BOOT", 2000)
    A, B = _synthetic(np.random.default_rng(4), shift=0.0)
    d = cmp_.boot_diff(A, B, np.random.default_rng(0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert lo < 0.0 < hi


def test_boot_diff_coverage_over_replicates(monkeypatch):
    """Over 40 synthetic datasets a 95% CI should cover the true shift most of the time."""
    monkeypatch.setattr(cmp_, "N_BOOT", 300)
    master = np.random.default_rng(5)
    hits = 0
    for _ in range(40):
        A, B = _synthetic(master, shift=0.7)
        d = cmp_.boot_diff(A, B, master)
        lo, hi = np.percentile(d, [2.5, 97.5])
        hits += lo < 0.7 < hi
    # Seeded, so not flaky. Measured coverage is ~84% (150 replicates), not 95%: resampling only 5 runs per
    # arm understates run-to-run spread. The bound guards against it getting worse, not against this.
    assert hits >= 28


# ---------------------------------------------------------------- td3_secondary.parse_run --------
def chunk(steps=100_000, d=0):
    return f"[Chunk] train_diff=D{d} | mastery_diff=D{d} | init=190 | steps={steps:,}"


def advanced(a, b):
    return f"  Curriculum ADVANCED: D{a} -> D{b} | chunk_eps=13 harvest_mg=3054.1 p25=2625.0 crash=0.00%"


def demoted(a, b):
    return f"  Curriculum DEMOTED: D{a} -> D{b} | crash=40.00%"


COMPLETE = "--- Training Complete. Final model saved. ---"


def write(run, n, lines):
    d = run / f"session{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_run_single_complete_session(tmp_path):
    write(tmp_path, 1, ["--- TD3+LRU ---", chunk(), chunk(), advanced(0, 1), chunk(), chunk(), advanced(1, 2),
                        chunk(), COMPLETE])
    r = sec.parse_run(str(tmp_path))
    assert r == dict(d1=200_000, d2=400_000, final_tier=2, demotions=0, abort=False, steps=500_000, complete=True)


def test_parse_run_resume_rewinds_and_drops_later_transitions(tmp_path):
    write(tmp_path, 1, [chunk(), advanced(0, 1),            # D1 at 100k
                        chunk(), chunk(), advanced(1, 2),   # D2 at 300k
                        chunk(), demoted(2, 1)])            # demoted at 400k; session dies (no "Complete")
    # The next Kaggle session restarts from a checkpoint saved at step 200,000: the D2 advance (300k)
    # and the demotion (400k) happened after it and are not part of the surviving run.
    write(tmp_path, 2, ["  [RESUME] step=200,000 D1 online_buffer=188 episodes",
                        chunk(), chunk(), chunk(), advanced(1, 2), chunk(), chunk(), COMPLETE])
    r = sec.parse_run(str(tmp_path))
    assert r["d1"] == 100_000                    # before the rewind point: kept
    assert r["d2"] == 500_000                    # re-earned after the resume, not the discarded 300k
    assert r["demotions"] == 0                   # the demotion was rolled back
    assert r["final_tier"] == 2
    assert r["steps"] == 700_000                 # 200k + 5 chunks
    assert r["complete"] is True and r["abort"] is False


def test_parse_run_resume_keeps_events_at_exactly_the_resume_step(tmp_path):
    write(tmp_path, 1, [chunk(), chunk(), advanced(0, 1), chunk(), advanced(1, 2)])    # D1@200k, D2@300k
    write(tmp_path, 2, ["[RESUME] step=300,000 D2", chunk()])
    r = sec.parse_run(str(tmp_path))
    assert (r["d1"], r["d2"], r["steps"]) == (200_000, 300_000, 400_000)


def test_parse_run_resume_before_everything_clears_history(tmp_path):
    write(tmp_path, 1, [chunk(), advanced(0, 1), chunk()])
    write(tmp_path, 2, ["[RESUME] step=0 D0", chunk()])
    r = sec.parse_run(str(tmp_path))
    assert r["d1"] is None and r["final_tier"] == 0 and r["steps"] == 100_000


def test_parse_run_demotion_and_final_tier(tmp_path):
    write(tmp_path, 1, [chunk(), advanced(0, 1), chunk(), advanced(1, 2), chunk(), demoted(2, 1), chunk(),
                        demoted(1, 0)])
    r = sec.parse_run(str(tmp_path))
    assert (r["d1"], r["d2"], r["demotions"], r["final_tier"]) == (100_000, 200_000, 2, 0)


def test_parse_run_capability_abort(tmp_path):
    write(tmp_path, 1, [chunk(), chunk(),
                        "  [CAPABILITY ABORT] deterministic gate failed 12 consecutive chunks at D0 - stopping run.",
                        COMPLETE])
    r = sec.parse_run(str(tmp_path))
    assert r["abort"] is True and r["complete"] is True and r["steps"] == 200_000


def test_parse_run_complete_flag_is_per_last_session(tmp_path):
    write(tmp_path, 1, [chunk(), COMPLETE])
    write(tmp_path, 2, [chunk()])                          # a later session that never finished
    assert sec.parse_run(str(tmp_path))["complete"] is False


def test_parse_run_sessions_sorted_numerically_not_lexically(tmp_path):
    write(tmp_path, 1, [chunk(), chunk()])                                        # -> 200k
    write(tmp_path, 2, ["[RESUME] step=100,000 D0", chunk()])                     # -> 200k
    write(tmp_path, 10, ["[RESUME] step=200,000 D0", chunk(), COMPLETE])          # -> 300k, complete
    r = sec.parse_run(str(tmp_path))
    # Lexical order (1, 10, 2) would end on session 2 and report 200k, not complete.
    assert r["steps"] == 300_000 and r["complete"] is True


def test_parse_run_empty_dir(tmp_path):
    r = sec.parse_run(str(tmp_path))
    assert r == dict(d1=None, d2=None, final_tier=0, demotions=0, abort=False, steps=0, complete=False)


def test_parse_run_ignores_non_utf8_bytes(tmp_path):
    d = tmp_path / "session1"
    d.mkdir()
    (d / "train_summary.log").write_bytes(b"\xff\xfe junk\n" + chunk().encode() + b"\n")
    assert sec.parse_run(str(tmp_path))["steps"] == 100_000


def test_censoring_helpers():
    assert sec.fmt_med([1_000_000, 1_500_000, 2_000_000]) == "1,500,000"
    assert sec.fmt_med([2_000_000, 2_000_000, 1_000_000]) == ">=2,000,000"
    assert sec.fmt_step(None) == "-" and sec.fmt_step(1_100_000) == "1,100,000"
    assert sec.num("1,234,567") == 1234567

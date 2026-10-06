"""Split integrity: seed ranges of search / pilot / test / final / training are disjoint and the
confirmatory ('final') split has the documented size and composition, for both testbeds. Also checks
that every committed final-split result file really ran on that split."""
import ast
import glob
import json
import os

import pytest

from _ppo_helpers import PC, PC_FINAL, PG, PG_FINAL, load_module

pc_h = load_module("pc_harness_under_test", os.path.join(PC, "harness.py"))
pg_h = load_module("pg_harness_under_test", os.path.join(PG, "harness.py"))


def const(path, name):
    """Read a module-level integer constant without importing the module (some import SB3/pcgym)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# ------------------------------------------------------------------ photobioreactor ---------------
def seeds(split):
    return [s for _, s in pc_h.split_jobs(split, 2)]


@pytest.mark.parametrize("split,n_main,n_high,base", [("final", 160, 40, 20_000_000), ("search", 9, 3, 3_000_000),
                                                      ("test", 40, 12, 1000)])
def test_pb_split_sizes_and_bases(split, n_main, n_high, base):
    jobs = pc_h.split_jobs(split, 2)
    assert len(jobs) == n_main + n_high
    main, high = jobs[:n_main], jobs[n_main:]
    assert [s for _, s in main] == list(range(base, base + n_main))
    assert [s for _, s in high] == list(range(base + 500_000, base + 500_000 + n_high))
    assert all(ic >= 600 for ic, _ in high)             # the high-inoculum block
    assert all(ic < 5001 for ic, _ in high)


def test_pb_final_has_175_yield_scored_episodes():
    """compare.py prints '175 yield-scored episodes per run': init > 80 cells among the 200."""
    jobs = pc_h.split_jobs("final", 2)
    assert sum(ic > pc_h.ADVERSARIAL_MAX for ic, _ in jobs) == 175
    assert pc_h.ADVERSARIAL_MAX == 80


def test_pb_splits_are_deterministic():
    assert pc_h.split_jobs("final", 2) == pc_h.split_jobs("final", 2)
    assert pc_h.split_jobs("final", 0) == pc_h.split_jobs("final", 2)       # difficulty does not change the draw


def test_pb_splits_pairwise_disjoint():
    pilot_base = const(os.path.join(PC, "blackbox", "plant_trial.py"), "SEED_BASE")
    pilot_max = pilot_base + 100_000 * 97 + 100_000      # worst-case block offset + a generous batch size
    groups = {name: set(seeds(name)) for name in ("search", "test", "final")}
    groups["det"] = {s for _, s in pc_h.split_jobs("det", 2)}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not groups[a] & groups[b], f"{a} and {b} share seeds"
    assert pilot_base == 5_000_000
    for name, g in groups.items():
        assert not any(pilot_base <= s <= pilot_max for s in g), f"{name} overlaps the pilot seed range"
    assert min(groups["final"]) == 20_000_000 and pilot_max < 20_000_000      # final lies above everything else
    assert max(groups["search"]) < pilot_base and max(groups["test"]) < pilot_base


def test_pb_init_cells_distribution_documented_in_harness():
    jobs = pc_h.split_jobs("final", 2)[:160]
    ics = [ic for ic, _ in jobs]
    assert all(30 <= ic < 80 or 100 <= ic <= 400 for ic in ics)       # 10% adversarial, rest 100-400
    assert 5 <= sum(ic <= 80 for ic in ics) <= 35                      # ~10% of 160


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(PC_FINAL, "*__*.json"))),
                         ids=lambda p: os.path.basename(p))
def test_pb_committed_final_results_ran_on_the_final_split(path):
    eps = json.load(open(path))["episodes"]
    assert [(e["init_cells"], e["seed"]) for e in eps] == pc_h.split_jobs("final", 2)


# ------------------------------------------------------------------ PC-Gym ------------------------
def test_pcgym_split_definitions():
    assert pg_h.SPLITS == {"search": (3_000_000, 12), "final": (20_000_000, 200)}


def test_pcgym_seed_ranges_disjoint():
    s_base, s_n = pg_h.SPLITS["search"]
    f_base, f_n = pg_h.SPLITS["final"]
    pilot_base = const(os.path.join(PG, "plant_trial.py"), "SEED_BASE")
    train_base = const(os.path.join(PG, "rl_sac.py"), "TRAIN_SEED_BASE")
    assert (pilot_base, train_base) == (5_000_000, 7_000_000)
    search = set(range(s_base, s_base + s_n))
    final = set(range(f_base, f_base + f_n))
    assert not search & final
    # SAC draws training seeds as TRAIN_SEED_BASE + randint(10_000_000); pilot blocks start at
    # 5M + 100k * (1 + h % 97). Both stay below the final split, and the search split sits below both.
    train_max = train_base + 10_000_000 - 1
    pilot_max = pilot_base + 100_000 * 97 + 100_000
    assert max(search) < pilot_base < train_base
    assert train_max < min(final) and pilot_max < min(final)


def test_pcgym_committed_final_results_ran_on_the_final_split():
    paths = sorted(glob.glob(os.path.join(PG_FINAL, "*__*.json")))
    assert paths
    for p in paths:
        eps = json.load(open(p))["episodes"]
        assert [e["seed"] for e in eps] == list(range(20_000_000, 20_000_200)), os.path.basename(p)

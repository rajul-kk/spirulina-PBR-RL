"""Section 9 checkpoint selection (docs/reports/comparison_protocol.md).

Candidates per run: the final 2M checkpoint and the best-det checkpoint (if one exists). Each is
scored once on the `val` split. Select the higher median harvest; a tie or a missing best-det
checkpoint selects the final checkpoint. Made on `val` only.

Inputs : td3_<arm>_<final|best>__s<seed>.val.json  (summary.median_mg), in this directory.
Output : selection.json  {arm: {seed: {final_val_median, best_val_median, selected}}}

  python selection.py lstmctl lructl      # compute and (re)write the entries for these arms
  python selection.py bi w150
Entries for arms not named are kept. A missing final val score is an error (exit 1).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3, 4, 5]
OUT = os.path.join(HERE, "selection.json")


def select(final_median, best_median):
    """'best' only if a best score exists and is strictly higher; otherwise 'final'."""
    if best_median is not None and best_median > final_median:
        return "best"
    return "final"


def _median(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))["summary"]["median_mg"]


def select_arm(arm, directory=HERE, seeds=SEEDS):
    out = {}
    for s in seeds:
        f = _median(os.path.join(directory, f"td3_{arm}_final__s{s}.val.json"))
        b = _median(os.path.join(directory, f"td3_{arm}_best__s{s}.val.json"))
        if f is None:
            raise FileNotFoundError(f"missing final val score for {arm} seed {s}")
        out[str(s)] = dict(final_val_median=f, best_val_median=b, selected=select(f, b))
    return out


def main(argv=None):
    arms = (sys.argv[1:] if argv is None else argv)
    if not arms:
        print(__doc__)
        return 2
    sel = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm in arms:
        try:
            sel[arm] = select_arm(arm)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1
    json.dump(sel, open(OUT, "w"), indent=1)
    for arm in arms:
        print(arm, {s: v["selected"] for s, v in sel[arm].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

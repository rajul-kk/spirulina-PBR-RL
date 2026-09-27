"""Run TD3 actor checkpoints through the program harness on the test split (D0 and D2)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "controllers", "td3_actor.py")

if __name__ == "__main__":
    # usage: run_rl_test.py label=actor_path [label=actor_path ...]
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        for d in (0, 2):
            s, res = evaluate(ADAPTER, "test", d, {"actor_path": path, "reset": 600}, workers=2)
            print(f"{label} D{d}: median {s['median_mg']:.0f} mg, p25 {s['p25_mg']:.0f} mg, crash "
                  f"{100*s['crash_rate']:.0f}%, time-avg od {s['median_time_avg_od']:.2f}", flush=True)
            with open(os.path.join(HERE, "results", "test", f"{label}_D{d}.json"), "w") as f:
                json.dump({"summary": s, "episodes": [{k: v for k, v in r.items() if k != "trace"} for r in res]}, f, indent=1)

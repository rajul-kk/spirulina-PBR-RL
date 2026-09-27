"""LLM program evolution for the photobioreactor controller: one loop, several writers.

Every writer gets the same prompt packet (task spec, the best programs so far with their
scores, episode feedback and traces), starts from the same seed program, has the same candidate
budget and is scored on the same search split. Only the writer differs:

  auto    a local model through Ollama (e.g. qwen3:8b, llama3.2:3b); fully automatic
  manual  a writer outside this script (e.g. Claude in an agent session): `prompt` writes the
          packet to <run>/prompt.md, the writer saves <run>/cand_NNN.py, `score` evaluates it

  python evolve.py auto   --run qwen3 --model qwen3:8b --budget 16
  python evolve.py prompt --run claude
  python evolve.py score  --run claude <run>/cand_001.py
  python evolve.py table

Generated code runs locally, so it is statically checked first (whitelisted imports, no
file/OS/exec access) and each episode has a wall-clock cap in the harness.
"""
import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import SENSOR_DOC, evaluate  # noqa: E402

RUNS = os.environ.get("EVOLVE_RUNS_DIR", os.path.join(HERE, "results", "evolve"))
WORKERS = int(os.environ.get("EVOLVE_WORKERS", "2"))   # parallel scoring episodes
SEED_PROGRAM = os.path.join(HERE, "controllers", "sensor_expert.py")
ALLOWED_IMPORTS = {"numpy", "math", "collections", "process_state"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr",
                "setattr", "delattr", "input", "breakpoint", "vars"}

SPEC = f"""\
# Task: write a photobioreactor controller program

You are designing a controller for a simulated Spirulina photobioreactor (150 L-scale, 144 h
batch, 7200 control steps of 72 s). The goal is to HARVEST AS MUCH BIOMASS AS POSSIBLE over the
batch WITHOUT the culture dying out (extinction = crash, heavily penalised). Starting
populations vary widely (30 to 5000 cells-equivalent), strains vary per episode (max growth
rate, temperature optimum), and at the evaluated difficulty sensors are noisy, drift and lag.

## Interface (the whole file is your answer)

```python
class Controller:
    def __init__(self, params=None): ...
    def act(self, obs: dict) -> tuple:   # (stir_rpm, light_umol, harvest_frac)
```
- stir_rpm 50-200 (mixing; more stir = better gas exchange but more shear/heat)
- light_umol 0-2000 (LED intensity; drives growth but saturates, photo-inhibits and heats the
  tank; the thermostat can only remove limited heat)
- harvest_frac 0-0.5

{SENSOR_DOC}

Allowed imports: numpy, math, collections, process_state. `process_state.ProcessState` is an
optional helper: `card = ProcessState().update(obs)` returns od_est (turbidity/250),
od_est_slow, growth_per_h, temp_c (smoothed), hours, hours_to_harvest, harvests_done, pump_L,
ph, light_obs_umol. You may keep any state you like on self. No file, network or OS access.
act() is called 7200 times per episode, so keep it cheap (well under 1 ms).

## Scoring
12 episodes per candidate. fitness = 0.5*median_harvest_mg + 0.5*p25_harvest_mg
- 10000*crash_rate (median/p25 over episodes starting above 80 cells; tiny starts are scored
on survival). A program that raises an exception loses the rest of that episode.

Reply with ONE complete Python file in a single ```python block, then at most 3 lines on what
you changed and why.
"""


# ─── static safety check ──────────────────────────────────────────────────────

def check_program(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of {a.name} not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from {node.module} not allowed"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of {node.id} not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            return f"dunder attribute {node.attr} not allowed"
    if not any(isinstance(n, ast.ClassDef) and n.name == "Controller" for n in tree.body):
        return "no top-level class Controller"
    return None


# ─── archive ──────────────────────────────────────────────────────────────────

def run_dir(run):
    d = os.path.join(RUNS, run)
    os.makedirs(d, exist_ok=True)
    return d


def load_archive(run):
    path = os.path.join(run_dir(run), "archive.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record(run, entry):
    with open(os.path.join(run_dir(run), "archive.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def score_file(run, path, note="", writer_seconds=None):
    src = open(path, encoding="utf-8").read()
    problem = check_program(src)
    entry = {"file": os.path.relpath(path, run_dir(run)), "time": time.strftime("%Y-%m-%d %H:%M"),
             "note": note, "writer_seconds": writer_seconds}
    if problem:
        entry.update(fitness=-1e9, rejected=problem)
        record(run, entry)
        print(f"  {entry['file']}: REJECTED ({problem})")
        return entry
    t0 = time.time()
    s, results = evaluate(path, "search", 2, workers=WORKERS, trace_every=600)
    entry.update(fitness=s["fitness"], summary=s, eval_seconds=round(time.time() - t0),
                 episodes=[{k: v for k, v in r.items() if k != "trace"} for r in results])
    with open(path + ".traces.json", "w") as f:
        json.dump([{"init_cells": r["init_cells"], "seed": r["seed"], "trace": r["trace"]} for r in results], f)
    record(run, entry)
    print(f"  {entry['file']}: fitness {s['fitness']:.0f} | median {s['median_mg']:.0f} mg, "
          f"p25 {s['p25_mg']:.0f} mg, crash {s['crash_rate']:.0%}, errors {s['errors']} "
          f"[{entry['eval_seconds']}s]", flush=True)
    return entry


def ensure_seed(run):
    arch = load_archive(run)
    if arch:
        return arch
    dst = os.path.join(run_dir(run), "cand_000.py")
    shutil.copy(SEED_PROGRAM, dst)
    score_file(run, dst, note="seed: hand-tuned sensor expert")
    return load_archive(run)


# ─── prompt packet ────────────────────────────────────────────────────────────

def _feedback(run, entry, max_trace_eps=2):
    lines = []
    for e in entry.get("episodes", []):
        flag = "CRASH" if e["crashed"] else ""
        err = f" ERROR {e['error']}" if e.get("error") else ""
        lines.append(f"  init={e['init_cells']:5d} harvested={e['harvested_mg']:8.0f} mg "
                     f"steps={e['steps']:5d} time_avg_od={e['time_avg_od']:.2f} {flag}{err}")
    tpath = os.path.join(run_dir(run), entry["file"] + ".traces.json")
    if os.path.exists(tpath):
        traces = json.load(open(tpath))
        eps = {(e["init_cells"], e["seed"]): e for e in entry["episodes"]}
        # Worst yielding episode and the largest start: where programs usually fail.
        yielding = [t for t in traces if t["init_cells"] > 80]
        pick = []
        if yielding:
            pick.append(min(yielding, key=lambda t: eps[(t["init_cells"], t["seed"])]["harvested_mg"]))
            big = max(yielding, key=lambda t: t["init_cells"])
            if big is not pick[0]:
                pick.append(big)
        for t in pick[:max_trace_eps]:
            lines.append(f"  trace init={t['init_cells']} (every 12 h; true_* fields are simulator "
                         f"truth, not visible to the controller):")
            for row in t["trace"]:
                lines.append("    " + json.dumps(row))
    return "\n".join(lines)


def build_prompt(run, top_k=2):
    arch = [a for a in load_archive(run) if "summary" in a]
    arch.sort(key=lambda a: -a["fitness"])
    parts = [SPEC, "\n## Results so far (search split, best first)\n"]
    for a in arch[:10]:
        s = a["summary"]
        parts.append(f"- {a['file']}: fitness {a['fitness']:.0f}, median {s['median_mg']:.0f} mg, "
                     f"p25 {s['p25_mg']:.0f} mg, crash {s['crash_rate']:.0%} {('- ' + a['note']) if a['note'] else ''}")
    rejected = [a for a in load_archive(run) if a.get("rejected")]
    if rejected:
        parts.append(f"\n({len(rejected)} earlier candidates were rejected: "
                     + "; ".join(sorted({a['rejected'] for a in rejected})) + ")")
    for a in arch[:top_k]:
        src = open(os.path.join(run_dir(run), a["file"]), encoding="utf-8").read()
        parts.append(f"\n## Program {a['file']} (fitness {a['fitness']:.0f})\n```python\n{src}\n```\n"
                     f"Per-episode results:\n{_feedback(run, a)}")
    parts.append("\nWrite an improved controller. Improve on the best program; do not just resubmit it.")
    return "\n".join(parts)


# ─── writers ──────────────────────────────────────────────────────────────────

def ollama_generate(model, prompt, timeout=3600):
    body = {"model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.7, "num_ctx": 16384}}
    if model.startswith("qwen3"):
        body["think"] = False     # qwen3 thinks by default; other models may reject the flag
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["message"]["content"], out.get("eval_count"), out.get("prompt_eval_count")


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    blocks = [b for b in blocks if "class Controller" in b]
    return max(blocks, key=len) if blocks else None


def next_cand_path(run):
    n = len([f for f in os.listdir(run_dir(run)) if re.fullmatch(r"cand_\d{3}\.py", f)])
    return os.path.join(run_dir(run), f"cand_{n:03d}.py")


def cmd_auto(args):
    ensure_seed(args.run)
    while True:
        n_done = len(load_archive(args.run)) - 1        # the seed doesn't count toward the budget
        if n_done >= args.budget:
            break
        prompt = build_prompt(args.run)
        t0 = time.time()
        try:
            text, out_tok, in_tok = ollama_generate(args.model, prompt)
        except Exception as e:
            print(f"  generation failed: {e}", flush=True)
            time.sleep(30)
            continue
        dt = round(time.time() - t0)
        path = next_cand_path(args.run)
        with open(path + ".reply.md", "w", encoding="utf-8") as f:
            f.write(text)
        code = extract_code(text)
        if code is None:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# no python block in reply\n")
            record(args.run, {"file": os.path.basename(path), "fitness": -1e9, "rejected": "no code block",
                              "writer_seconds": dt, "time": time.strftime("%Y-%m-%d %H:%M"), "note": ""})
            print(f"  {os.path.basename(path)}: no code block [{dt}s]", flush=True)
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        tail = text.split("```")[-1].strip().splitlines()
        score_file(args.run, path, note=" ".join(tail[:3])[:200],
                   writer_seconds=dt)
        print(f"    (writer {args.model}: {dt}s, {out_tok} out / {in_tok} in tokens)", flush=True)


def cmd_prompt(args):
    ensure_seed(args.run)
    p = os.path.join(run_dir(args.run), "prompt.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(build_prompt(args.run))
    print(p, "| next candidate:", next_cand_path(args.run))


def cmd_score(args):
    score_file(args.run, os.path.abspath(args.file), note=args.note or "")


def cmd_table(args):
    for run in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        arch = load_archive(run)
        ok = [a for a in arch if "summary" in a]
        if not ok:
            continue
        best = max(ok, key=lambda a: a["fitness"])
        rej = sum(1 for a in arch if a.get("rejected"))
        errs = sum(1 for a in ok if a["summary"]["errors"])
        print(f"{run:>12}: {len(arch)-1:2d} candidates ({rej} rejected, {errs} with runtime errors) | "
              f"best {best['file']} fitness {best['fitness']:.0f}, median {best['summary']['median_mg']:.0f}, "
              f"p25 {best['summary']['p25_mg']:.0f}, crash {best['summary']['crash_rate']:.0%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("auto"); a.add_argument("--run", required=True); a.add_argument("--model", required=True)
    a.add_argument("--budget", type=int, default=16)
    p = sub.add_parser("prompt"); p.add_argument("--run", required=True)
    s = sub.add_parser("score"); s.add_argument("--run", required=True); s.add_argument("file")
    s.add_argument("--note", default="")
    sub.add_parser("table")
    args = ap.parse_args()
    {"auto": cmd_auto, "prompt": cmd_prompt, "score": cmd_score, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()

"""Run one of the committed analysis scripts as __main__ in a child process, without letting it
write into results/.

  python _runner.py <script.py> [--nboot N] [--redirect-write SUFFIX=OUTFILE]

--nboot N      sets compare.N_BOOT (program_control/results/final/compare.py) before the script
               runs; the pcgym comparison reuses that module, so this speeds up both.
--redirect-write td3_secondary.txt=/tmp/x.txt   any open(..., 'w') whose path ends with the
               suffix is sent to OUTFILE instead (td3_secondary.py writes its txt unconditionally).
The script's stdout is passed through unchanged.
"""
import builtins
import os
import runpy
import sys

args = sys.argv[1:]
script = args.pop(0)
nboot, redirects = None, {}
while args:
    flag = args.pop(0)
    if flag == "--nboot":
        nboot = int(args.pop(0))
    elif flag == "--redirect-write":
        suffix, out = args.pop(0).split("=", 1)
        redirects[suffix] = out
    else:
        raise SystemExit(f"unknown flag {flag}")

if nboot is not None:
    pc_final = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments",
                            "program_control", "results", "final")
    sys.path.insert(0, pc_final)
    import compare
    compare.N_BOOT = nboot

if redirects:
    _real_open = builtins.open

    def _open(path, mode="r", *a, **k):
        if "w" in mode or "a" in mode:
            for suffix, out in redirects.items():
                if str(path).endswith(suffix):
                    return _real_open(out, mode, *a, **k)
        return _real_open(path, mode, *a, **k)

    builtins.open = _open

sys.argv = [script]
src = open(script, encoding="utf-8").read()
if nboot is not None and "N_BOOT = 20000" in src:
    # program_control's compare.py defines N_BOOT itself, so patch it textually (the file on disk
    # is untouched); the pcgym script reads compare.N_BOOT from the module patched above.
    exec(compile(src.replace("N_BOOT = 20000", f"N_BOOT = {nboot}"), script, "exec"),
         {"__name__": "__main__", "__file__": script})
else:
    runpy.run_path(script, run_name="__main__")

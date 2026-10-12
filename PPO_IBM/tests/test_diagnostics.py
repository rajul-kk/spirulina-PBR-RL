"""Smoke test: summarize_sweep parses a tiny fake sweep JSON."""
import importlib.util
import json
import os

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "experiments", "program_control", "diagnostics", "summarize_sweep.py")


def test_summarize_sweep_parses_fake_json(tmp_path):
    spec = importlib.util.spec_from_file_location("summarize_sweep", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for seed, mg in ((1, 4000.0), (2, 6000.0)):
        (tmp_path / f"lru600_s{seed}_r600.json").write_text(
            json.dumps({"summary": {"median_mg": mg, "crash_rate": 0.1}, "episodes": []}))
    (tmp_path / "ignored.json").write_text("{}")
    rows = mod.load(str(tmp_path))
    assert rows[("lru600", 1, "600")] == (4.0, 0.1)
    text = mod.render(rows)
    assert "lru600" in text and "5.00 / 0.100 [2]" in text

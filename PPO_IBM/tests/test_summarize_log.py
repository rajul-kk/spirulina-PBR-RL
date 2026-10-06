"""experiments/program_control/kaggle/summarize_log.py: train.log -> train_summary.log."""
import glob
import os
import subprocess
import sys

import pytest

from _ppo_helpers import PC, load_module

SCRIPT = os.path.join(PC, "kaggle", "summarize_log.py")
sl = load_module("summarize_log_under_test", SCRIPT)

RAW = (
    "--- TD3+LRU | core=DiagonalLRU | threads=4 ---\n"
    "[Chunk] train_diff=D0 | mastery_diff=D0 | init=190 | steps=100,000\n"
    "D0:   0%|          | 0/100000 [00:00<?, ?it/s]\rD0:   2%|2         | 2000/100000 [00:10<08:00, 200.0it/s]"
    "\rD0: 100%|##########| 100000/100000 [08:20<00:00, 199.9it/s]\n"
    "  Curriculum ADVANCED: D0 -> D1 | chunk_eps=13 harvest_mg=3054.1\n"
    "  [RESUME] step=1,300,000 D1 online_buffer=188 episodes\n"
    "--- Training Complete. Final model saved. ---\n"
)


def test_carriage_returns_become_newlines_and_it_per_s_lines_are_dropped():
    out = sl.summarize_text(RAW)
    assert "it/s" not in out
    assert "\r" not in out
    assert out.splitlines() == [
        "--- TD3+LRU | core=DiagonalLRU | threads=4 ---",
        "[Chunk] train_diff=D0 | mastery_diff=D0 | init=190 | steps=100,000",
        "  Curriculum ADVANCED: D0 -> D1 | chunk_eps=13 harvest_mg=3054.1",
        "  [RESUME] step=1,300,000 D1 online_buffer=188 episodes",
        "--- Training Complete. Final model saved. ---",
    ]
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_blank_lines_and_final_unterminated_line_are_kept():
    assert sl.summarize_text("a\n\nb") == "a\n\nb\n"
    assert sl.summarize_text("") == ""
    assert sl.summarize_text("x it/s\n") == ""


def test_matches_the_shell_pipeline_semantics_on_a_progress_frame_at_end_of_file():
    assert sl.summarize_text("keep\nbar 1it/s\rbar 2it/s") == "keep\n"


def test_summarize_file_default_destination_and_encoding(tmp_path):
    src = tmp_path / "train.log"
    src.write_bytes(RAW.encode("utf-8") + b"bad byte \xff here\n")
    dst = sl.summarize_file(str(src))
    assert dst == str(tmp_path / "train_summary.log")
    data = open(dst, "rb").read()
    assert b"\r" not in data
    assert "�".encode() in data          # undecodable bytes are replaced, not fatal
    assert data.startswith(b"--- TD3+LRU")


def test_summarize_file_explicit_destination(tmp_path):
    src, out = tmp_path / "x.log", tmp_path / "y.txt"
    src.write_bytes(b"hello\n")
    assert sl.summarize_file(str(src), str(out)) == str(out)
    assert out.read_text() == "hello\n"


def test_cli_on_session_directories(tmp_path):
    for n in (1, 2):
        d = tmp_path / f"session{n}"
        d.mkdir()
        (d / "train.log").write_bytes(RAW.encode("utf-8"))
    r = subprocess.run([sys.executable, SCRIPT, str(tmp_path / "session1"), str(tmp_path / "session2")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for n in (1, 2):
        assert (tmp_path / f"session{n}" / "train_summary.log").read_text(encoding="utf-8") == sl.summarize_text(RAW)


def test_cli_reports_missing_log(tmp_path):
    r = subprocess.run([sys.executable, SCRIPT, str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 1 and "no such log" in r.stderr


RAW2 = ("[Chunk] train_diff=D0 | mastery_diff=D0 | init=190 | steps=100,000\n"
        "D0:   2%|2 | 2000/100000 [00:10<08:00, 200.0it/s]\rD0: 100%|##| 100000/100000 [08:20<00:00, 199.9it/s]\n"
        "  Curriculum ADVANCED: D0 -> D1 | chunk_eps=13\n"
        "[Chunk] train_diff=D1 | mastery_diff=D1 | init=300 | steps=100,000\n"
        "D1: 100%|##| 100000/100000 [08:20<00:00, 199.9it/s]\n"
        "--- Training Complete. Final model saved. ---\n")


def test_output_parses_with_td3_secondary(tmp_path):
    from _ppo_helpers import PC_FINAL
    sec = load_module("td3_secondary_for_summarize_test", os.path.join(PC_FINAL, "td3_secondary.py"))
    sess = tmp_path / "session1"
    sess.mkdir()
    (sess / "train.log").write_bytes(RAW2.encode("utf-8"))
    sl.summarize_file(str(sess / "train.log"))
    r = sec.parse_run(str(tmp_path))
    assert (r["d1"], r["steps"], r["complete"], r["final_tier"]) == (100_000, 200_000, True, 1)


COMMITTED = sorted(glob.glob(os.path.join(PC, "results", "rl_v3", "rl-v3-lru-s*", "session*", "train.log")))


@pytest.mark.skipif(not COMMITTED, reason="committed Kaggle logs not present")
def test_reproduces_committed_summaries_up_to_the_trailing_blank_line():
    checked = 0
    for log in COMMITTED:
        want = os.path.join(os.path.dirname(log), "train_summary.log")
        if not os.path.exists(want):
            continue
        got = sl.summarize_text(open(log, "rb").read().decode("utf-8", errors="replace"))
        assert got.rstrip("\n") == open(want, "rb").read().decode("utf-8", errors="replace").rstrip("\n"), log
        checked += 1
    assert checked >= 1

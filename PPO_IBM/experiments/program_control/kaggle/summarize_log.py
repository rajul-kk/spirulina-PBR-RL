"""Turn a Kaggle session's train.log into train_summary.log, the file td3_secondary.py and
compare_followup.py read.

tqdm redraws its progress bar with carriage returns, so train.log is dominated by "it/s" frames. The
summary splits those frames onto their own lines ('\r' -> '\n') and drops every line containing
'it/s'; everything else (chunk headers, curriculum decisions, [RESUME], [CAPABILITY ABORT],
"--- Training Complete") is kept verbatim. Equivalent to:

    tr '\r' '\n' < train.log | grep -v 'it/s' > train_summary.log

  python summarize_log.py <train.log> [<train_summary.log>]    # default output: next to the input
  python summarize_log.py <session_dir> [<session_dir> ...]     # train.log -> train_summary.log in each
"""
import os
import sys


def summarize_text(text):
    """The summary of a raw log: progress-bar frames split out and dropped, other lines untouched."""
    lines = text.replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":          # the newline that ended the last line is not a line
        lines.pop()
    return "".join(line + "\n" for line in lines if "it/s" not in line)


def summarize_file(src, dst=None):
    """Write the summary of `src` to `dst` (default: train_summary.log beside it). Returns dst."""
    dst = dst or os.path.join(os.path.dirname(os.path.abspath(src)), "train_summary.log")
    with open(src, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(summarize_text(text))
    return dst


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    if len(argv) == 2 and not os.path.isdir(argv[0]):
        print(summarize_file(argv[0], argv[1]))
        return 0
    for arg in argv:
        src = os.path.join(arg, "train.log") if os.path.isdir(arg) else arg
        if not os.path.isfile(src):
            print(f"no such log: {src}", file=sys.stderr)
            return 1
        print(summarize_file(src))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

import re

_WORD_RE = re.compile(r'\w+')

_MARKERS = {
    "toc":      "## Table of Contents",
    "body":     "## 1. Environment Overview",
    "refs":     "## References",
    "appendix": "## Appendix A",
}

_FILES = {
    "Genetic": "genetic_env.md",
    "Light":   "light_env.md",
}


def _is_heading(line):
    return line.strip().startswith('#')


def get_word_count(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    indices = {key: None for key in _MARKERS}
    for i, line in enumerate(lines):
        for key, marker in _MARKERS.items():
            if indices[key] is None and marker in line:
                indices[key] = i
        if all(v is not None for v in indices.values()):
            break

    def count_in_range(start, end, include_headings=False):
        chunk = "\n".join(
            line for line in lines[start:end]
            if include_headings or not _is_heading(line)
        )
        return sum(1 for _ in _WORD_RE.finditer(chunk))

    toc_end   = indices["toc"]  or 0
    body_start = indices["body"] or 0
    limit = min(
        idx for idx in [indices["refs"], indices["appendix"], len(lines)]
        if idx is not None
    )

    # count1: pre-TOC content (headings excluded — only author/date lines)
    # count2: body sections 1-onwards (headings included)
    count1 = count_in_range(0, toc_end, include_headings=False)
    count2 = count_in_range(body_start, limit, include_headings=True)
    return count1 + count2


for label, path in _FILES.items():
    print(f"{label} Word Count: {get_word_count(path)}")

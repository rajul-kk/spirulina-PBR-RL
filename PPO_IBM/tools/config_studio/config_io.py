"""config_io.py — read/write schema.py's target files via targeted regex substitution
(not a full AST rewrite, to avoid mangling hand-formatted comments). Each write is
followed by a git commit scoped to that one file."""

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # PPO_IBM/
REPO_ROOT = os.path.dirname(ROOT)  # e:/SEGP


def _path(rel_file):
    return os.path.join(ROOT, rel_file)


def _read(rel_file):
    with open(_path(rel_file), encoding="utf-8") as fh:
        return fh.read()


def _write(rel_file, text):
    with open(_path(rel_file), "w", encoding="utf-8") as fh:
        fh.write(text)


def _tier_block_span(text, tier):
    m = re.search(rf"^\s*{tier}: \{{.*?\}},", text, re.M | re.S)
    if not m:
        raise ValueError(f"could not find ADVANCE_TARGETS tier {tier} block")
    return m


def _cast(value, kind):
    return int(value) if kind == "int" else float(value)


def read_field(field):
    text = _read(field["file"])
    kind = field["kind"]

    if kind == "simple":
        m = re.search(rf'^{field["name"]}\s*=\s*([0-9.eE+-]+)', text, re.M)
        if not m:
            raise ValueError(f"could not find {field['name']} in {field['file']}")
        return _cast(m.group(1), field["cast"])

    if kind == "tuple2":
        m = re.search(rf'{field["name"]}\s*=\s*\(([^,]+),\s*([^)]+)\)', text)
        if not m:
            raise ValueError(f"could not find {field['name']} in {field['file']}")
        return _cast(m.group(field["slot"] + 1), field["cast"])

    if kind == "tier_dict":
        block = _tier_block_span(text, field["tier"]).group(0)
        m = re.search(rf'"{field["key"]}":\s*([0-9.eE+-]+)', block)
        if not m:
            raise ValueError(f"could not find {field['key']} in tier {field['tier']} block")
        return float(m.group(1))

    raise ValueError(f"unknown field kind {kind!r}")


def read_all():
    from schema import FIELDS
    out = {}
    for f in FIELDS:
        try:
            out[f["id"]] = read_field(f)
        except Exception as exc:  # surface as None so one bad field doesn't break the page
            out[f["id"]] = None
            print(f"[config_io] WARN: failed to read {f['id']}: {exc}")
    return out


def write_field(field, new_value):
    text = _read(field["file"])
    kind = field["kind"]
    old_value = read_field(field)

    if kind == "simple":
        pattern = rf'^({field["name"]}\s*=\s*)[0-9.eE+-]+'
        new_text, n = re.subn(pattern, rf"\g<1>{new_value}", text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"could not update {field['name']} in {field['file']}")

    elif kind == "tuple2":
        m = re.search(rf'{field["name"]}\s*=\s*\(([^,]+),\s*([^)]+)\)', text)
        if not m:
            raise ValueError(f"could not find {field['name']} in {field['file']}")
        a, b = m.group(1).strip(), m.group(2).strip()
        if field["slot"] == 0:
            a = str(new_value)
        else:
            b = str(new_value)
        new_text = text[:m.start()] + f'{field["name"]} = ({a}, {b})' + text[m.end():]

    elif kind == "tier_dict":
        m = _tier_block_span(text, field["tier"])
        block = m.group(0)
        new_block, n = re.subn(rf'("{field["key"]}":\s*)[0-9.eE+-]+', rf"\g<1>{new_value}",
                                block, count=1)
        if n != 1:
            raise ValueError(f"could not update {field['key']} in tier {field['tier']} block")
        new_text = text[:m.start()] + new_block + text[m.end():]

    else:
        raise ValueError(f"unknown field kind {kind!r}")

    _write(field["file"], new_text)

    # Verify the write round-trips before committing — refuse to commit a file we can't
    # confirm actually changed to the intended value.
    confirmed = read_field(field)
    if abs(confirmed - float(new_value)) > 1e-9:
        raise ValueError(f"write verification failed: expected {new_value}, file now reads {confirmed}")

    _git_commit(field, old_value, new_value)
    return confirmed


def _git_commit(field, old_value, new_value):
    rel_path = os.path.join("PPO_IBM", field["file"]).replace("\\", "/")
    msg = f"config_studio: {field['id']} {old_value} -> {new_value}"
    try:
        subprocess.run(["git", "add", rel_path], cwd=REPO_ROOT, check=True,
                        capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True,
                        capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # "nothing to commit" (e.g. re-saving the same value) is not an error worth surfacing.
        combined = (exc.stdout or "") + (exc.stderr or "")
        if "nothing to commit" not in combined.lower():
            raise


def git_log(rel_file, n=10):
    try:
        out = subprocess.run(
            ["git", "log", f"-{n}", "--oneline", "--", os.path.join("PPO_IBM", rel_file).replace("\\", "/")],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        return [line for line in out.splitlines() if line.strip()]
    except subprocess.CalledProcessError:
        return []

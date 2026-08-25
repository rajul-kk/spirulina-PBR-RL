"""
config_studio — a Tina-CMS-style visual editor pointed at this project's environment
config instead of a website's content: the curriculum gate thresholds and the
scripted-expert control law, with a "run N episodes and show the result" live preview
in place of a rendered page, and git-backed saves in place of a CMS's content commits.

No new dependencies: built on the standard library's http.server, since this project's
venv doesn't have a web framework installed and this is a small local dev tool, not a
service. Run it and open the printed URL in a browser.

Usage (from repo root, PPO_IBM/):
    python tools/config_studio/server.py [--port 8765]
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_io
import preview_runner
from schema import FIELDS, FIELDS_BY_ID, CURRICULUM_FILE, EXPERT_FILE

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _json_bytes(obj):
    return json.dumps(obj).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[config_studio] {self.address_string()} - {fmt % args}")

    def _send_json(self, obj, status=200):
        body = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        elif self.path == "/app.js":
            self._send_file(os.path.join(STATIC_DIR, "app.js"), "application/javascript; charset=utf-8")
        elif self.path == "/api/fields":
            values = config_io.read_all()
            self._send_json({
                "fields": FIELDS,
                "values": values,
                "curriculum_log": config_io.git_log(CURRICULUM_FILE),
                "expert_log": config_io.git_log(EXPERT_FILE),
            })
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        try:
            if self.path == "/api/fields/update":
                body = self._read_json_body()
                field = FIELDS_BY_ID.get(body.get("id"))
                if not field:
                    return self._send_json({"error": f"unknown field id {body.get('id')!r}"}, status=400)
                confirmed = config_io.write_field(field, body["value"])
                return self._send_json({"id": field["id"], "value": confirmed})

            if self.path == "/api/preview":
                body = self._read_json_body()
                result = preview_runner.run_preview(
                    difficulty=int(body["difficulty"]),
                    n_episodes=int(body.get("n_episodes", 8)),
                    stir_min=float(body["stir_min"]), stir_max=float(body["stir_max"]),
                    light_min=float(body["light_min"]), light_max=float(body["light_max"]),
                    od_setpoint=float(body["od_setpoint"]), gain=float(body["gain"]),
                    frac_cap=float(body["frac_cap"]),
                    gate={"harvest": float(body["gate_harvest"]), "p25": float(body["gate_p25"]),
                          "crash": float(body["gate_crash"]), "od": float(body["gate_od"])},
                )
                return self._send_json(result)

            self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"config_studio running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

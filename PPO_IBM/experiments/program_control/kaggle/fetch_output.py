"""Download a kernel's output like `kaggle kernels output`, but skip large files by name.

The RL sessions' replay buffer (online_buffer.pkl, ~84 MB) repeatedly breaks the CLI download
before train.log is reached; it is only needed to resume on Kaggle, where the previous session's
output is attached directly.

  python fetch_output.py rajulkabir/pbr-rl-v3-lru-s1 ../results/rl_v3/rl-v3-lru-s1/session1 [--skip online_buffer.pkl]
"""
import argparse
import os

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ap = argparse.ArgumentParser()
ap.add_argument("kernel")
ap.add_argument("out")
ap.add_argument("--skip", nargs="*", default=["online_buffer.pkl"])
args = ap.parse_args()

api = KaggleApi()
api.authenticate()
owner, slug = args.kernel.split("/")
with api.build_kaggle_client() as k:
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = owner, slug
    resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
for item in resp.files:
    if os.path.basename(item.file_name) in args.skip:
        print(f"skipped {item.file_name}")
        continue
    dst = os.path.join(args.out, item.file_name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(4):
        try:
            r = requests.get(item.url, timeout=300)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"retry {item.file_name}: {e}")
    else:
        raise SystemExit(f"failed to download {item.file_name}")
    with open(dst, "wb") as f:
        f.write(r.content)
    print(f"{item.file_name} ({len(r.content) // 1024} KB)")
if resp.log:
    with open(os.path.join(args.out, slug + ".log"), "w", encoding="utf-8") as f:
        f.write(resp.log)

#!/usr/bin/env python3
"""Run PyRUA-Lean Coder v0.1-4B over ORA's shared 130-cell set (sweep-130).

One attempt per cell, with the settings of the paper's 4B evaluation (docs/models/coder.md): the in-process
openai loop, at most 40 code cells, 90 minutes, 8,192 output tokens per reply, temperature 1.0, top_p 0.95,
top_k 20, images on demand, pure feedback, RPent's guides, the full primitive set (VLA on).

    python repro/run_sweep.py --cells repro/sweep-130.json --out OUT --base-url http://HOST:PORT/v1 \
        --libero-parallel 8 --robotwin-parallel 4 --cuda-devices 2,3 \
        [--vla-endpoint-libero H:P --sam3-endpoint H:P --vla-endpoint-robotwin H:P] [--only libero|robotwin]
        [--limit N] [--primitives full|no-vla] [--dry-run]

A cell whose OUT/<cell_id>/result.json exists is skipped, so the command resumes. Each episode's stdout and
stderr go to OUT/<cell_id>/play.log. The environment (RPENT_ROOT, asset and checkpoint paths) comes from the
caller: source b2/pyrua-lean/env.sh first.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B"


def cell_id(cell: dict) -> str:
    if cell["environment"] == "libero_pro":
        return f"lp-{cell['suite']}-t{cell['task_id']:02d}-s{cell['seed']}"
    return f"rt-{cell['task']}-s{cell['seed']}"


def argv_for(cell: dict, out: Path, args: argparse.Namespace, device: str) -> list[str]:
    common = [
        "--arm", "cells", "--runtime", "openai", "--base-url", args.base_url,
        "--api-key-env", "OPENAI_API_KEY", "--model", args.model,
        "--openai-max-tokens", "8192", "--openai-tool-choice", "required",
        "--openai-temperature", "1.0", "--openai-extra-body", json.dumps({"top_p": 0.95, "top_k": 20}),
        "--images", "on-demand", "--feedback", "pure", "--max-turns", "40", "--budget-s", "5400",
        "--out", str(out / cell_id(cell)), "--cuda-device", device,
    ]
    if args.guides:
        common += ["--guides"]
    if args.primitives != "full":
        common += ["--primitives", args.primitives]
    if cell["environment"] == "libero_pro":
        argv = ["pyrualean", "play", "--robot", "libero", "--suite", cell["suite"],
                "--task", str(cell["task_id"]), "--seed", str(cell["seed"])] + common
        if args.vla_endpoint_libero:
            argv += ["--vla-endpoint", args.vla_endpoint_libero]
        if args.sam3_endpoint:
            argv += ["--sam3-endpoint", args.sam3_endpoint]
        return argv
    argv = ["pyrualean", "play", "--robot", "robotwin", "--task", cell["task"],
            "--suite", cell.get("task_config", "demo_randomized"), "--seed", str(cell["seed"])] + common
    if args.vla_endpoint_robotwin:
        argv += ["--vla-endpoint", args.vla_endpoint_robotwin]
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--libero-parallel", type=int, default=8)
    ap.add_argument("--robotwin-parallel", type=int, default=4)
    ap.add_argument("--cuda-devices", default="0", help="cards the episodes' simulators run on, round robin")
    ap.add_argument("--vla-endpoint-libero")
    ap.add_argument("--sam3-endpoint")
    ap.add_argument("--vla-endpoint-robotwin")
    ap.add_argument("--only", choices=["libero", "robotwin"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--primitives", default="full", choices=["full", "no-vla"])
    ap.add_argument("--no-guides", dest="guides", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = json.loads(args.cells.read_text())["cells"]
    if args.only:
        cells = [c for c in cells if (c["environment"] == "libero_pro") == (args.only == "libero")]
    cells = [c for c in cells if not (args.out / cell_id(c) / "result.json").exists()]
    if args.limit:
        cells = cells[: args.limit]
    devices = itertools.cycle(args.cuda_devices.split(","))
    lock = threading.Lock()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cells)} cells to run", flush=True)

    def run(cell: dict) -> tuple[str, int, float]:
        with lock:
            device = next(devices)
        cid = cell_id(cell)
        argv = argv_for(cell, args.out, args, device)
        if args.dry_run:
            print(" ".join(argv))
            return cid, 0, 0.0
        (args.out / cid).mkdir(parents=True, exist_ok=True)
        started = time.time()
        with (args.out / cid / "play.log").open("w") as log:
            code = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        wall = time.time() - started
        success = None
        result = args.out / cid / "result.json"
        if result.exists():
            success = json.loads(result.read_text()).get("environment_success")
        print(f"{time.strftime('%H:%M:%S')} {cid} exit={code} success={success} wall={wall:.0f}s", flush=True)
        return cid, code, wall

    libero = [c for c in cells if c["environment"] == "libero_pro"]
    robotwin = [c for c in cells if c["environment"] != "libero_pro"]
    with ThreadPoolExecutor(max(1, args.libero_parallel)) as lp, ThreadPoolExecutor(max(1, args.robotwin_parallel)) as rt:
        futures = [lp.submit(run, c) for c in libero] + [rt.submit(run, c) for c in robotwin]
        failed = [f.result()[0] for f in futures if f.result()[1] != 0]
    print(f"done; non-zero exits: {len(failed)} {failed[:10]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

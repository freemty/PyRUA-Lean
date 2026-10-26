#!/usr/bin/env python3
"""One row per sweep-130 cell: the 4B run and the VLA-only control side by side (CSV on stdout).

    python repro/export_index.py --main OUT/full --control OUT/control --cells repro/sweep-130.json > index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

VLA = {"lp": {"pi0_pick", "pi0_doubled"}, "rt": {"lingbot_act"}}
GRIPPER = {"set_gripper", "release"}
DIFFERENT_SCENE = ("libero_spatial_task", "libero_10_task")
UNRUNNABLE = {"lp-libero_spatial_task-t03-s1", "lp-libero_spatial_task-t07-s1", "lp-libero_10_task-t02-s1"}


def cell_id(cell: dict) -> str:
    if cell["environment"] == "libero_pro":
        return f"lp-{cell['suite']}-t{cell['task_id']:02d}-s{cell['seed']}"
    return f"rt-{cell['task']}-s{cell['seed']}"


def moved(c: dict) -> bool:
    return bool(c.get("stateful")) and (c.get("env_steps") or 0) > 0


def episode(run: Path, key: str) -> dict:
    out = {"status": "", "success": "", "cells": "", "robo_calls": "", "vla_calls": "", "scripted_motions": "",
           "vla_step_share": "", "closing_call": "", "input_tokens": "", "wall_s": ""}
    path = run / "result.json"
    if not path.exists():
        return out
    result = json.loads(path.read_text())
    out["status"] = result.get("status", "")
    if result.get("status") == "boot_error":
        return out
    calls = [json.loads(l) for l in (run / "calls.jsonl").read_text().splitlines() if l.strip()] \
        if (run / "calls.jsonl").exists() else []
    turns = json.loads((run / "tool_turns.json").read_text()) if (run / "tool_turns.json").exists() else []
    steps = sum(c.get("env_steps") or 0 for c in calls)
    vsteps = sum(c.get("env_steps") or 0 for c in calls if c["tool"] in VLA[key])
    close = None
    for i, c in enumerate(calls):
        if (c.get("summary") or {}).get("terminated"):
            close = c if moved(c) else next((p for p in reversed(calls[:i + 1]) if moved(p)), c)
            break
    out.update(
        success=int(bool(result.get("environment_success"))),
        cells=sum(t.get("tool") == "python" for t in turns) if turns else "",
        robo_calls=len(calls),
        vla_calls=sum(c["tool"] in VLA[key] for c in calls),
        scripted_motions=sum(moved(c) and c["tool"] not in VLA[key] and c["tool"] not in GRIPPER for c in calls),
        vla_step_share=f"{vsteps / steps:.3f}" if steps else "",
        closing_call=close["tool"] if close else "",
        input_tokens=(result.get("generation") or {}).get("input_tokens") or "",
        wall_s=f"{result.get('total_wall_s', 0):.0f}",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    args = ap.parse_args()
    cells = json.loads(args.cells.read_text())["cells"]
    fields = ["cell", "environment", "suite_or_task", "task_id", "seed", "scene_vs_ora"]
    per = ["status", "success", "cells", "robo_calls", "vla_calls", "scripted_motions", "vla_step_share",
           "closing_call", "input_tokens", "wall_s"]
    header = fields + [f"coder4b_{f}" for f in per] + [f"vla_only_{f}" for f in ("status", "success", "robo_calls",
                                                                                   "vla_calls", "wall_s")]
    w = csv.writer(sys.stdout)
    w.writerow(header)
    for cell in cells:
        cid = cell_id(cell)
        key = cid[:2]
        if cid in UNRUNNABLE:
            scene = "not runnable on rpent-liberopro 0.2.0"
        elif key == "lp" and cell["suite"].startswith(DIFFERENT_SCENE):
            scene = "different initial state"
        else:
            scene = "same"
        m = episode(args.main / cid, key)
        c = episode(args.control / cid, key)
        w.writerow([cid, cell["environment"], cell.get("suite") or cell.get("task"), cell.get("task_id", ""),
                    cell["seed"], scene] + [m[f] for f in per] +
                   [c[f] for f in ("status", "success", "robo_calls", "vla_calls", "wall_s")])


if __name__ == "__main__":
    main()

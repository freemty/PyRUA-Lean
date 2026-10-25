#!/usr/bin/env python3
"""Summarise a run_sweep.py output: success, VLA dependence and primitive-call density per episode.

    python repro/analyze.py OUT [--json OUT/summary.json] [--md OUT/summary.md]

Per episode (from result.json, calls.jsonl and tool_turns.json):

- success: the benchmark's verdict (``environment_success``);
- cells: ``python`` turns; calls: every ``robo`` call; stateful: calls that stepped the simulator;
- vla_calls / vla_steps: calls of the robot's VLA primitives and the simulator steps they took,
  against all stateful calls and all steps;
- closing_call: the call during which the task predicate first fired (``summary.terminated``),
  and whether it was a VLA call;
- calls_per_cell: the robo calls each cell made (needs the fork's ``ledger_span``);
- pattern: ``no_vla`` (no VLA call), ``vla_closed`` (the predicate fired inside a VLA call),
  ``hybrid_scripted_close`` (VLA used, predicate fired in a scripted call), ``vla_only_motion`` (every
  stateful call was VLA or a gripper command), or ``unsolved`` subtypes by the same rules.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VLA = {"libero": {"pi0_pick", "pi0_doubled"}, "robotwin": {"lingbot_act"}}
GRIPPER = {"set_gripper", "release"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episode(run: Path) -> dict | None:
    result_path = run / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    robot = "libero" if run.name.startswith("lp-") else "robotwin"
    vla = VLA[robot]
    calls = load_jsonl(run / "calls.jsonl")
    turns = json.loads((run / "tool_turns.json").read_text()) if (run / "tool_turns.json").exists() else []
    cells = [t for t in turns if t.get("tool") == "python"]
    stateful = [c for c in calls if c.get("stateful")]
    vla_calls = [c for c in calls if c.get("tool") in vla]
    steps = sum(c.get("env_steps") or 0 for c in calls)
    vla_steps = sum(c.get("env_steps") or 0 for c in vla_calls)
    closing = next((c for c in calls if (c.get("summary") or {}).get("terminated")), None)
    per_cell = [t["ledger_span"][1] - t["ledger_span"][0] for t in cells if "ledger_span" in t]
    motion_non_vla = [c for c in stateful if c["tool"] not in vla and c["tool"] not in GRIPPER]
    if not vla_calls:
        pattern = "no_vla"
    elif closing is not None and closing["tool"] in vla:
        pattern = "vla_closed"
    elif not motion_non_vla:
        pattern = "vla_only_motion"
    elif closing is not None:
        pattern = "hybrid_scripted_close"
    else:
        pattern = "hybrid_unsolved"
    generation = result.get("generation") or {}
    return {
        "cell": run.name,
        "robot": robot,
        "success": bool(result.get("environment_success")),
        "status": result.get("status"),
        "cells": len(cells),
        "calls": len(calls),
        "stateful": len(stateful),
        "vla_calls": len(vla_calls),
        "vla_share_of_stateful": (len(vla_calls) / len(stateful)) if stateful else None,
        "steps": steps,
        "vla_step_share": (vla_steps / steps) if steps else None,
        "scripted_motion_calls": len(motion_non_vla),
        "closing_tool": closing["tool"] if closing else None,
        "closed_by_vla": bool(closing and closing["tool"] in vla),
        "calls_per_cell": per_cell,
        "tool_mix": dict(Counter(c["tool"] for c in calls)),
        "pattern": pattern,
        "input_tokens": generation.get("input_tokens"),
        "output_tokens": generation.get("output_tokens"),
        "llm_requests": generation.get("requests"),
        "wall_s": result.get("total_wall_s"),
    }


def median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for robot in ("libero", "robotwin", "all"):
        group = [r for r in rows if robot == "all" or r["robot"] == robot]
        if not group:
            continue
        won = [r for r in group if r["success"]]
        out[robot] = {
            "episodes": len(group),
            "success": len(won),
            "patterns_all": dict(Counter(r["pattern"] for r in group)),
            "patterns_success": dict(Counter(r["pattern"] for r in won)),
            "success_closed_by_vla": sum(r["closed_by_vla"] for r in won),
            "episodes_using_vla": sum(r["vla_calls"] > 0 for r in group),
            "median_cells": median([r["cells"] for r in group]),
            "median_calls": median([r["calls"] for r in group]),
            "median_calls_per_cell": median([c for r in group for c in r["calls_per_cell"]]),
            "median_vla_share_of_stateful": median([r["vla_share_of_stateful"] for r in group]),
            "median_vla_step_share": median([r["vla_step_share"] for r in group]),
            "median_vla_step_share_success": median([r["vla_step_share"] for r in won]),
            "median_input_tokens": median([r["input_tokens"] for r in group]),
            "tool_mix": dict(sum((Counter(r["tool_mix"]) for r in group), Counter()).most_common(15)),
        }
    return out


def markdown(summary: dict, rows: list[dict]) -> str:
    lines = ["| | episodes | success | used VLA | success closed by VLA | median cells | median calls/cell | median VLA share of stateful calls | median VLA share of sim steps |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for robot, s in summary.items():
        lines.append(f"| {robot} | {s['episodes']} | {s['success']} | {s['episodes_using_vla']} | {s['success_closed_by_vla']} | "
                     f"{s['median_cells']} | {s['median_calls_per_cell']} | {s['median_vla_share_of_stateful']} | {s['median_vla_step_share']} |")
    lines += ["", "| cell | success | pattern | cells | calls | VLA calls | scripted motion | closing call | VLA step share |",
              "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for r in sorted(rows, key=lambda r: r["cell"]):
        share = "" if r["vla_step_share"] is None else f"{r['vla_step_share']:.2f}"
        lines.append(f"| {r['cell']} | {'yes' if r['success'] else 'no'} | {r['pattern']} | {r['cells']} | {r['calls']} | "
                     f"{r['vla_calls']} | {r['scripted_motion_calls']} | {r['closing_tool'] or ''} | {share} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    rows = [e for run in sorted(args.out.iterdir()) if run.is_dir() and (e := episode(run))]
    summary = summarise(rows)
    (args.json or args.out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    (args.md or args.out / "summary.md").write_text(markdown(summary, rows))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

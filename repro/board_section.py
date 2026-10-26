#!/usr/bin/env python3
"""Build the board section for the reproduction from the result directories (HTML fragment on stdout).

    python repro/board_section.py --main OUT/full --control OUT/control > section.html
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import statistics as st
from collections import Counter
from pathlib import Path

VLA = {"lp": {"pi0_pick", "pi0_doubled"}, "rt": {"lingbot_act"}}
GRIPPER = {"set_gripper", "release"}
# Six LIBERO-PRO suites are byte-identical to ORA's data in RPent's package; two are not (NOTES.md).
DIFFERENT_SCENE = ("libero_spatial_task", "libero_10_task")
UNRUNNABLE = {"lp-libero_spatial_task-t03-s1", "lp-libero_spatial_task-t07-s1", "lp-libero_10_task-t02-s1"}


def load(root: Path) -> dict[str, dict]:
    out = {}
    for path in glob.glob(str(root / "*" / "result.json")):
        run = Path(path).parent
        result = json.loads(Path(path).read_text())
        if result.get("status") == "boot_error":
            continue
        calls = [json.loads(l) for l in (run / "calls.jsonl").read_text().splitlines() if l.strip()] \
            if (run / "calls.jsonl").exists() else []
        turns = json.loads((run / "tool_turns.json").read_text()) if (run / "tool_turns.json").exists() else []
        out[run.name] = {"run": run, "result": result, "calls": calls,
                         "turns": [t for t in turns if t.get("tool") == "python"]}
    return out


def closing_call(calls: list[dict]) -> dict | None:
    for i, c in enumerate(calls):
        if (c.get("summary") or {}).get("terminated"):
            moved = lambda x: x.get("stateful") and (x.get("env_steps") or 0) > 0
            return c if moved(c) else next((p for p in reversed(calls[:i + 1]) if moved(p)), c)
    return None


def q(values: list[float]) -> str:
    v = sorted(values)
    if not v:
        return "–"
    n = len(v)
    return f"{st.median(v):g} <span class=\"note\">({v[n // 4]:g}–{v[3 * n // 4]:g})</span>"


def pct(a: int, b: int) -> str:
    return f"{a}/{b} <span class=\"note\">({100 * a / b:.0f}%)</span>" if b else "–"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def prompt_kind(prompt: str, task: str) -> str:
    p, t = norm(prompt), norm(task)
    if p == t:
        return "原句"
    if p and (p in t or t in p):
        return "原句的片段"
    tw, pw = set(t.split()), set(p.split())
    return "改写（词重合 ≥60%）" if pw and len(tw & pw) / len(pw) >= 0.6 else "自拟"


def per_episode(e: dict, key: str) -> dict:
    calls = e["calls"]
    vla = VLA[key]
    stateful = [c for c in calls if c.get("stateful")]
    steps = sum(c.get("env_steps") or 0 for c in calls)
    vsteps = sum(c.get("env_steps") or 0 for c in calls if c["tool"] in vla)
    scripted = [c for c in stateful if c["tool"] not in vla and c["tool"] not in GRIPPER and (c.get("env_steps") or 0) > 0]
    close = closing_call(calls)
    lines, per_cell, moving = [], [], []
    for i, t in enumerate(e["turns"]):
        src_path = e["run"] / "workspace" / f"cell_{i + 1:03d}.py"
        src = src_path.read_text() if src_path.exists() else ""
        lines.append(len([l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]))
        a, b = t.get("ledger_span", [0, 0])
        per_cell.append(b - a)
        moving.append(sum(1 for c in calls[a:b] if c.get("stateful") and (c.get("env_steps") or 0) > 0))
    return {
        "success": bool(e["result"].get("environment_success")),
        "cells": len(e["turns"]),
        "vla_calls": sum(c["tool"] in vla for c in calls),
        "vla_step_share": vsteps / steps if steps else None,
        "scripted": len(scripted),
        # No call carried the predicate (the verdict came at the end): with no scripted motion, the VLA did it.
        "closed_by_vla": bool(close and close["tool"] in vla) or (close is None and not scripted),
        "closed_by_script": bool(close and close["tool"] not in vla),
        "lines": lines, "per_cell": per_cell, "moving": moving,
        "input_tokens": (e["result"].get("generation") or {}).get("input_tokens"),
    }


def section(main: dict, control: dict) -> str:
    out = []
    rows = []
    for key, name in (("lp", "LIBERO-Pro"), ("rt", "RoboTwin")):
        m = {k: v for k, v in main.items() if k.startswith(key + "-")}
        c = {k: v for k, v in control.items() if k.startswith(key + "-")}
        paired = sorted(set(m) & set(c))
        if key == "lp":
            same = [k for k in paired if not k.split("-", 1)[1].startswith(DIFFERENT_SCENE)]
        else:
            same = paired
        ms = sum(m[k]["result"]["environment_success"] for k in paired)
        cs = sum(c[k]["result"]["environment_success"] for k in paired)
        only_m = sum(m[k]["result"]["environment_success"] and not c[k]["result"]["environment_success"] for k in paired)
        only_c = sum(c[k]["result"]["environment_success"] and not m[k]["result"]["environment_success"] for k in paired)
        sm = sum(m[k]["result"]["environment_success"] for k in same)
        sc = sum(c[k]["result"]["environment_success"] for k in same)
        rows.append((name, len(m), sum(v["result"]["environment_success"] for v in m.values()), len(paired), ms, cs,
                     only_m, only_c, len(same), sm, sc))
    out.append('<div class="tablewrap"><table><thead><tr><th></th><th class="num">4B 跑完</th><th class="num">4B 成功</th>'
               '<th class="num">配对格数</th><th class="num">4B 成功</th><th class="num">只用 VLA 成功</th>'
               '<th class="num">只有 4B 成</th><th class="num">只有 VLA 成</th></tr></thead><tbody>')
    for r in rows:
        out.append(f"<tr><td>{r[0]}</td><td class=\"num\">{r[1]}</td><td class=\"num\">{pct(r[2], r[1])}</td>"
                   f"<td class=\"num\">{r[3]}</td><td class=\"num\">{pct(r[4], r[3])}</td><td class=\"num\">{pct(r[5], r[3])}</td>"
                   f"<td class=\"num\">{r[6]}</td><td class=\"num\">{r[7]}</td></tr>")
    out.append("</tbody></table></div>")
    lp = rows[0]
    out.append(f'<p class="note">LIBERO 的配对里，场景与 ORA 逐字节一致的 {lp[8]} 格：4B {lp[9]}，只用 VLA {lp[10]}。'
               '另外 3 格在 RPent 的 LIBERO-PRO 里没有初始状态，跑不了；两个 suite 的初始场景与 ORA 不同（NOTES.md）。</p>')

    # Patterns and density.
    out.append('<div class="tablewrap"><table><thead><tr><th></th><th class="num">成功局里判定在 VLA 调用中触发</th>'
               '<th class="num">成功局里完全没有脚本动作</th><th class="num">成功局 VLA 步数占比（中位）</th>'
               '<th class="num">每局轮数 · 成功</th><th class="num">每局轮数 · 失败</th><th class="num">每局 VLA 调用 · 成功</th>'
               '<th class="num">每局 VLA 调用 · 失败</th><th class="num">每轮代码行</th><th class="num">每轮原语调用</th>'
               '<th class="num">每轮动机器人的调用</th></tr></thead><tbody>')
    for key, name in (("lp", "LIBERO-Pro"), ("rt", "RoboTwin")):
        eps = [per_episode(v, key) for k, v in main.items() if k.startswith(key + "-")]
        won = [e for e in eps if e["success"]]
        lost = [e for e in eps if not e["success"]]
        out.append(
            f"<tr><td>{name}</td><td class=\"num\">{pct(sum(e['closed_by_vla'] for e in won), len(won))}</td>"
            f"<td class=\"num\">{pct(sum(e['scripted'] == 0 for e in won), len(won))}</td>"
            f"<td class=\"num\">{st.median([e['vla_step_share'] for e in won if e['vla_step_share'] is not None]):.2f}</td>"
            f"<td class=\"num\">{q([e['cells'] for e in won])}</td><td class=\"num\">{q([e['cells'] for e in lost])}</td>"
            f"<td class=\"num\">{q([e['vla_calls'] for e in won])}</td><td class=\"num\">{q([e['vla_calls'] for e in lost])}</td>"
            f"<td class=\"num\">{q([x for e in eps for x in e['lines']])}</td>"
            f"<td class=\"num\">{q([x for e in eps for x in e['per_cell']])}</td>"
            f"<td class=\"num\">{q([x for e in eps for x in e['moving']])}</td></tr>")
    out.append("</tbody></table></div>")

    # LIBERO prompt kinds.
    kinds, closing_kinds = Counter(), Counter()
    for k, v in main.items():
        if not k.startswith("lp-"):
            continue
        task = v["result"].get("task_language") or ""
        close = closing_call(v["calls"])
        for c in v["calls"]:
            if c["tool"] in VLA["lp"]:
                kind = prompt_kind(str(c["kwargs"].get("prompt", "")), task)
                kinds[kind] += 1
                if c is close and v["result"]["environment_success"]:
                    closing_kinds[kind] += 1
    total, ctotal = sum(kinds.values()), sum(closing_kinds.values())
    out.append('<div class="tablewrap"><table><thead><tr><th>π0.5 提示词与任务原句的关系</th>'
               '<th class="num">所有 VLA 调用</th><th class="num">成功局的收尾调用</th></tr></thead><tbody>')
    for kind in ("原句", "原句的片段", "改写（词重合 ≥60%）", "自拟"):
        out.append(f"<tr><td>{kind}</td><td class=\"num\">{pct(kinds[kind], total)}</td>"
                   f"<td class=\"num\">{pct(closing_kinds[kind], ctotal)}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True)
    args = ap.parse_args()
    print(section(load(args.main), load(args.control)))


if __name__ == "__main__":
    main()

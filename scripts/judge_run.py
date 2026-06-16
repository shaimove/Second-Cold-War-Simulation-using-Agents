#!/usr/bin/env python3
"""Run quality monitor (gates + LLM judge) on saved scenario runs."""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from app import db
from app.monitor.service import judge_saved_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge saved scenario runs")
    parser.add_argument("run_id", nargs="?", help="Single run_id to judge")
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        metavar="N",
        help="Judge the N most recent saved runs",
    )
    args = parser.parse_args()

    run_ids = []
    if args.run_id:
        run_ids = [args.run_id]
    elif args.last > 0:
        rows = db.list_scenario_runs(limit=args.last)
        run_ids = [r["run_id"] for r in rows]
    else:
        parser.print_help()
        return 1

    for rid in run_ids:
        updated = judge_saved_run(rid)
        if updated is None:
            print("NOT FOUND:", rid)
            continue
        mon = updated.get("monitor") or {}
        gates = mon.get("gates") or {}
        judge = mon.get("judge") or {}
        score = judge.get("overall_score", "—")
        passed = gates.get("passed")
        print(
            rid,
            "| gates=" + ("ok" if passed else "FAIL"),
            "| score=" + str(score),
            "|",
            (judge.get("one_line_verdict") or mon.get("judge_skip_reason") or "")[:80],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

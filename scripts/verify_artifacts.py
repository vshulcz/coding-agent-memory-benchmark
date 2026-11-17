#!/usr/bin/env python3
"""Validate committed benchmark artifacts and print verification tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_RETRIEVAL_KEYS = {
    "benchmark",
    "harness",
    "run_at",
    "dataset",
    "questions",
    "wall_seconds",
    "median_search_ms",
    "rows",
    "total",
}
REQUIRED_ROW_KEYS = {"name", "n", "hit@1", "hit@5", "hit@10", "hit@20", "mrr"}
TOLERANCE = 1e-9


def verify_retrieval_artifact(path: Path) -> dict[str, Any]:
    """Validate structure, row count sum, and weighted totals for a retrieval run."""
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_RETRIEVAL_KEYS - set(data))
    if missing:
        raise ValueError(f"{path.name}: missing top-level keys: {missing}")

    ds = data["dataset"]
    for key in ("name", "bytes", "sha256"):
        if key not in ds:
            raise ValueError(f"{path.name}: dataset missing {key}")
    if len(ds["sha256"]) != 64 or ds["bytes"] <= 0:
        raise ValueError(f"{path.name}: invalid dataset metadata {ds}")

    rows = data["rows"]
    total = data["total"]
    if not rows:
        raise ValueError(f"{path.name}: rows must be non-empty")

    for row in [*rows, total]:
        row_missing = sorted(REQUIRED_ROW_KEYS - set(row))
        if row_missing:
            raise ValueError(f"{path.name}: row {row.get('name')} missing {row_missing}")

    row_sum = sum(r["n"] for r in rows)
    if row_sum != total["n"] or total["n"] != data["questions"]:
        raise ValueError(
            f"{path.name}: count mismatch sum(rows)={row_sum}, "
            f"total.n={total['n']}, questions={data['questions']}"
        )

    for metric in ("hit@1", "hit@5", "hit@10", "hit@20", "mrr"):
        weighted = sum(r["n"] * r[metric] for r in rows) / total["n"]
        if abs(weighted - total[metric]) > TOLERANCE:
            raise ValueError(
                f"{path.name}: weighted {metric}={weighted:.12f} != total={total[metric]:.12f}"
            )
    for row in [*rows, total]:
        if not (row["hit@1"] <= row["hit@5"] <= row["hit@10"] <= row["hit@20"]):
            raise ValueError(f"{path.name}: non-monotonic hit@k in {row['name']}")
    return data


def verify_day_zero_artifact(path: Path) -> dict[str, Any]:
    """Validate Day-Zero 19,195-session, task-cost, and compaction numbers."""
    data = json.loads(path.read_text(encoding="utf-8"))
    q = data["questions"]
    if data["sessions_on_disk"] != 19195 or q != 100:
        raise ValueError(f"{path.name}: unexpected corpus parameters")

    tools = data["tools"]
    if len(tools) != 7:
        raise ValueError(f"{path.name}: expected 7 tools, got {len(tools)}")
    for tool in tools:
        h1, h5, f50 = tool["hit_at_1"], tool["hit_at_5"], tool["found_at_50"]
        if not (0 <= h1 <= h5 <= f50 <= q):
            raise ValueError(f"{path.name}: invalid ranks for {tool['name']}: {h1}, {h5}, {f50}")
        if tool["mcp_schema_tokens_o200k"] < 0 or tool["mcp_tool_count"] < 0:
            raise ValueError(f"{path.name}: negative MCP tool count for {tool['name']}")

    ctx = data["context_readiness"]
    if not (ctx["deja_median_tokens"] < ctx["full_session_replay_median_tokens"] < ctx["naive_raw_log_grep_median_tokens"]):
        raise ValueError(f"{path.name}: unexpected context_readiness ordering")
    return data


def format_retrieval_table(data: dict[str, Any]) -> str:
    """Render one retrieval artifact as an aligned plain-text table."""
    lines = [
        f"=== {data['benchmark']} (n={data['questions']}, dataset={data['dataset']['name']}, median={data['median_search_ms']:.3f} ms) ===",
        f"{'slice':<26} {'n':>5} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'hit@20':>8} {'MRR':>7}",
    ]
    for r in [*data["rows"], data["total"]]:
        lines.append(
            f"{r['name']:<26} {r['n']:>5d} {r['hit@1']:>7.1f}% {r['hit@5']:>7.1f}% "
            f"{r['hit@10']:>7.1f}% {r['hit@20']:>7.1f}% {r['mrr']:>7.3f}"
        )
    ev = data.get("extra", {}).get("evidence_recall")
    if ev:
        lines.append(
            f"{'evidence_recall':<26} {'':>5} {ev['@1']:>7.1f}% {ev['@5']:>7.1f}% "
            f"{ev['@10']:>7.1f}% {ev['@20']:>7.1f}%"
        )
    return "\n".join(lines)


def format_day_zero_table(data: dict[str, Any]) -> str:
    """Render Day-Zero comparison metrics as an aligned table."""
    lines = [
        f"=== {data['benchmark']} ({data['sessions_on_disk']} sessions, {data['questions']} questions) ===",
        f"{'tool':<13} {'mcp_tok(tools)':>14} {'build_s':>8} {'first_ms':>9} {'p50_ms':>8} {'h@1':>4} {'h@5':>4} {'f@50':>5}",
    ]
    for t in data["tools"]:
        mcp = f"{t['mcp_schema_tokens_o200k']} ({t['mcp_tool_count']})"
        bld = "-" if t["build_seconds"] is None else f"{t['build_seconds']:.1f}"
        fst = "-" if t["first_answer_ms"] is None else f"{t['first_answer_ms']}"
        p50 = "-" if t["p50_search_ms"] is None else f"{t['p50_search_ms']}"
        lines.append(
            f"{t['name']:<13} {mcp:>14} {bld:>8} {fst:>9} {p50:>8} "
            f"{t['hit_at_1']:>4d} {t['hit_at_5']:>4d} {t['found_at_50']:>5d}"
        )
    return "\n".join(lines)


def verify_directory(results_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify all committed JSON artifacts under results_dir."""
    retrieval_names = (
        "longmemeval-s-cleaned.json",
        "longmemeval-s-full.json",
        "locomo.json",
    )
    retrieval = [verify_retrieval_artifact(results_dir / name) for name in retrieval_names]
    day_zero = verify_day_zero_artifact(results_dir / "day-zero-19195-sessions.json")
    return retrieval, day_zero


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify committed benchmark JSON artifacts.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Path to the results/ directory",
    )
    args = parser.parse_args(argv)
    try:
        retrieval, day_zero = verify_directory(args.results_dir)
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

    for art in retrieval:
        print(format_retrieval_table(art))
        print()
    print(format_day_zero_table(day_zero))
    print("\nALL_ARTIFACTS_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

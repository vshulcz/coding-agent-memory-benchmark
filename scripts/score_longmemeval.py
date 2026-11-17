#!/usr/bin/env python3
"""Score any external system's LongMemEval-S session ranking using exact hit@k, MRR, and official evidence-recall."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def describe_dataset(path: Path) -> dict[str, Any]:
    """Compute file name, byte size, and SHA-256 hex digest for the dataset."""
    h = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            total_bytes += len(chunk)
    return {"name": path.name, "bytes": total_bytes, "sha256": h.hexdigest()}


def load_submission(path: Path) -> dict[str, list[str]]:
    """Load flat {qid: [session_ids]} or wrapped {'answers': {qid: [session_ids]}} JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "answers" in raw and isinstance(raw["answers"], dict):
        sub = raw["answers"]
    elif isinstance(raw, dict):
        sub = raw
    else:
        raise ValueError(f"{path} is neither {{question_id: [ids]}} nor {{'answers': ...}}")
    if not sub:
        raise ValueError(f"{path} holds no answers")
    return sub


def rank_in(ranked: list[str], evidence: list[str]) -> int:
    """Return 1-based rank of the first evidence session in ranked, or 0 if absent."""
    want = set(evidence)
    for idx, sid in enumerate(ranked):
        if sid in want:
            return idx + 1
    return 0


def evidence_recall_in(ranked: list[str], evidence: list[str], k: int) -> float:
    """Return the fraction of evidence sessions present in the top k ranked results."""
    if not evidence:
        return 0.0
    top = set(ranked[:k])
    found = sum(1 for sid in evidence if sid in top)
    return found / len(evidence)


def pct(hits: int, n: int) -> float:
    return (100.0 * hits / n) if n else 0.0


def score_submission(
    questions: list[dict[str, Any]],
    sub: dict[str, list[str]],
    dataset_meta: dict[str, Any],
    submission_label: str,
) -> tuple[dict[str, Any], list[str]]:
    """Score submission against loaded questions and return (result_dict, missing_qids)."""
    start = time.monotonic()
    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0, "r20": 0, "miss": 0, "mrr": 0.0}
    )
    total = {"n": 0, "r1": 0, "r5": 0, "r10": 0, "r20": 0, "miss": 0, "mrr": 0.0}
    ev_sum = {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
    ev_n = 0
    missing: list[str] = []

    for q in questions:
        qid = q["question_id"]
        qtype = q["question_type"]
        evidence = q.get("answer_session_ids") or []
        if qid not in sub:
            missing.append(qid)
        ranked = sub.get(qid) or []
        rank = rank_in(ranked, evidence)

        for target in (buckets[qtype], total):
            target["n"] += 1
            if rank >= 1:
                target["mrr"] += 1.0 / rank
            if rank == 0:
                target["miss"] += 1
                continue
            for k, field in ((1, "r1"), (5, "r5"), (10, "r10"), (20, "r20")):
                if rank <= k:
                    target[field] += 1

        for k in (1, 5, 10, 20):
            ev_sum[k] += evidence_recall_in(ranked, evidence, k)
        ev_n += 1

    def to_row(name: str, b: dict[str, float]) -> dict[str, Any]:
        n = int(b["n"])
        return {
            "name": name,
            "n": n,
            "hit@1": pct(int(b["r1"]), n),
            "hit@5": pct(int(b["r5"]), n),
            "hit@10": pct(int(b["r10"]), n),
            "hit@20": pct(int(b["r20"]), n),
            "mrr": (b["mrr"] / n) if n else 0.0,
        }

    result: dict[str, Any] = {
        "benchmark": "LongMemEval-S",
        "harness": "scripts/score_longmemeval.py",
        "run_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "dataset": dataset_meta,
        "flags": {"submission": submission_label},
        "questions": int(total["n"]),
        "wall_seconds": time.monotonic() - start,
        "median_search_ms": 0.0,
        "rows": [to_row(t, buckets[t]) for t in sorted(buckets)],
        "total": to_row("TOTAL", total),
        "notes": {
            "metric": "hit@k credits a question when any of its evidence sessions ranks in the top k",
            "official": "evidence_recall is the stricter per-evidence metric the dataset defines",
            "source": "external submission scored with shared arithmetic",
        },
    }
    if ev_n > 0:
        result["extra"] = {
            "evidence_recall": {f"@{k}": 100.0 * ev_sum[k] / ev_n for k in (1, 5, 10, 20)}
        }
    return result, missing


def format_report(result: dict[str, Any], submission_path: str, missing: list[str]) -> str:
    """Format the scored summary matching scripts/longmemeval output."""
    ds = result["dataset"]
    lines = [
        f"LongMemEval-S · submitted ranking from {submission_path}",
        f"questions: {result['questions']} · dataset {ds['name']} · sha256 {ds['sha256']}",
    ]
    if missing:
        lines.append(
            f"unanswered: {len(missing)} of {result['questions']} questions carry no ranking — counted as misses"
        )
    lines.append("")
    lines.append(f"{'type':<28} {'n':>6} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8} {'hit@20':>8} {'MRR':>8}")
    for row in [*result["rows"], result["total"]]:
        lines.append(
            f"{row['name']:<28} {row['n']:>6d} {row['hit@1']:>7.1f}% {row['hit@5']:>7.1f}% "
            f"{row['hit@10']:>7.1f}% {row['hit@20']:>7.1f}%   {row['mrr']:.3f}"
        )
    ev = result.get("extra", {}).get("evidence_recall")
    if ev:
        lines.append(
            f"{'evidence-recall (official)':<28} {'':>6} {ev['@1']:>7.1f}% {ev['@5']:>7.1f}% "
            f"{ev['@10']:>7.1f}% {ev['@20']:>7.1f}%"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a LongMemEval-S retrieval submission JSON.")
    parser.add_argument("--data", type=Path, required=True, help="Path to longmemeval_s.json")
    parser.add_argument("--score", type=Path, required=True, help="Path to submission JSON")
    parser.add_argument("--skip-abs", action="store_true", help="Drop abstention (_abs) questions")
    parser.add_argument("--out", type=Path, help="Optional path to write benchresult JSON")
    args = parser.parse_args(argv)

    questions = json.loads(args.data.read_text(encoding="utf-8"))
    if args.skip_abs:
        questions = [q for q in questions if "_abs" not in q.get("question_id", "")]

    sub = load_submission(args.score)
    ds = describe_dataset(args.data)
    result, missing = score_submission(questions, sub, ds, str(args.score))
    print(format_report(result, str(args.score), missing))

    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

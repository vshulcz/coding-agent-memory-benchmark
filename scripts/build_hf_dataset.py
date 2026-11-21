#!/usr/bin/env python3
"""Build normalized Hugging Face Dataset JSONL splits and card from results/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FORBIDDEN_TEXT_KEYS = frozenset(
    {
        "conversation",
        "conversations",
        "dialogue",
        "dialogues",
        "haystack_sessions",
        "message",
        "messages",
        "prompt",
        "question",
        "session",
        "sessions",
        "text",
        "transcript",
    }
)

DATASET_CARD = """\
---
license: mit
pretty_name: Coding Agent Memory Benchmark
task_categories:
  - text-retrieval
  - question-answering
language:
  - en
size_categories:
  - n<1K
tags:
  - coding-agents
  - agent-memory
  - longmemeval
  - locomo
  - bm25
  - model-context-protocol
  - deja-vu
configs:
  - config_name: longmemeval_slices
    data_files:
      - split: test
        path: data/longmemeval_slices.jsonl
  - config_name: locomo_retrieval_metrics
    data_files:
      - split: test
        path: data/locomo_retrieval_metrics.jsonl
  - config_name: day_zero_tools
    data_files:
      - split: test
        path: data/day_zero_tools.jsonl
  - config_name: task_cost_and_compaction
    data_files:
      - split: test
        path: data/task_cost_and_compaction.jsonl
---

# Coding Agent Memory Benchmark

Machine-readable summary splits generated from the verified JSON artifacts in [`vshulcz/coding-agent-memory-benchmark`](https://github.com/vshulcz/coding-agent-memory-benchmark) (Zenodo archive: [`10.5281/zenodo.24416712`](https://doi.org/10.5281/zenodo.24416712)) for [`vshulcz/deja-vu`](https://github.com/vshulcz/deja-vu).

> Disclosure: published and maintained by the author of `deja-vu`. Upstream datasets carry their own licenses (`LongMemEval`: MIT; `LoCoMo`: `CC-BY-NC-4.0`). To respect `LoCoMo`'s license, this dataset redistributes aggregate retrieval metrics and dataset digests only—no raw dialogue text or prompts are included.

## Configs & Splits

| Config | Rows | Contents | Documentation |
| :--- | ---: | :--- | :--- |
| `longmemeval_slices` | 14 | Per-category and total retrieval metrics (`hit@1/5/10/20`, `MRR`, latency, `sha256`) on cleaned (`470q`) and full (`500q`) LongMemEval-S sets (`deja-vu` `0.23.2`) | [Benchmarks guide](https://vshulcz.github.io/deja-vu/guide/benchmarks.html) |
| `locomo_retrieval_metrics` | 6 | Aggregate retrieval-only metrics over 1,982 LoCoMo questions (`multi-hop`, `temporal`, `open-domain`, `single-hop`, `adversarial*`, `TOTAL`) | [Benchmarks guide](https://vshulcz.github.io/deja-vu/guide/benchmarks.html) |
| `day_zero_tools` | 7 | Head-to-head cold-start build time, query latency, `hit@1/5`, `found@50`, and standing MCP `tools/list` schema overhead over 19,195 on-disk sessions | [Day-zero evaluation](https://vshulcz.github.io/deja-vu/guide/day-zero.html) |
| `task_cost_and_compaction` | 10 | End-to-end repository task token/call ablations (`11` runs per arm), 43-compaction retention measurements, and context-readiness baselines | [Day-zero evaluation](https://vshulcz.github.io/deja-vu/guide/day-zero.html) |

## Reproducing & Submitting External Rankings

Clone the benchmark repository and score any external ranking JSON with Python 3 standard library only:

```bash
git clone https://github.com/vshulcz/coding-agent-memory-benchmark
cd coding-agent-memory-benchmark
python3 scripts/verify_artifacts.py
python3 scripts/score_longmemeval.py --data /path/to/longmemeval_s_cleaned.json --score submission.json --skip-abs
```
"""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic JSONL rows and assert no raw dialogue keys appear."""
    for idx, row in enumerate(rows):
        bad = sorted(FORBIDDEN_TEXT_KEYS & {k.lower() for k in row})
        if bad:
            raise ValueError(f"{path.name} row {idx} contains forbidden text fields: {bad}")
    lines = [json.dumps(row, sort_keys=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_longmemeval_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Flatten cleaned and full LongMemEval-S results into uniform JSONL rows."""
    rows: list[dict[str, Any]] = []
    specs = (
        ("cleaned_470", "longmemeval-s-cleaned.json"),
        ("full_500", "longmemeval-s-full.json"),
    )
    for variant, filename in specs:
        data = json.loads((results_dir / filename).read_text(encoding="utf-8"))
        ev = data.get("extra", {}).get("evidence_recall", {})
        for entry in [*data["rows"], data["total"]]:
            is_total = entry["name"] == "TOTAL"
            rows.append(
                {
                    "benchmark": data["benchmark"],
                    "split_variant": variant,
                    "skip_abs": bool(data["flags"]["skip_abs"]),
                    "slice": entry["name"],
                    "is_total": is_total,
                    "questions": entry["n"],
                    "hit_at_1": round(entry["hit@1"], 4),
                    "hit_at_5": round(entry["hit@5"], 4),
                    "hit_at_10": round(entry["hit@10"], 4),
                    "hit_at_20": round(entry["hit@20"], 4),
                    "mrr": round(entry["mrr"], 6),
                    "evidence_recall_at_1": round(ev["@1"], 4) if is_total and ev else None,
                    "evidence_recall_at_5": round(ev["@5"], 4) if is_total and ev else None,
                    "evidence_recall_at_10": round(ev["@10"], 4) if is_total and ev else None,
                    "evidence_recall_at_20": round(ev["@20"], 4) if is_total and ev else None,
                    "median_search_ms": data["median_search_ms"],
                    "wall_seconds": round(data["wall_seconds"], 3),
                    "run_at": data["run_at"],
                    "system": "deja-vu",
                    "version": "0.23.2",
                    "harness": data["harness"],
                    "dataset_name": data["dataset"]["name"],
                    "dataset_bytes": data["dataset"]["bytes"],
                    "dataset_sha256": data["dataset"]["sha256"],
                }
            )
    return rows


def build_locomo_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Flatten LoCoMo aggregate retrieval metrics (zero dialogue text)."""
    data = json.loads((results_dir / "locomo.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in [*data["rows"], data["total"]]:
        rows.append(
            {
                "benchmark": data["benchmark"],
                "slice": entry["name"],
                "is_total": entry["name"] == "TOTAL",
                "questions": entry["n"],
                "hit_at_1": round(entry["hit@1"], 4),
                "hit_at_5": round(entry["hit@5"], 4),
                "hit_at_10": round(entry["hit@10"], 4),
                "hit_at_20": round(entry["hit@20"], 4),
                "mrr": round(entry["mrr"], 6),
                "median_search_ms": data["median_search_ms"],
                "wall_seconds": round(data["wall_seconds"], 3),
                "run_at": data["run_at"],
                "system": "deja-vu",
                "version": "0.23.2",
                "harness": data["harness"],
                "dataset_name": data["dataset"]["name"],
                "dataset_bytes": data["dataset"]["bytes"],
                "dataset_sha256": data["dataset"]["sha256"],
            }
        )
    return rows


def build_day_zero_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Flatten the 19,195-session Day-Zero tool comparison matrix."""
    data = json.loads((results_dir / "day-zero-19195-sessions.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for tool in data["tools"]:
        rows.append(
            {
                "benchmark": data["benchmark"],
                "sessions_on_disk": data["sessions_on_disk"],
                "questions": data["questions"],
                "corpus_source": data["corpus_source"],
                "tool": tool["name"],
                "version": tool["version"],
                "reads_existing_history": tool["reads_existing_history"],
                "agent_formats_supported": tool["agent_formats_supported"],
                "install_size": tool["install_size"],
                "background_process": tool["background_process"],
                "needs_model_or_key": tool["needs_model_or_key"],
                "mcp_schema_tokens_o200k": tool["mcp_schema_tokens_o200k"],
                "mcp_tool_count": tool["mcp_tool_count"],
                "build_seconds": tool["build_seconds"],
                "first_answer_ms": tool["first_answer_ms"],
                "p50_search_ms": tool["p50_search_ms"],
                "hit_at_1": tool["hit_at_1"],
                "hit_at_5": tool["hit_at_5"],
                "found_at_50": tool["found_at_50"],
                "notes": tool["notes"],
            }
        )
    return rows


def build_task_cost_and_compaction_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Flatten task-cost ablation arms, compaction retention, and context readiness."""
    data = json.loads((results_dir / "day-zero-19195-sessions.json").read_text(encoding="utf-8"))
    tc = data["task_cost_ablation"]
    rows: list[dict[str, Any]] = []
    for run_name, arms in (
        ("three_arm_run", tc["three_arm_run"]),
        ("alternating_arm_run", tc["alternating_arm_run"]),
    ):
        deja_ref = arms.get("deja_ref")
        run_date = arms.get("date")
        for arm_name, stats in arms.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                {
                    "experiment": "task_cost_ablation",
                    "run": run_name,
                    "arm": arm_name,
                    "runs_per_arm": tc["runs_per_arm"],
                    "repository_go_files": tc["repository_go_files"],
                    "deja_ref": deja_ref,
                    "date": run_date,
                    "tool_calls": stats["tool_calls"],
                    "median_tokens": stats["median_tokens"],
                    "mean_tokens": stats["mean_tokens"],
                    "cache_read_tokens": stats["cache_read_tokens"],
                    "wall_seconds": stats["wall_seconds"],
                    "value": None,
                    "unit": "tokens",
                }
            )
    cr = data["compaction_retention"]
    rows.append(
        {
            "experiment": "compaction_retention",
            "run": "real_compactions",
            "arm": "architectural_decisions",
            "runs_per_arm": cr["compactions_measured"],
            "repository_go_files": None,
            "deja_ref": None,
            "date": None,
            "tool_calls": None,
            "median_tokens": None,
            "mean_tokens": None,
            "cache_read_tokens": None,
            "wall_seconds": None,
            "value": cr["architectural_decisions_retained_pct"],
            "unit": "percent_retained",
        }
    )
    rows.append(
        {
            "experiment": "compaction_retention",
            "run": "real_compactions",
            "arm": "exact_commands",
            "runs_per_arm": cr["compactions_measured"],
            "repository_go_files": None,
            "deja_ref": None,
            "date": None,
            "tool_calls": None,
            "median_tokens": None,
            "mean_tokens": None,
            "cache_read_tokens": None,
            "wall_seconds": None,
            "value": cr["exact_commands_retained_pct"],
            "unit": "percent_retained",
        }
    )
    ctx = data["context_readiness"]
    for arm_name, key in (
        ("deja_recall", "deja_median_tokens"),
        ("full_session_replay", "full_session_replay_median_tokens"),
        ("naive_raw_log_grep", "naive_raw_log_grep_median_tokens"),
    ):
        rows.append(
            {
                "experiment": "context_readiness",
                "run": "seeded_multi_session_chains",
                "arm": arm_name,
                "runs_per_arm": None,
                "repository_go_files": None,
                "deja_ref": None,
                "date": None,
                "tool_calls": None,
                "median_tokens": ctx[key],
                "mean_tokens": None,
                "cache_read_tokens": None,
                "wall_seconds": None,
                "value": ctx["target_verbatim_fact_coverage"],
                "unit": "verbatim_fact_coverage",
            }
        )
    return rows


def build_dataset(results_dir: Path, out_dir: Path) -> dict[str, int]:
    """Generate hf_dataset/README.md and all JSONL data splits under out_dir."""
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    card_text = DATASET_CARD.replace("day_zero.html", "day-zero.html")
    (out_dir / "README.md").write_text(card_text, encoding="utf-8")

    splits: dict[str, list[dict[str, Any]]] = {
        "longmemeval_slices.jsonl": build_longmemeval_rows(results_dir),
        "locomo_retrieval_metrics.jsonl": build_locomo_rows(results_dir),
        "day_zero_tools.jsonl": build_day_zero_rows(results_dir),
        "task_cost_and_compaction.jsonl": build_task_cost_and_compaction_rows(results_dir),
    }
    counts: dict[str, int] = {}
    for filename, rows in splits.items():
        write_jsonl(data_dir / filename, rows)
        counts[filename] = len(rows)
    return counts


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build Hugging Face dataset bundle from results/.")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--out-dir", type=Path, default=root / "hf_dataset")
    args = parser.parse_args(argv)
    counts = build_dataset(args.results_dir, args.out_dir)
    for name, count in counts.items():
        print(f"wrote hf_dataset/data/{name} ({count} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

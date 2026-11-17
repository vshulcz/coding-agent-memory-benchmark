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

Machine-readable summary splits generated from the verified JSON artifacts in [`vshulcz/coding-agent-memory-benchmark`](https://github.com/vshulcz/coding-agent-memory-benchmark) for [`vshulcz/deja-vu`](https://github.com/vshulcz/deja-vu).

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

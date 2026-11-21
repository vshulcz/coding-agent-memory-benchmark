# Coding Agent Memory Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24416712.svg)](https://doi.org/10.5281/zenodo.24416712)

Reproducible evaluation artifacts, day-zero cold-start comparisons, and zero-dependency scoring scripts for local coding-agent memory and session search tools.

Every committed JSON result in [`results/`](results/) records the dataset filename, byte count, SHA-256 digest, flags, and exact per-slice metrics produced by the benchmark harnesses in [`vshulcz/deja-vu`](https://github.com/vshulcz/deja-vu).

> [!NOTE]
> Disclosure: these runs were executed and published by the maintainer of `deja-vu`. All scoring scripts accept external submissions under [`SUBMISSION.md`](SUBMISSION.md) so any memory tool can be evaluated with the same arithmetic.

## 1. Day-Zero Cold Start (19,195 Sessions on Disk)

Before any memory tool is installed, `19,195` sessions from `longmemeval_s_cleaned.json` are written to disk in native agent transcript layouts (`~/.claude/projects/<project>/<id>.jsonl` and `~/.codex/sessions/<date>/rollout-<id>.jsonl`). Each tool is then installed and queried with `100` questions grounded in that existing history, plus its `tools/list` schema overhead counted with `tiktoken` (`o200k_base`).

| Tool | Version | Install Size | Background Process | MCP Schema Overhead | Build Time | First Query | Search p50 | hit@1 / 100 | hit@5 / 100 | found@50 / 100 |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **deja-vu** | 0.23.2 / 0.21.1 | 18.6 MB binary | none | 477 tok (1 tool) | 13.6 s | 26 ms | 97 ms | 19 | 36 | 65 |
| **ctx** | 1.4.12 / 2.2.8 | 43 MB binary | daemon | 1,897 tok (6 tools) | 72.0 s | 1.04 s | 1.03 s | 7 | 14 | 34 |
| **CASS** | 0.8.0 / main | 58.5 MB binary | none | 0 tok (CLI only) | 56 min | 0.38 s | 0.37 s | 5 | 8 | 16 |
| **agentsview** | 0.44.0 / 0.43.0 | 115 MB binary | daemon | 8,283 tok (7 tools) | 3 min 56 s | 1.44 s | 113 ms | 7 | 13 | 19 |
| **agentmemory** | 0.9.29 | 689 MB npm + 28 MB | worker + engine | 745 tok (7 tools) | 95.0 s | 185 ms | 145 ms | 14 | 34 | 65 |
| **MemPalace** | 3.9.0 | 312 MB + 168 MB model | Chroma | 8,183 tok (45 tools) | ~3 h | 4.50 s | 2.60 s | 14 | 20 | 46 |
| **claude-mem** | 13.25.3 | npm + plugin | worker | 2,307 tok (15 tools) | — | — | — | 0 | 0 | 0 |

Notes on methodology (see [`results/day-zero-19195-sessions.json`](results/day-zero-19195-sessions.json) and [day-zero methodology](https://vshulcz.github.io/deja-vu/guide/day-zero.html)):
- **Empty-directory control:** pointing `deja` at an empty directory scores `0 / 100`.
- **Corpus trade-off:** LongMemEval dialogues are personal-assistant conversations (not source code), scaled roughly 400x larger than the standard ~50-session per-question haystack; absolute `hit@1` reflects that corpus scale and the absence of embeddings on lexical baselines.
- **Stop-word concessions:** `CASS` (strict AND) and `agentsview --fts` returned zero hits on verbatim questions, so both were evaluated with stop words removed.
- **Multi-directory `ctx` retest:** default `ctx setup` skipped indexing when all 19,195 sessions lived inside a single project folder; spread across 200 project folders on `ctx 2.2.8`, setup completed in `72.7 s` (`hit@1: 7`, `hit@5: 18`, `found@50: 37`).
- **Forward-only capture:** `claude-mem` captures session-end summaries via hooks with an LLM provider, so prior history already on disk is intentionally outside its model.

## 2. End-to-End Agent Task Cost (`scripts/taskcost`)

Same repository (`329` Go files), same prompt requiring discovery of a non-obvious build tag and environment variable recorded in two prior local sessions (`11` runs per arm).

### Three-arm run
| Arm | Mean Tool Calls | Median Tokens | Mean Tokens | Cache Read Tokens | Wall Time |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Nothing wired | 17.9 | 126,222 | 119,140 | 87,645 | 67 s |
| `agentmemory` 0.9.29 | 15.9 | 104,974 | 124,960 | 90,950 | 70 s |
| `deja-vu` | 9.8 | 53,558 | 59,167 | 35,700 | 45 s |

### Alternating-arm run (`deja` main `54f6f6ab`, arms interleaved 1:1)
| Arm | Mean Tool Calls | Median Tokens | Mean Tokens | Cache Read Tokens | Wall Time |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Nothing wired | 17.8 | 103,443 | 107,624 | 78,476 | 62 s |
| `deja-vu` | 7.1 | 52,815 | 48,590 | 32,628 | 43 s |

## 3. LongMemEval-S Retrieval (`results/longmemeval-s-cleaned.json` & `longmemeval-s-full.json`)

Standard ~50-session-per-question haystack evaluation run with `deja` 0.23.2 (zero embeddings, zero LLM query rewriting), indexing transcript files through the production ingestion path and querying verbatim question text.

### Cleaned set (`--skip-abs`, `n=470`, median `15.488 ms`)
| Slice | n | hit@1 | hit@5 | hit@10 | hit@20 | MRR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `knowledge-update` | 72 | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| `multi-session` | 121 | 88.4% | 98.3% | 98.3% | 99.2% | 0.927 |
| `single-session-assistant` | 56 | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| `single-session-preference` | 30 | 43.3% | 90.0% | 93.3% | 100.0% | 0.606 |
| `single-session-user` | 64 | 95.3% | 96.9% | 98.4% | 98.4% | 0.963 |
| `temporal-reasoning` | 127 | 85.8% | 96.1% | 99.2% | 100.0% | 0.907 |
| **TOTAL** | **470** | **88.9%** | **97.4%** | **98.7%** | **99.6%** | **0.926** |
| Official `evidence_recall` | — | **57.2%** | **91.5%** | **95.3%** | **97.7%** | — |

### Full set (abstention included, `n=500`, median `14.488 ms`)
| Slice | n | hit@1 | hit@5 | hit@10 | hit@20 | MRR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `knowledge-update` | 78 | 98.7% | 100.0% | 100.0% | 100.0% | 0.994 |
| `multi-session` | 133 | 88.0% | 97.7% | 98.5% | 99.2% | 0.923 |
| `single-session-assistant` | 56 | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| `single-session-preference` | 30 | 43.3% | 90.0% | 93.3% | 100.0% | 0.606 |
| `single-session-user` | 70 | 92.9% | 95.7% | 97.1% | 97.1% | 0.945 |
| `temporal-reasoning` | 133 | 85.0% | 96.2% | 99.2% | 100.0% | 0.901 |
| **TOTAL** | **500** | **88.2%** | **97.2%** | **98.6%** | **99.4%** | **0.921** |
| Official `evidence_recall` | — | **56.4%** | **91.0%** | **94.9%** | **97.2%** | — |

## 4. LoCoMo Retrieval-Only (`results/locomo.json`, `n=1,982`, median `3.481 ms`)

Session-level retrieval over `locomo10.json` without an answering LLM (most LoCoMo papers report LLM end-to-end QA accuracy instead). Category 5 (`adversarial*`) is unanswerable in the upstream benchmark and reported in its own row. No raw LoCoMo dialogue text is redistributed in this repository (`locomo10.json` carries upstream `CC-BY-NC-4.0` terms).

| Slice | n | hit@1 | hit@5 | hit@10 | hit@20 | MRR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `multi-hop` | 282 | 50.7% | 86.5% | 93.3% | 95.7% | 0.652 |
| `temporal` | 321 | 69.8% | 89.1% | 95.3% | 98.8% | 0.784 |
| `open-domain` | 92 | 39.1% | 70.7% | 80.4% | 88.0% | 0.534 |
| `single-hop` | 841 | 79.5% | 94.4% | 97.6% | 99.2% | 0.859 |
| `adversarial*` | 446 | 79.1% | 94.6% | 97.1% | 97.8% | 0.859 |
| **TOTAL** | **1,982** | **71.9%** | **91.4%** | **95.7%** | **97.8%** | **0.802** |

## 5. Compaction Survival & Context Readiness

* **Compaction retention (`n=43` real compactions):** comparing post-compaction summaries against the original transcripts retained **77%** of architectural decisions and **0.2%** of exact executed commands.
* **Context readiness (multi-session task chains):** `deja-vu` reached `1.00` verbatim fact coverage at a median of **1,096 tokens**, versus **80,547 tokens** for full-session replay and **273,238 tokens** for naive raw-log grep.

## 6. Reproducing & Verifying Locally

All verification and scoring scripts use Python 3 standard library only (`tiktoken` is optional for token counting in `measure_mcp_schema_tokens.py`):

```bash
# 1. Verify committed artifact integrity and weighted totals
python3 scripts/verify_artifacts.py

# 2. Run the unit test suite
python3 -m unittest discover -s tests -v

# 3. Score any external LongMemEval-S ranking JSON (flat or wrapped)
python3 scripts/score_longmemeval.py \
  --data /path/to/longmemeval_s_cleaned.json \
  --score submission.json \
  --skip-abs \
  --out scored.json

# 4. Measure standing MCP tools/list schema size for any stdio server
python3 scripts/measure_mcp_schema_tokens.py -- deja mcp
# or without tiktoken installed:
python3 scripts/measure_mcp_schema_tokens.py --bytes-only -- deja mcp

# 5. Rebuild normalized Hugging Face dataset splits under hf_dataset/
python3 scripts/build_hf_dataset.py
```

## Dataset Bundle & Citation

- **Normalized JSONL splits (`hf_dataset/`):** Built deterministically from `results/*.json` with [`scripts/build_hf_dataset.py`](scripts/build_hf_dataset.py) and publishable with `HF_HUB_DISABLE_XET=1` via [`scripts/publish_hf_dataset.py`](scripts/publish_hf_dataset.py). Excludes all raw `LoCoMo` dialogue text.
- **Citation metadata:** [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json) record the exact benchmark snapshot (`0.23.2` / `54f6f6ab` / `9906e203`) and link back to [`vshulcz/deja-vu`](https://github.com/vshulcz/deja-vu).

## License

Code, verification scripts, and result summaries in this repository are MIT-licensed ([`LICENSE`](LICENSE)). Upstream datasets follow their own licenses (`LongMemEval`: MIT; `LoCoMo`: `CC-BY-NC-4.0`, referenced by filename and SHA-256 only).

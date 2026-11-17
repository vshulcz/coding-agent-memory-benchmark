# Submitting External Rankings

Any retrieval system can be scored on **LongMemEval-S** with the same arithmetic used to generate `results/longmemeval-s-cleaned.json` and `results/longmemeval-s-full.json`. No Go toolchain or index build is required.

## 1. Submission File Format

Accepts either a flat mapping or a wrapped `answers` object:

```json
{
  "q_001": ["session_a", "session_b", "session_c"],
  "q_002": ["session_d"]
}
```

or:

```json
{
  "answers": {
    "q_001": ["session_a", "session_b", "session_c"]
  }
}
```

Each value is an ordered list of retrieved session IDs, ranked best first.

## 2. Scoring Rules

* **`hit@k`** (`k in {1, 5, 10, 20}`): `1` when at least one session in `answer_session_ids` appears in `ranked[:k]`, otherwise `0`.
* **`evidence-recall@k`**: `|ranked[:k] ∩ answer_session_ids| / |answer_session_ids|`, averaged across all scored questions (`0` when `answer_session_ids` is empty, such as abstention rows).
* **`MRR`**: `1 / rank` of the first matching session ID (`1`-based), or `0` if none matches.
* **Missing questions**: Any question in the dataset omitted from the submission JSON is counted as a miss (`rank = 0`), never dropped from the denominator.
* **Cleaned vs. Full set**: Pass `--skip-abs` to drop `_abs` abstention rows (`470` questions instead of `500`), matching published cleaned-set runs. Always report the dataset SHA-256 and whether `_abs` rows were excluded.

## 3. Running the Scorer

Python reference scorer (stdlib only):

```bash
python3 scripts/score_longmemeval.py \
  --data /path/to/longmemeval_s_cleaned.json \
  --score your_submission.json \
  --skip-abs \
  --out scored_result.json
```

Equivalent Go harness in [`vshulcz/deja-vu`](https://github.com/vshulcz/deja-vu):

```bash
go run ./scripts/longmemeval \
  -skip-abs \
  -data /path/to/longmemeval_s_cleaned.json \
  -score your_submission.json
```

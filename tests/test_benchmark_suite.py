#!/usr/bin/env python3
"""Unit tests for artifact integrity, LongMemEval submission scoring, and MCP schema probing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_hf_dataset  # noqa: E402
import create_wikidata_item  # noqa: E402
import publish_hf_dataset  # noqa: E402
import score_longmemeval  # noqa: E402
import verify_artifacts  # noqa: E402


MOCK_MCP_SERVER = """\
import json, sys

for line in sys.stdin:
    text = line.strip()
    if not text:
        continue
    msg = json.loads(text)
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        resp = {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2024-11-05", "capabilities": {}}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "tools/list":
        resp = {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "tools": [
                    {"name": "deja", "description": "Search session memory", "inputSchema": {"type": "object"}},
                    {"name": "ping", "description": "Health check", "inputSchema": {"type": "object"}},
                ]
            },
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
"""


class BenchmarkSuiteTests(unittest.TestCase):
    def test_committed_artifacts_verify(self) -> None:
        retrieval, day_zero = verify_artifacts.verify_directory(RESULTS_DIR)
        by_name = {item["benchmark"] + f"_{item['questions']}": item for item in retrieval}
        self.assertEqual(by_name["LongMemEval-S_470"]["total"]["n"], 470)
        self.assertEqual(by_name["LongMemEval-S_500"]["total"]["n"], 500)
        self.assertEqual(by_name["LoCoMo_1982"]["total"]["n"], 1982)
        self.assertEqual(day_zero["sessions_on_disk"], 19195)
        self.assertEqual(day_zero["questions"], 100)
        self.assertEqual(len(day_zero["tools"]), 7)

    def test_verify_detects_corrupted_total(self) -> None:
        src = json.loads((RESULTS_DIR / "longmemeval-s-cleaned.json").read_text(encoding="utf-8"))
        src["total"]["hit@1"] += 0.5
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text(json.dumps(src), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_artifacts.verify_retrieval_artifact(bad_path)

    def test_score_longmemeval_skip_abs_and_missing(self) -> None:
        questions = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "answer_session_ids": ["s1", "s2"],
            },
            {
                "question_id": "q2",
                "question_type": "temporal-reasoning",
                "answer_session_ids": ["s3"],
            },
            {
                "question_id": "q3_abs",
                "question_type": "single-session-user",
                "answer_session_ids": [],
            },
        ]
        flat_sub = {
            "q1": ["s1", "s9", "s2"],  # hit@1, evidence_recall@1=0.5, evidence_recall@5=1.0
            "q2": ["s0", "s3"],        # rank 2 -> hit@5, evidence_recall@1=0.0, evidence_recall@5=1.0
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "lme.json"
            sub_flat = tmp / "sub_flat.json"
            sub_wrapped = tmp / "sub_wrapped.json"
            out_path = tmp / "out.json"

            raw_bytes = json.dumps(questions).encode("utf-8")
            data_path.write_bytes(raw_bytes)
            sub_flat.write_text(json.dumps(flat_sub), encoding="utf-8")
            sub_wrapped.write_text(json.dumps({"answers": flat_sub}), encoding="utf-8")

            rc = score_longmemeval.main(
                ["--data", str(data_path), "--score", str(sub_flat), "--skip-abs", "--out", str(out_path)]
            )
            self.assertEqual(rc, 0)
            scored = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(scored["questions"], 2)
            self.assertAlmostEqual(scored["total"]["hit@1"], 50.0)
            self.assertAlmostEqual(scored["total"]["hit@5"], 100.0)
            self.assertAlmostEqual(scored["total"]["mrr"], (1.0 + 0.5) / 2.0)
            self.assertEqual(scored["dataset"]["sha256"], hashlib.sha256(raw_bytes).hexdigest())
            self.assertAlmostEqual(scored["extra"]["evidence_recall"]["@1"], 25.0)
            self.assertAlmostEqual(scored["extra"]["evidence_recall"]["@5"], 100.0)

            full_res, missing = score_longmemeval.score_submission(
                questions,
                score_longmemeval.load_submission(sub_wrapped),
                scored["dataset"],
                "wrapped",
            )
            self.assertEqual(full_res["questions"], 3)
            self.assertEqual(missing, ["q3_abs"])
            self.assertAlmostEqual(full_res["total"]["hit@1"], 100.0 / 3.0)
            self.assertAlmostEqual(full_res["total"]["hit@5"], 200.0 / 3.0)

    def test_measure_mcp_schema_tokens_mock_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            server = Path(tmpdir) / "mock_mcp.py"
            server.write_text(MOCK_MCP_SERVER, encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "measure_mcp_schema_tokens.py"),
                    "--bytes-only",
                    "--json",
                    "--",
                    sys.executable,
                    str(server),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["tools"], 2)
            self.assertGreater(payload["bytes"], 50)
            self.assertEqual(payload["encoding"], "o200k_base")

    def test_citation_and_zenodo_metadata_match(self) -> None:
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertEqual(zenodo["upload_type"], "dataset")
        self.assertEqual(zenodo["license"], "MIT")
        self.assertEqual(zenodo["version"], "1.0.0")
        self.assertIn("version: \"1.0.0\"", cff)
        self.assertIn("license: MIT", cff)
        self.assertIn("https://github.com/vshulcz/deja-vu", cff)
        related = {item["identifier"]: item["relation"] for item in zenodo["related_identifiers"]}
        self.assertEqual(related.get("https://github.com/vshulcz/deja-vu"), "isSupplementTo")

    def test_build_hf_dataset_contains_no_locomo_dialogue_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "hf_dataset"
            counts = build_hf_dataset.build_dataset(RESULTS_DIR, out_dir)
            self.assertEqual(counts["longmemeval_slices.jsonl"], 14)
            self.assertEqual(counts["locomo_retrieval_metrics.jsonl"], 6)
            self.assertEqual(counts["day_zero_tools.jsonl"], 7)
            self.assertEqual(counts["task_cost_and_compaction.jsonl"], 10)
            validated = publish_hf_dataset.validate_bundle(out_dir)
            self.assertEqual(sum(validated.values()), 37)
            locomo_lines = (out_dir / "data" / "locomo_retrieval_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            for line in locomo_lines:
                row = json.loads(line)
                self.assertEqual(row["benchmark"], "LoCoMo")
                self.assertFalse(build_hf_dataset.FORBIDDEN_TEXT_KEYS & {k.lower() for k in row})

        self.assertEqual(publish_hf_dataset.main(["--dry-run"]), 0)

    def test_wikidata_payload_and_quickstatements(self) -> None:
        qs_path = ROOT / "wikidata" / "quickstatements_deja_vu.txt"
        qs_lines = [ln for ln in qs_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(qs_lines[0], "CREATE")
        self.assertTrue(any("\tP31\tQ341\t" in ln for ln in qs_lines))
        self.assertTrue(any("\tP275\tQ334661\t" in ln for ln in qs_lines))
        self.assertTrue(any("\tP277\tQ37227\t" in ln for ln in qs_lines))
        self.assertNotIn("P5568", "\n".join(qs_lines))

        payload = create_wikidata_item.build_entity_payload(zenodo_doi="10.5281/zenodo.999999")
        self.assertEqual(payload["claims"]["P356"][0]["mainsnak"]["datavalue"]["value"], "10.5281/ZENODO.999999")
        self.assertNotIn("P5568", payload["claims"])
        self.assertEqual(create_wikidata_item.main(["--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()

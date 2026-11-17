#!/usr/bin/env python3
"""Validate and publish hf_dataset/ to Hugging Face Datasets with XET disabled."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EXPECTED_SPLITS = (
    "data/longmemeval_slices.jsonl",
    "data/locomo_retrieval_metrics.jsonl",
    "data/day_zero_tools.jsonl",
    "data/task_cost_and_compaction.jsonl",
)


def validate_bundle(dataset_dir: Path) -> dict[str, int]:
    """Verify hf_dataset/README.md and all JSONL split files before upload."""
    readme = dataset_dir / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(f"missing dataset card: {readme}")
    card = readme.read_text(encoding="utf-8")
    if not card.startswith("---\n"):
        raise ValueError("hf_dataset/README.md must start with YAML frontmatter")

    counts: dict[str, int] = {}
    for rel in EXPECTED_SPLITS:
        path = dataset_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing split file: {path}")
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"empty split file: {path}")
        for idx, line in enumerate(lines):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{idx + 1} is not a JSON object")
        counts[rel] = len(lines)
    return counts


def resolve_token() -> str:
    """Return HF_TOKEN / HUGGING_FACE_HUB_TOKEN or cached Hugging Face CLI token."""
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        val = token_file.read_text(encoding="utf-8").strip()
        if val:
            return val
    return ""


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Publish hf_dataset/ to Hugging Face Datasets.")
    parser.add_argument("--repo-id", default="vshulcz/coding-agent-memory-benchmark")
    parser.add_argument("--dataset-dir", type=Path, default=root / "hf_dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate bundle and exit without uploading")
    args = parser.parse_args(argv)

    counts = validate_bundle(args.dataset_dir)
    if args.dry_run:
        for rel, count in counts.items():
            print(f"verified {rel}: {count} rows (repo={args.repo_id}, xet_disabled={os.environ.get('HF_HUB_DISABLE_XET')})")
        return 0

    token = resolve_token()
    if not token:
        print("missing HF_TOKEN (run `hf auth login` or export HF_TOKEN)", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(args.dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Publish coding-agent-memory-benchmark summary dataset",
    )
    print(f"published https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

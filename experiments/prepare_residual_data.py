#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub", "numpy", "pyarrow", "tiktoken", "torch"]
# ///
"""Prepare pinned FineWeb-Edu document-split GPT-2 caches on a CPU host.

Example: python experiments/prepare_residual_data.py --output-dir data/residual
Repeat --source-file sample/10BT/<filename>.parquet to select explicit shards.
The runner receives train.pt/val.pt via --train-cache/--val-cache;
non-overlapping sequence packing (suggested context 512) belongs to the runner.
"""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

REPO_ID = "HuggingFaceFW/fineweb-edu"
SOURCE_PREFIX = "sample/10BT/"
EOS_ID = 50256
VOCAB_SIZE = 50257
SPLITS = ("train", "val")
SPLIT_RULE = "big_endian_uint64(SHA256(exact_UTF8_document)[:8]) % 1000 < 10 => val; else train"


def document_split(text: str) -> str:
    """Identical document content always has the same split, regardless of order."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:8], "big") % 1000 < 10 else "train"


def encode_documents(documents, tokenizer, limits, *, buffers=None):
    """Fill exact token budgets; defaults to compact arrays for offline use.

    Buffers may instead be preallocated int32 memmaps. A document is encoded
    with encode_ordinary, followed by EOS, then truncated to its split's budget.
    A filled split skips later documents without tokenizing or redirecting them.
    No next document is consumed after both budgets are reached. Exhaustion is
    an error rather than a successful partial dataset.
    """
    if set(limits) != set(SPLITS) or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in limits.values()
    ):
        raise ValueError("limits must contain non-negative integer train and val budgets")
    if array("i").itemsize != 4:
        raise RuntimeError("This host does not provide 32-bit array('i') storage")
    if buffers is None:
        buffers = {split: array("i") for split in SPLITS}
    stats = {
        "raw_documents_seen": 0,
        "raw_documents_by_split": dict.fromkeys(SPLITS, 0),
        "documents_written": dict.fromkeys(SPLITS, 0),
        "documents_skipped_full_split": dict.fromkeys(SPLITS, 0),
        "tokens": dict.fromkeys(SPLITS, 0),
        "eos_tokens_written": dict.fromkeys(SPLITS, 0),
        "truncated_documents": [],
    }
    if all(limits[split] == 0 for split in SPLITS):
        return buffers, stats
    for text in documents:
        if not isinstance(text, str):
            raise ValueError("Every parquet text row must be a string")
        split = document_split(text)
        stats["raw_documents_seen"] += 1
        stats["raw_documents_by_split"][split] += 1
        remaining = limits[split] - stats["tokens"][split]
        if remaining == 0:
            stats["documents_skipped_full_split"][split] += 1
            continue
        ids = tokenizer.encode_ordinary(text)
        ids.append(EOS_ID)
        kept = min(remaining, len(ids))
        start = stats["tokens"][split]
        buffers[split][start : start + kept] = array("i", ids[:kept])
        stats["tokens"][split] += kept
        stats["documents_written"][split] += 1
        if kept == len(ids):
            stats["eos_tokens_written"][split] += 1
        else:
            stats["truncated_documents"].append({
                "split": split,
                "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "encoded_tokens_including_eos": len(ids),
                "written_tokens": kept,
                "eos_kept": False,
            })
        if all(stats["tokens"][s] == limits[s] for s in SPLITS):
            break
    if any(stats["tokens"][s] != limits[s] for s in SPLITS):
        raise RuntimeError(f"Source exhausted: collected {stats['tokens']}, requested {limits}")
    return buffers, stats


def resolve_sources(api, requested=None, *, revision=None):
    """Verify the commit, then use its siblings or list only the sample directory."""
    requested_revision = revision
    if requested_revision is not None:
        if not isinstance(requested_revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", requested_revision
        ):
            raise ValueError("--revision must be a full 40-character commit SHA")
        requested_revision = requested_revision.lower()
    try:
        info = (
            api.dataset_info(REPO_ID, revision=requested_revision)
            if requested_revision is not None else api.dataset_info(REPO_ID)
        )
        revision = info.sha
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(f"Dataset API did not return a commit SHA: {revision!r}")
        revision = revision.lower()
        if requested_revision is not None and revision != requested_revision:
            raise ValueError(f"Revision mismatch: requested {requested_revision}, API returned {revision}")
        siblings = getattr(info, "siblings", None)
        if siblings:
            files = [sibling.rfilename for sibling in siblings]
        else:
            # Full-repository pagination can follow mirror Link headers off-host.
            files = [entry.path for entry in api.list_repo_tree(
                repo_id=REPO_ID, path_in_repo=SOURCE_PREFIX.rstrip("/"),
                recursive=False, revision=revision, repo_type="dataset",
            )]
    except Exception as exc:
        raise RuntimeError(f"Cannot pin/list official dataset {REPO_ID}: {exc}") from exc
    candidates = sorted(f for f in files if f.startswith(SOURCE_PREFIX) and f.endswith(".parquet"))
    if not candidates:
        raise RuntimeError(f"No {SOURCE_PREFIX}*.parquet files at {REPO_ID}@{revision}")
    selected = list(dict.fromkeys(requested)) if requested else candidates
    missing = [f for f in selected if f not in candidates]
    if missing:
        raise ValueError(f"Invalid --source-file paths: {missing}; candidates include {candidates[:8]}")
    return revision, selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_parquet_documents(revision, files, cache_dir, records):
    """Download only consumed shards; retain the raw HF cache for reruns."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    for filename in files:
        try:
            path = Path(hf_hub_download(
                repo_id=REPO_ID, filename=filename, repo_type="dataset",
                revision=revision, cache_dir=str(cache_dir), token=False,
            ))
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise RuntimeError(f"Cannot download/open {REPO_ID}@{revision}:{filename}: {exc}") from exc
        record = {
            "repository_path": filename,
            "revision": revision,
            "cached_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows_available": parquet.metadata.num_rows,
            "rows_read": 0,
        }
        records.append(record)
        for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column(0).to_pylist():
                record["rows_read"] += 1
                yield text


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("token budgets must be non-negative")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=nonnegative_int, default=250_000_000)
    parser.add_argument("--val-tokens", type=nonnegative_int, default=2_000_000)
    parser.add_argument("--source-file", action="append", help="Exact sample/10BT/ parquet path; repeatable")
    parser.add_argument("--revision", help="Official dataset commit SHA to verify and pin")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {split: output_dir / f"{split}.pt" for split in SPLITS}
    manifest_path = output_dir / "manifest.json"
    for path in [*output_paths.values(), manifest_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new --output-dir")
    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)

    # Lazy imports keep offline helpers usable without any third-party packages.
    from huggingface_hub import HfApi
    import numpy as np
    import tiktoken
    import torch

    api = HfApi(token=False)
    revision, files = resolve_sources(api, args.source_file, revision=args.revision)
    tokenizer = tiktoken.get_encoding("gpt2")
    if tokenizer.n_vocab != VOCAB_SIZE or tokenizer.eot_token != EOS_ID:
        raise RuntimeError("Unexpected GPT-2 tokenizer vocabulary/EOS contract")
    limits = {"train": args.train_tokens, "val": args.val_tokens}
    raw_paths = {split: output_dir / f".{split}.tokens.int32.partial" for split in SPLITS}
    buffers = {
        split: np.memmap(raw_paths[split], dtype="<i4", mode="w+", shape=(limits[split],))
        if limits[split] else np.empty(0, dtype="<i4")
        for split in SPLITS
    }
    records = []
    documents = iter_parquet_documents(revision, files, output_dir / "raw_cache", records)
    try:
        _, stats = encode_documents(documents, tokenizer, limits, buffers=buffers)
    finally:
        documents.close()

    outputs = {}
    for split in SPLITS:
        buffer = buffers[split]
        if limits[split]:
            buffer.flush()
            token_sha = sha256_file(raw_paths[split])
        else:
            token_sha = hashlib.sha256(b"").hexdigest()
        partial_path = output_paths[split].with_suffix(".pt.partial")
        torch.save(torch.from_numpy(buffer), partial_path)
        outputs[split] = {
            "path": str(output_paths[split]),
            "tokens": stats["tokens"][split],
            "dtype": "torch.int32",
            "shape": [stats["tokens"][split]],
            "bytes": partial_path.stat().st_size,
            "file_sha256": sha256_file(partial_path),
            "token_sha256": token_sha,
            "token_hash_encoding": "contiguous little-endian signed int32 token IDs",
        }
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": REPO_ID,
        "dataset_config": "sample/10BT",
        "source_revision": revision,
        "requested_source_revision": args.revision,
        "transport_endpoint": api.endpoint,
        "preparation_script": {"path": str(script_path), "sha256": script_sha256},
        "selected_source_files": files,
        "consumed_source_files": records,
        "tokenizer": {"package": "tiktoken", "encoding": "gpt2", "api": "encode_ordinary",
                      "vocab_size": VOCAB_SIZE, "eos_token_id": EOS_ID},
        "split_rule": SPLIT_RULE,
        "duplicate_policy": "identical exact UTF-8 text stays in one split; within-split duplicates retained",
        "target_tokens": limits,
        "actual_tokens": stats["tokens"],
        "document_counts": stats,
        "truncation_policy": "prefix of document tokens plus EOS; final truncated document may omit EOS",
        "train_cache_path": str(output_paths["train"]),
        "val_cache_path": str(output_paths["val"]),
        "outputs": outputs,
        "packing": {"performed": False, "recommended_context_length": 512,
                    "instruction": "Runner must pack non-overlapping sequences within each split"},
        "raw_cache_policy": "retained under output-dir/raw_cache; never deleted by this script",
        "package_versions": {name: version(name) for name in
                             ("huggingface-hub", "pyarrow", "tiktoken", "numpy", "torch")},
        "python_version": sys.version,
    }
    manifest_partial = manifest_path.with_suffix(".json.partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for split in SPLITS:
        output_paths[split].with_suffix(".pt.partial").replace(output_paths[split])
    manifest_partial.replace(manifest_path)
    for path in raw_paths.values():
        path.unlink(missing_ok=True)  # Token scratch only; raw parquet cache is retained.
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()

"""Offline checks for document-level data preparation; no HF packages required."""

from array import array
import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from experiments import prepare_residual_data as preparation
from experiments.prepare_residual_data import (
    EOS_ID,
    REPO_ID,
    document_split,
    encode_documents,
    resolve_sources,
)

TRAIN_TEXT = "train-document-0"  # SHA bucket 58
VAL_TEXT = "val-document-186"  # SHA bucket 3
REVISION = "a" * 40
SOURCE_FILE = "sample/10BT/000_00000.parquet"


class FakeEncoder:
    def __init__(self, mapping=None):
        self.mapping = mapping or {TRAIN_TEXT: [101, 102], VAL_TEXT: [201, 202]}
        self.calls = []

    def encode_ordinary(self, text):
        self.calls.append(text)
        return list(self.mapping[text])


class FakeApi:
    def __init__(self, files=None, revision=REVISION, include_siblings=True):
        self.files = files if files is not None else [SOURCE_FILE]
        self.revision = revision
        self.include_siblings = include_siblings
        self.endpoint = "https://hf-mirror.com"
        self.calls = []

    def dataset_info(self, repo_id, revision=None):
        self.calls.append(("info", repo_id, revision) if revision is not None else ("info", repo_id))
        attributes = {"sha": self.revision}
        if self.include_siblings:
            attributes["siblings"] = [type("Sibling", (), {"rfilename": f})() for f in self.files]
        return type("Info", (), attributes)()

    def list_repo_files(self, **kwargs):
        raise AssertionError("Full-repository pagination is forbidden")

    def list_repo_tree(self, **kwargs):
        self.calls.append(("tree", kwargs))
        return (type("Entry", (), {"path": f})() for f in self.files)


class ResidualDataTests(unittest.TestCase):
    def test_module_import_does_not_require_external_packages(self):
        real_import = builtins.__import__

        def offline_import(name, *args, **kwargs):
            if name.split(".")[0] in {"huggingface_hub", "pyarrow", "tiktoken", "torch", "numpy"}:
                raise ImportError(f"Offline test forbids {name}")
            return real_import(name, *args, **kwargs)

        path = Path(__file__).resolve().parents[1] / "experiments/prepare_residual_data.py"
        spec = importlib.util.spec_from_file_location("offline_residual_data", path)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=offline_import):
            spec.loader.exec_module(module)
            streams, _ = module.encode_documents(
                [TRAIN_TEXT], FakeEncoder(), {"train": 3, "val": 0}
            )
        self.assertEqual(list(streams["train"]), [101, 102, EOS_ID])

    def test_known_split_buckets_and_utf8_content_hash(self):
        self.assertEqual(document_split(TRAIN_TEXT), "train")
        self.assertEqual(document_split(VAL_TEXT), "val")
        text = "正文\nUTF-8 café"
        bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % 1000
        self.assertEqual(document_split(text), "val" if bucket < 10 else "train")

    def test_identical_content_never_crosses_splits_and_eos_is_per_document(self):
        encoder = FakeEncoder()
        streams, stats = encode_documents(
            [TRAIN_TEXT, VAL_TEXT, TRAIN_TEXT, VAL_TEXT],
            encoder,
            {"train": 6, "val": 6},
        )
        self.assertEqual(list(streams["train"]), [101, 102, EOS_ID] * 2)
        self.assertEqual(list(streams["val"]), [201, 202, EOS_ID] * 2)
        self.assertEqual(stats["raw_documents_by_split"], {"train": 2, "val": 2})
        self.assertEqual(stats["eos_tokens_written"], {"train": 2, "val": 2})
        self.assertEqual(stats["truncated_documents"], [])
        self.assertIsInstance(streams["train"], array)
        self.assertEqual(streams["train"].itemsize, 4)

    def test_exact_limits_truncate_final_document_and_record_eos_omission(self):
        streams, stats = encode_documents(
            [VAL_TEXT, TRAIN_TEXT, TRAIN_TEXT],
            FakeEncoder(),
            {"train": 4, "val": 1},
        )
        self.assertEqual(list(streams["train"]), [101, 102, EOS_ID, 101])
        self.assertEqual(list(streams["val"]), [201])
        self.assertEqual(stats["tokens"], {"train": 4, "val": 1})
        self.assertEqual(stats["eos_tokens_written"], {"train": 1, "val": 0})
        self.assertEqual([d["split"] for d in stats["truncated_documents"]], ["val", "train"])
        for record in stats["truncated_documents"]:
            self.assertEqual(record["encoded_tokens_including_eos"], 3)
            self.assertEqual(record["written_tokens"], 1)
            self.assertFalse(record["eos_kept"])
            text = VAL_TEXT if record["split"] == "val" else TRAIN_TEXT
            self.assertEqual(record["document_sha256"], hashlib.sha256(text.encode()).hexdigest())

    def test_full_split_skips_encoding_without_rerouting_or_overconsumption(self):
        encoder = FakeEncoder()
        seen = []

        def documents():
            for text in [TRAIN_TEXT, TRAIN_TEXT, VAL_TEXT]:
                seen.append(text)
                yield text
            self.fail("Document iterator advanced after both targets were satisfied")

        streams, stats = encode_documents(documents(), encoder, {"train": 3, "val": 3})
        self.assertEqual(seen, [TRAIN_TEXT, TRAIN_TEXT, VAL_TEXT])
        self.assertEqual(encoder.calls, [TRAIN_TEXT, VAL_TEXT])
        self.assertEqual(stats["documents_skipped_full_split"], {"train": 1, "val": 0})
        self.assertEqual(list(streams["val"]), [201, 202, EOS_ID])

    def test_reproducible_and_not_dependent_on_iterator_batch_boundaries(self):
        docs = [TRAIN_TEXT, VAL_TEXT, TRAIN_TEXT, VAL_TEXT]
        first, first_stats = encode_documents(docs, FakeEncoder(), {"train": 5, "val": 4})

        def batches():
            for batch in [docs[:1], docs[1:3], docs[3:]]:
                yield from batch

        second, second_stats = encode_documents(batches(), FakeEncoder(), {"train": 5, "val": 4})
        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(first["train"].tobytes(), second["train"].tobytes())

    def test_zero_budgets_do_not_touch_stateful_document_iterator(self):
        def documents():
            self.fail("Zero budgets must not consume documents")
            yield TRAIN_TEXT

        streams, stats = encode_documents(documents(), FakeEncoder(), {"train": 0, "val": 0})
        self.assertEqual(len(streams["train"]), 0)
        self.assertEqual(stats["raw_documents_seen"], 0)

    def test_empty_document_still_has_one_eos(self):
        split = document_split("")
        limits = {"train": 0, "val": 0}
        limits[split] = 1
        streams, _ = encode_documents([""], FakeEncoder({"": []}), limits)
        self.assertEqual(list(streams[split]), [EOS_ID])

    def test_source_exhaustion_and_invalid_budgets_fail(self):
        with self.assertRaisesRegex(RuntimeError, "Source exhausted"):
            encode_documents([TRAIN_TEXT] * 5, FakeEncoder(), {"train": 3, "val": 3})
        for limits in ({"train": -1, "val": 0}, {"train": 1}, {"train": True, "val": 0}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                encode_documents([], FakeEncoder(), limits)
        with self.assertRaisesRegex(ValueError, "must be a string"):
            encode_documents([None], FakeEncoder(), {"train": 1, "val": 1})

    def test_source_resolution_pins_listing_and_sorts_only_official_sample(self):
        second = "sample/10BT/000_00001.parquet"
        api = FakeApi([second, "data/other.parquet", SOURCE_FILE, "sample/10BT/README.md"])
        revision, files = resolve_sources(api)
        self.assertEqual(revision, REVISION)
        self.assertEqual(files, [SOURCE_FILE, second])
        self.assertEqual(api.calls, [("info", REPO_ID)])
        _, selected = resolve_sources(api, [second, SOURCE_FILE, second])
        self.assertEqual(selected, [second, SOURCE_FILE])

    def test_explicit_revision_verifies_same_commit_without_querying_head(self):
        api = FakeApi()
        revision, files = resolve_sources(api, [SOURCE_FILE], revision=REVISION.upper())
        self.assertEqual((revision, files), (REVISION, [SOURCE_FILE]))
        self.assertEqual(api.calls, [("info", REPO_ID, REVISION)])

    def test_pinned_siblings_avoid_all_tree_and_full_repository_pagination(self):
        api = FakeApi()
        with patch.object(api, "list_repo_tree", side_effect=AssertionError("Tree must not be queried")):
            revision, files = resolve_sources(api, revision=REVISION)
        self.assertEqual((revision, files), (REVISION, [SOURCE_FILE]))
        self.assertEqual(api.calls, [("info", REPO_ID, REVISION)])

    def test_missing_siblings_lists_only_nonrecursive_pinned_sample_directory(self):
        api = FakeApi(include_siblings=False)
        revision, files = resolve_sources(api, revision=REVISION)
        self.assertEqual((revision, files), (REVISION, [SOURCE_FILE]))
        self.assertEqual(api.calls, [
            ("info", REPO_ID, REVISION),
            ("tree", {
                "repo_id": REPO_ID, "path_in_repo": "sample/10BT", "recursive": False,
                "revision": REVISION, "repo_type": "dataset",
            }),
        ])

    def test_sample_tree_failure_does_not_fallback_to_whole_repository(self):
        api = FakeApi(include_siblings=False)
        with patch.object(api, "list_repo_tree", side_effect=OSError("sample tree unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Cannot pin/list.*sample tree unavailable"):
                resolve_sources(api, revision=REVISION)
        self.assertEqual(api.calls, [("info", REPO_ID, REVISION)])

    def test_explicit_revision_mismatch_fails_before_listing_files(self):
        api = FakeApi(revision="b" * 40)
        with self.assertRaisesRegex(RuntimeError, "Revision mismatch"):
            resolve_sources(api, revision=REVISION)
        self.assertEqual(api.calls, [("info", REPO_ID, REVISION)])

    def test_explicit_revision_must_be_a_full_commit_sha(self):
        api = FakeApi()
        for revision in ("main", "a" * 39, "z" * 40):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "commit SHA"):
                resolve_sources(api, revision=revision)
        self.assertEqual(api.calls, [])

    def test_source_api_or_selection_failure_has_no_alternative_source_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "commit SHA"):
            resolve_sources(FakeApi(revision="main"))
        with self.assertRaisesRegex(RuntimeError, "No sample/10BT"):
            resolve_sources(FakeApi(files=["data/other.parquet"]))
        with self.assertRaisesRegex(ValueError, "candidates include"):
            resolve_sources(FakeApi(), ["private/other.parquet"])
        api = FakeApi()
        api.dataset_info = lambda _: (_ for _ in ()).throw(OSError("offline"))
        with self.assertRaisesRegex(RuntimeError, "Cannot pin/list.*offline"):
            resolve_sources(api)
        self.assertEqual(api.calls, [])

    def test_existing_cache_is_protected_before_external_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "train.pt"
            cache.write_bytes(b"existing cache")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                preparation.main(["--output-dir", directory])
            self.assertEqual(cache.read_bytes(), b"existing cache")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Optional parquet check needs local pyarrow")
    def test_parquet_batches_pin_downloads_count_consumed_rows_and_retain_raw_file(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        hf = ModuleType("huggingface_hub")
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.parquet"
            pq.write_table(pa.table({"text": [TRAIN_TEXT, VAL_TEXT, TRAIN_TEXT]}), path)

            def download(**kwargs):
                calls.append(kwargs)
                return str(path)

            hf.hf_hub_download = download
            records = []
            cache_dir = Path(directory) / "raw_cache"
            with patch.dict(sys.modules, {"huggingface_hub": hf}):
                docs = preparation.iter_parquet_documents(
                    REVISION, [SOURCE_FILE, "sample/10BT/not_needed.parquet"], cache_dir, records
                )
                _, stats = encode_documents(docs, FakeEncoder(), {"train": 3, "val": 3})
                docs.close()
            self.assertEqual(calls, [{
                "repo_id": REPO_ID, "filename": SOURCE_FILE, "repo_type": "dataset",
                "revision": REVISION, "cache_dir": str(cache_dir), "token": False,
            }])
            self.assertEqual(stats["raw_documents_seen"], 2)
            self.assertEqual(records[0]["rows_read"], 2)
            self.assertEqual(records[0]["rows_available"], 3)
            self.assertEqual(records[0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(path.exists())

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
        "Optional cache serialization check needs locally installed numpy/torch",
    )
    def test_mmap_caches_manifest_hashes_and_failed_run_without_complete_manifest(self):
        import torch

        hf = ModuleType("huggingface_hub")
        hf.HfApi = lambda **kwargs: FakeApi()
        tiktoken = ModuleType("tiktoken")
        encoder = FakeEncoder()
        encoder.n_vocab = 50257
        encoder.eot_token = EOS_ID
        tiktoken.get_encoding = lambda name: encoder

        def documents(revision, files, cache_dir, records):
            records.append({"repository_path": files[0], "revision": revision, "rows_read": 2})
            yield TRAIN_TEXT
            yield VAL_TEXT

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"huggingface_hub": hf, "tiktoken": tiktoken}), patch.object(
                preparation, "iter_parquet_documents", documents
            ), patch.object(preparation, "version", return_value="offline-fixture"):
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = preparation.main([
                        "--output-dir", directory, "--train-tokens", "3", "--val-tokens", "3",
                        "--revision", REVISION,
                    ])
                failed_dir = Path(directory) / "failed"
                with self.assertRaisesRegex(RuntimeError, "Source exhausted"):
                    preparation.main([
                        "--output-dir", str(failed_dir), "--train-tokens", "4", "--val-tokens", "3"
                    ])
                self.assertFalse((failed_dir / "manifest.json").exists())
                self.assertFalse((failed_dir / "train.pt").exists())
                self.assertFalse((failed_dir / "val.pt").exists())
                mismatched_dir = Path(directory) / "mismatched"
                with self.assertRaisesRegex(RuntimeError, "Revision mismatch"):
                    preparation.main([
                        "--output-dir", str(mismatched_dir), "--revision", "b" * 40
                    ])
                self.assertEqual(list(mismatched_dir.iterdir()), [])

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["source_revision"], REVISION)
            self.assertEqual(manifest["requested_source_revision"], REVISION)
            self.assertEqual(manifest["transport_endpoint"], "https://hf-mirror.com")
            script_path = Path(preparation.__file__).resolve()
            self.assertEqual(manifest["preparation_script"], {
                "path": str(script_path), "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest()
            })
            self.assertEqual(manifest["actual_tokens"], {"train": 3, "val": 3})
            self.assertEqual(json.loads((Path(directory) / "manifest.json").read_text()), manifest)
            for split, expected in (("train", [101, 102, EOS_ID]), ("val", [201, 202, EOS_ID])):
                path = Path(manifest[f"{split}_cache_path"])
                tensor = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
                self.assertEqual(tensor.dtype, torch.int32)
                self.assertEqual(tensor.ndim, 1)
                self.assertEqual(tensor.tolist(), expected)
                canonical_bytes = b"".join(token.to_bytes(4, "little", signed=True) for token in expected)
                self.assertEqual(
                    manifest["outputs"][split]["token_sha256"], hashlib.sha256(canonical_bytes).hexdigest()
                )
                self.assertEqual(
                    manifest["outputs"][split]["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )


if __name__ == "__main__":
    unittest.main()

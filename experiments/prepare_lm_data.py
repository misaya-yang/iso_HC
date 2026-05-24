"""Prepare LM tokenizer and token caches before GPU experiments."""

import argparse
from array import array
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm.data import get_tokenizer, save_token_cache


def _iter_hf_text(dataset_name, split, streaming=True, dataset_config=None, text_key=None):
    from datasets import load_dataset

    if dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)
        text_keys = ("text", "story")
    else:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        text_keys = (text_key,) if text_key else ("text", "content", "story")

    for example in dataset:
        text = ""
        for key in text_keys:
            if key in example and example[key]:
                text = example[key]
                break
        if text and len(text) > 10:
            yield text


def save_streaming_token_cache(
    dataset_name,
    tokenizer,
    cache_path,
    split,
    max_samples=None,
    target_tokens=None,
    streaming=True,
    dataset_config=None,
    text_key=None,
):
    """Stream/tokenize a HF dataset without materializing full parquet locally."""
    token_ids = array("i")
    eos = tokenizer.eos_token_id
    seen = 0
    progress = tqdm(desc=f"{dataset_name}:{split}", unit="sample")

    for text in _iter_hf_text(
        dataset_name,
        split,
        streaming=streaming,
        dataset_config=dataset_config,
        text_key=text_key,
    ):
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)
        token_ids.append(eos)
        seen += 1
        progress.update(1)
        progress.set_postfix(tokens=len(token_ids))

        if max_samples is not None and seen >= max_samples:
            break
        if target_tokens is not None and len(token_ids) >= target_tokens:
            break

    progress.close()
    if not token_ids:
        raise RuntimeError(f"No tokens collected for {dataset_name}:{split}")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tensor = torch.tensor(token_ids, dtype=torch.int32)
    torch.save(tensor, cache_path)
    print(f"Saved {len(token_ids)} int32 tokens from {seen} samples to {cache_path}")
    return cache_path


def save_streaming_train_val_from_one_stream(
    dataset_name,
    tokenizer,
    train_cache_path,
    val_cache_path,
    split,
    train_tokens,
    val_tokens,
    dataset_config=None,
    text_key=None,
    max_samples=None,
):
    """Build train/val token caches from one streaming split without overlap."""
    if train_tokens is None or val_tokens is None:
        raise ValueError("--val_after_train requires --target_train_tokens and --target_val_tokens")

    train_ids = array("i")
    val_ids = array("i")
    eos = tokenizer.eos_token_id
    seen = 0
    progress = tqdm(desc=f"{dataset_name}:{split}:train+val", unit="sample")

    for text in _iter_hf_text(
        dataset_name,
        split,
        streaming=True,
        dataset_config=dataset_config,
        text_key=text_key,
    ):
        ids = tokenizer.encode(text, add_special_tokens=False)
        target = train_ids if len(train_ids) < train_tokens else val_ids
        target.extend(ids)
        target.append(eos)
        seen += 1
        progress.update(1)
        progress.set_postfix(train_tokens=len(train_ids), val_tokens=len(val_ids))

        if max_samples is not None and seen >= max_samples:
            break
        if len(train_ids) >= train_tokens and len(val_ids) >= val_tokens:
            break

    progress.close()
    if len(train_ids) < train_tokens or len(val_ids) < val_tokens:
        raise RuntimeError(
            f"Only collected train={len(train_ids)} val={len(val_ids)} tokens "
            f"from {dataset_name}:{split}"
        )

    os.makedirs(os.path.dirname(train_cache_path) or ".", exist_ok=True)
    torch.save(torch.tensor(train_ids, dtype=torch.int32), train_cache_path)
    torch.save(torch.tensor(val_ids, dtype=torch.int32), val_cache_path)
    print(
        f"Saved non-overlapping streaming caches from {seen} samples: "
        f"train={len(train_ids)} -> {train_cache_path}, "
        f"val={len(val_ids)} -> {val_cache_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare LM dataset caches")
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--text_key", default=None)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--output_dir", default="data/lm_cache")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument(
        "--val_after_train",
        action="store_true",
        help="Use the same streaming split and collect validation tokens after train tokens.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_val", type=int, default=None)
    parser.add_argument("--target_train_tokens", type=int, default=None)
    parser.add_argument("--target_val_tokens", type=int, default=None)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no_streaming", dest="streaming", action="store_false")
    parser.add_argument(
        "--hard_exit",
        action="store_true",
        help="Force process exit after files are written; useful for HF streaming cleanup hangs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = get_tokenizer()

    safe_name = args.dataset.replace("/", "__")
    if args.dataset_config:
        safe_name = f"{safe_name}__{args.dataset_config}"
    train_cache = os.path.join(args.output_dir, f"{safe_name}_{args.train_split}_ctx{args.context_length}.pt")
    val_split_label = f"{args.train_split}_heldout" if args.val_after_train else args.val_split
    val_cache = os.path.join(args.output_dir, f"{safe_name}_{val_split_label}_ctx{args.context_length}.pt")

    if args.streaming and args.dataset != "random" and args.val_after_train:
        save_streaming_train_val_from_one_stream(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            train_cache_path=train_cache,
            val_cache_path=val_cache,
            split=args.train_split,
            train_tokens=args.target_train_tokens,
            val_tokens=args.target_val_tokens,
            dataset_config=args.dataset_config,
            text_key=args.text_key,
            max_samples=args.max_samples,
        )
    elif args.streaming and args.dataset != "random":
        save_streaming_token_cache(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            cache_path=train_cache,
            split=args.train_split,
            max_samples=args.max_samples,
            target_tokens=args.target_train_tokens,
            streaming=True,
            dataset_config=args.dataset_config,
            text_key=args.text_key,
        )
        save_streaming_token_cache(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            cache_path=val_cache,
            split=args.val_split,
            max_samples=args.max_samples_val,
            target_tokens=args.target_val_tokens,
            streaming=True,
            dataset_config=args.dataset_config,
            text_key=args.text_key,
        )
    else:
        save_token_cache(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            cache_path=train_cache,
            context_length=args.context_length,
            split=args.train_split,
            max_samples=args.max_samples,
        )
        save_token_cache(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            cache_path=val_cache,
            context_length=args.context_length,
            split=args.val_split,
            max_samples=args.max_samples_val,
        )

    manifest = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "text_key": args.text_key,
        "context_length": args.context_length,
        "train_split": args.train_split,
        "val_split": val_split_label,
        "val_after_train": args.val_after_train,
        "train_cache_path": train_cache,
        "val_cache_path": val_cache,
        "vocab_size": tokenizer.vocab_size,
        "storage_note": "Token caches are self-contained; remote training does not need HuggingFace network access when cache paths are supplied.",
    }
    manifest_path = os.path.join(args.output_dir, f"{safe_name}_ctx{args.context_length}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    if args.hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

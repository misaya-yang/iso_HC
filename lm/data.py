"""Data loaders for LM experiments.

Supports:
  - TinyStories (Hugging Face datasets)
  - WikiText-103
  - FineWeb-Edu (future)

All datasets are tokenized and packed into contiguous sequences.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


def get_tokenizer(vocab_size=50257):
    """Get a GPT-2 tokenizer (ByteLevelBPE)."""
    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError("transformers library required. Install: pip install transformers")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # GPT-2 tokenizer doesn't have pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TokenizedTextDataset(Dataset):
    """Generic tokenized text dataset.

    Loads pre-tokenized data from a .pt file (tensor of token IDs)
    or tokenizes text on-the-fly.
    """

    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class HuggingFaceDataset(Dataset):
    """Load and tokenize a HuggingFace dataset on-the-fly.

    Supports streaming for large datasets.
    """

    def __init__(self, dataset_name, split, text_key, tokenizer,
                 context_length, max_samples=None, shuffle_seed=42):
        super().__init__()
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.text_key = text_key

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required. Install: pip install datasets")

        self.dataset = load_dataset(dataset_name, split=split, streaming=False)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        # Pre-tokenize all text
        self.token_ids = self._tokenize_all()

    def _tokenize_all(self):
        all_ids = []
        for example in self.dataset:
            text = example[self.text_key]
            if text and len(text) > 10:  # skip empty/short
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                all_ids.extend(ids)
                all_ids.append(self.tokenizer.eos_token_id)
        return torch.tensor(all_ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class TinyStoriesDataset(Dataset):
    """TinyStories dataset wrapper."""

    def __init__(self, tokenizer, context_length, split='train', max_samples=None):
        self.context_length = context_length
        self.tokenizer = tokenizer

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required")

        self.dataset = load_dataset('roneneldan/TinyStories', split=split, streaming=False)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        self.token_ids = self._tokenize_all()

    def _tokenize_all(self):
        all_ids = []
        for example in self.dataset:
            text = example.get('text', example.get('story', ''))
            if text and len(text) > 10:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                all_ids.extend(ids)
                all_ids.append(self.tokenizer.eos_token_id)
        return torch.tensor(all_ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloader(dataset_name, tokenizer, context_length, batch_size,
                      split='train', max_samples=None, num_workers=0):
    """Create a dataloader for the specified dataset.

    Args:
        dataset_name: 'tinystories', 'wikitext-103', or a HuggingFace dataset name
        tokenizer: tokenizer instance
        context_length: sequence length
        batch_size: batch size
        split: dataset split
        max_samples: max number of samples to load (for debugging)
        num_workers: dataloader workers
    """
    if dataset_name == 'tinystories':
        dataset = TinyStoriesDataset(tokenizer, context_length, split=split,
                                     max_samples=max_samples)
    elif dataset_name == 'wikitext-103':
        dataset = HuggingFaceDataset('wikitext', split, 'text', tokenizer,
                                     context_length, max_samples=max_samples)
    else:
        # Generic HuggingFace dataset
        dataset = HuggingFaceDataset(dataset_name, split, 'text', tokenizer,
                                     context_length, max_samples=max_samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset


def save_tokenized_cache(dataset_name, tokenizer, cache_path, max_samples=None):
    """Pre-tokenize and cache a dataset to disk."""
    if dataset_name == 'tinystories':
        ds = TinyStoriesDataset(tokenizer, context_length=1024, max_samples=max_samples)
    else:
        raise ValueError(f"Cache not supported for {dataset_name}")

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save(ds.token_ids, cache_path)
    print(f"Saved {len(ds.token_ids)} tokens to {cache_path}")
    return cache_path

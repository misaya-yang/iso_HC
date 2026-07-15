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


def load_token_cache(cache_path):
    """Load token cache with mmap when supported to reduce CPU RAM pressure."""
    try:
        token_ids = torch.load(
            cache_path, map_location='cpu', mmap=True, weights_only=True
        )
    except TypeError:
        token_ids = torch.load(cache_path, map_location='cpu')
    if isinstance(token_ids, dict):
        token_ids = token_ids['token_ids']
    return token_ids


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
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim not in (1, 2):
            raise ValueError("token_ids must be a 1D token stream or 2D row cache")
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if token_ids.ndim == 2 and token_ids.shape[1] < 2:
            raise ValueError("row caches need at least two tokens per row")
        self.token_ids = token_ids
        self.context_length = (
            min(context_length, token_ids.shape[1] - 1)
            if token_ids.ndim == 2
            else context_length
        )

    def __len__(self):
        if self.token_ids.ndim == 2:
            return self.token_ids.shape[0]
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, idx):
        if self.token_ids.ndim == 2:
            chunk = self.token_ids[idx, :self.context_length + 1]
        else:
            chunk = self.token_ids[idx:idx + self.context_length + 1]
        x = chunk[:-1].long()
        y = chunk[1:].long()
        return x, y


class RandomTokenDataset(Dataset):
    """Deterministic random-token dataset for offline smoke tests."""

    def __init__(self, vocab_size, context_length, length=1024, seed=1234):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.length = length
        generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(
            0,
            vocab_size,
            (length + context_length + 1,),
            generator=generator,
            dtype=torch.long,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.context_length + 1]
        return chunk[:-1], chunk[1:]


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


def create_dataset(dataset_name, tokenizer, context_length, split='train',
                   max_samples=None, cache_path=None):
    """Create a dataset, optionally loading pre-tokenized token IDs."""
    if cache_path is not None and os.path.exists(cache_path):
        token_ids = load_token_cache(cache_path)
        return TokenizedTextDataset(token_ids, context_length)

    if dataset_name == 'random':
        vocab_size = getattr(tokenizer, 'vocab_size', 50304)
        return RandomTokenDataset(
            vocab_size=vocab_size,
            context_length=context_length,
            length=max_samples or 4096,
            seed=42 if split == 'train' else 43,
        )
    elif dataset_name == 'tinystories':
        return TinyStoriesDataset(tokenizer, context_length, split=split,
                                  max_samples=max_samples)
    elif dataset_name == 'wikitext-103':
        return HuggingFaceDataset('wikitext', split, 'text', tokenizer,
                                  context_length, max_samples=max_samples)
    else:
        return HuggingFaceDataset(dataset_name, split, 'text', tokenizer,
                                  context_length, max_samples=max_samples)


def save_token_cache(dataset_name, tokenizer, cache_path, context_length=1024,
                     split='train', max_samples=None):
    """Create and save a token cache for fast future dataloader startup."""
    dataset = create_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        context_length=context_length,
        split=split,
        max_samples=max_samples,
        cache_path=None,
    )
    token_ids = getattr(dataset, 'token_ids', None)
    if token_ids is None:
        token_ids = getattr(dataset, 'tokens', None)
    if token_ids is None:
        raise ValueError(f"Dataset {dataset_name} does not expose token IDs")

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save(token_ids.cpu().long(), cache_path)
    print(f"Saved {len(token_ids)} tokens to {cache_path}")
    return cache_path


def create_dataloader(dataset_name, tokenizer, context_length, batch_size,
                      split='train', max_samples=None, num_workers=0,
                      cache_path=None, prefetch_factor=4,
                      persistent_workers=True, drop_last=True):
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
    dataset = create_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        context_length=context_length,
        split=split,
        max_samples=max_samples,
        cache_path=cache_path,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": (split == 'train'),
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, dataset


def save_tokenized_cache(dataset_name, tokenizer, cache_path, max_samples=None):
    """Pre-tokenize and cache a dataset to disk."""
    return save_token_cache(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        cache_path=cache_path,
        context_length=1024,
        split='train',
        max_samples=max_samples,
    )

# Compatibility entry

Current agent instructions are in [AGENTS.md](AGENTS.md). Research authority and document statuses are defined there. This file only retains historical server storage conventions; it does not define the current experiment or scientific claim.

## Historical server storage conventions

The previous setup used root at `connect.westd.seetacloud.com`, with an instance-specific port, source `/root/isoHC`, Python `/root/miniconda3/bin/python3`, and data/results `/root/autodl-tmp/isoHC`. Verify current endpoints and runtime before use; these are not a live connection record.

Keep large caches, temporary files, datasets and results on the data disk:

```bash
export HF_HOME=/root/autodl-tmp/isoHC/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/isoHC/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/isoHC/hf_cache/transformers
export TMPDIR=/root/autodl-tmp/isoHC/tmp
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/isoHC/torchinductor_cache
```

Historical pipeline commands are described in `experiments/README.md`; they are not the active experiment plan.

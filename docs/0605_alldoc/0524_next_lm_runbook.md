# 0524 IsoHC Next LM Runbook

## 本地已验证

```bash
python3 -m unittest tests.test_lm_next_phase_contracts tests.test_stage1_contracts -v
python3 experiments/lm_verify.py
python3 experiments/lm_5090_next_runs.py --preset run0 --methods baseline isohc \
  --dataset random --output_dir /tmp/isohc_lm_run0_smoke \
  --total_tokens 1024 --batch_size 2 --no_compile --num_workers 0 \
  --max_samples 32 --max_samples_val 16
python3 experiments/lm_5090_next_runs.py --preset headmix --methods baseline headmix-iso \
  --dataset random --output_dir /tmp/isohc_lm_headmix_smoke \
  --total_tokens 512 --batch_size 1 --no_compile --num_workers 0 \
  --max_samples 16 --max_samples_val 8
```

## 同步到服务器

服务器开机后：

```bash
rsync -avz --exclude='*.pdf' --exclude='docs/0605_alldoc/0510_results_raw' \
  -e 'ssh -p 19596' \
  /Users/yang/projects/isoHC/ \
  root@connect.westd.seetacloud.com:/root/isoHC/
```

## 数据环境注意事项：中国服务器无 HuggingFace 外网

当前服务器在中国网络环境下可能不能直连 HuggingFace；无卡模式已经观察到：

```text
Network is unreachable ... https://huggingface.co/gpt2/resolve/main/tokenizer_config.json
```

因此正式实验不要依赖服务器在线下载 HuggingFace tokenizer/dataset。推荐路径：

1. 在本地用 streaming 方式生成 TinyStories token cache，只保留最终 `.pt` 文件。
2. 将 `data/lm_cache/*.pt` 和 manifest 上传到服务器数据盘，不放系统盘。
3. 服务器训练命令必须传 `--train_cache_path`、`--val_cache_path`、`--vocab_size 50257`，这样 runner 不会调用 HuggingFace。
4. 上传确认后，删除本地 HuggingFace cache 和本地 `.pt` 中间文件，避免占用本机存储。

服务器存储约定：

- 代码目录：`/root/isoHC`
- 数据盘：`/root/autodl-tmp/isoHC`
- token cache：`/root/autodl-tmp/isoHC/data/lm_cache`
- 后续大结果：`/root/autodl-tmp/isoHC/results`
- 不要把新数据集或长跑 checkpoint 放到 `/root/isoHC/data` 或 `/root/isoHC/results`，这两个在 30GB 系统盘上。

本地生成 cache：

```bash
cd /Users/yang/projects/isoHC
mkdir -p data/lm_cache
python3 experiments/prepare_lm_data.py \
  --dataset tinystories \
  --context_length 512 \
  --output_dir data/lm_cache \
  --target_train_tokens 25000000 \
  --target_val_tokens 1000000
```

上传 cache：

```bash
rsync -avz -e 'ssh -p 19596' \
  /Users/yang/projects/isoHC/data/lm_cache/ \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/isoHC/data/lm_cache/
```

本地清理：

```bash
rm -rf /Users/yang/projects/isoHC/data/lm_cache
rm -rf ~/.cache/huggingface/datasets/roneneldan___tiny_stories
```

如果以后要尝试服务器侧下载，先尝试镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

DashScope/ModelScope 可以作为后续数据通道，但当前 0524 实验脚本默认使用本地预生成 token cache，不依赖 DashScope API。

FineWeb-Edu 小切片准备命令：

```bash
cd /Users/yang/projects/isoHC
tmp_cache=/tmp/isohc_fineweb_edu_cache
rm -rf "$tmp_cache"
python3 experiments/prepare_lm_data.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --text_key text \
  --context_length 512 \
  --output_dir "$tmp_cache" \
  --train_split train \
  --val_after_train \
  --target_train_tokens 100000000 \
  --target_val_tokens 2000000 \
  --hard_exit
rsync -avz -e 'ssh -p 19596' \
  "$tmp_cache"/ \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/isoHC/data/lm_cache/
rm -rf "$tmp_cache"
```

## Run 0: 30-60 分钟正确性

如果服务器镜像可用，可以在服务器准备 token cache；否则使用上一节的本地生成并上传流程。

```bash
cd /root/isoHC
export HF_ENDPOINT=https://hf-mirror.com
/root/miniconda3/bin/python3 experiments/prepare_lm_data.py \
  --dataset tinystories \
  --context_length 512 \
  --output_dir /root/autodl-tmp/isoHC/data/lm_cache
```

```bash
ssh -p 19596 root@connect.westd.seetacloud.com
cd /root/isoHC
mkdir -p /root/autodl-tmp/isoHC/results/0524_run0
/root/miniconda3/bin/python3 experiments/lm_5090_next_runs.py \
  --preset run0 \
  --methods baseline unconstrained mhc isohc \
  --dataset tinystories \
  --output_dir /root/autodl-tmp/isoHC/results/0524_run0 \
  --total_tokens 1000000 \
  --batch_size 16 \
  --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_train_ctx512.pt \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_validation_ctx512.pt \
  --vocab_size 50257 \
  --compile_mode reduce-overhead \
  --num_workers 2
```

## Run 1: 最高优先级 deep-stress LM

如果只够跑一个方向，跑这个。默认使用 bf16 AMP、SDPA/FlashAttention 路径、fused AdamW、`torch.compile(max-autotune)`，并用 `--auto_batch` 向 32GB 显存靠近。

```bash
cd /root/isoHC
mkdir -p /root/autodl-tmp/isoHC/results/0524_deep_stress_512
nohup /root/miniconda3/bin/python3 experiments/lm_5090_next_runs.py \
  --preset deep-stress-512 \
  --methods baseline unconstrained mhc isohc \
  --dataset tinystories \
  --output_dir /root/autodl-tmp/isoHC/results/0524_deep_stress_512 \
  --total_tokens 20000000 \
  --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_train_ctx512.pt \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_validation_ctx512.pt \
  --vocab_size 50257 \
  --auto_batch \
  --memory_target_gb 29 \
  --compile_mode max-autotune \
  --num_workers 4 \
  > /root/autodl-tmp/isoHC/results/0524_deep_stress_512/runner.out 2>&1 &
```

## Run 2: 125M smoke

这只作为能否跑、overhead、稳定性证据，不写成 PPL 主结论。

```bash
cd /root/isoHC
mkdir -p /root/autodl-tmp/isoHC/results/0524_125m_smoke
nohup /root/miniconda3/bin/python3 experiments/lm_5090_next_runs.py \
  --preset 125m-smoke \
  --methods mhc isohc \
  --dataset tinystories \
  --output_dir /root/autodl-tmp/isoHC/results/0524_125m_smoke \
  --total_tokens 5000000 \
  --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_train_ctx512.pt \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_validation_ctx512.pt \
  --vocab_size 50257 \
  --auto_batch \
  --memory_target_gb 29 \
  --compile_mode max-autotune \
  --num_workers 4 \
  > /root/autodl-tmp/isoHC/results/0524_125m_smoke/runner.out 2>&1 &
```

## Run 3: HeadMix micro-ablation

这个作为扩展实验，不替代 residual-stream 主实验。

```bash
cd /root/isoHC
mkdir -p /root/autodl-tmp/isoHC/results/0524_headmix
nohup /root/miniconda3/bin/python3 experiments/lm_5090_next_runs.py \
  --preset headmix \
  --methods baseline headmix-unconstrained headmix-birkhoff headmix-iso headmix-fixed-random-iso \
  --dataset tinystories \
  --output_dir /root/autodl-tmp/isoHC/results/0524_headmix \
  --total_tokens 5000000 \
  --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_train_ctx512.pt \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/tinystories_validation_ctx512.pt \
  --vocab_size 50257 \
  --auto_batch \
  --memory_target_gb 29 \
  --compile_mode max-autotune \
  --num_workers 4 \
  > /root/autodl-tmp/isoHC/results/0524_headmix/runner.out 2>&1 &
```

## 监控与取回

```bash
nvidia-smi
tail -f /root/autodl-tmp/isoHC/results/0524_deep_stress_512/runner.out
find /root/autodl-tmp/isoHC/results/0524_deep_stress_512 -name run_summary.json -print
```

本地取回：

```bash
rsync -avz -e 'ssh -p 19596' \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/isoHC/results/0524_deep_stress_512 \
  /Users/yang/projects/isoHC/docs/0605_alldoc/0524_results_raw/
```

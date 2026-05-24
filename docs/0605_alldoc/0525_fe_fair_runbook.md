# 0525 FE Fair Deep-Stress Runbook

目标：在 32GB 5090 上重跑公平的 FE/FineWeb-Edu deep-stress LM，对齐 batch、optimizer steps 和 token budget，避免 0524 TinyStories 中 baseline 自动拿到更大 batch 的问题。

## 关键约束

- 数据和结果都放在数据盘：`/root/autodl-tmp/isoHC`，不要写系统盘。
- 服务器在中国网络环境下不要依赖直连 Hugging Face；FE token cache 已经提前准备好。
- 主实验不使用逐方法 `--auto_batch`；使用 `--fair_auto_batch` 先探测所有方法，再统一采用所有方法都能跑的最大 common batch。
- 所有方法共享：同一 `preset`、同一 `batch_size`、同一 `grad_accum_steps`、同一 `total_tokens`、同一 tokenizer/cache、同一 eval 设置。
- 大模型机制实验默认 `--no_save_checkpoints`，只保存 `summary.json`、`run_summary.json` 和 fair batch probe，避免数据盘写大 checkpoint 干扰速度。

## 已准备的数据

远端数据盘：

```bash
/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt
/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt
/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_ctx512_manifest.json
```

manifest 摘要：

- train tokens: `100,002,129`
- heldout tokens: `2,009,986`
- vocab size: `50257`
- context length: `512`

## GPU 开机后的正式启动命令

```bash
ssh -p 19596 root@connect.westd.seetacloud.com
cd /root/isoHC
chmod +x experiments/run_0525_fe_fair_deep.sh
mkdir -p /root/autodl-tmp/isoHC/results/0525_fe_fair_deep36
nohup bash experiments/run_0525_fe_fair_deep.sh \
  > /root/autodl-tmp/isoHC/results/0525_fe_fair_deep36/launcher.out 2>&1 &
```

默认配置：

```bash
PRESET=fe-deep-36l-512
TOTAL_TOKENS=30000000
MEMORY_TARGET_GB=29
GRAD_ACCUM_STEPS=1
NUM_WORKERS=4
PREFETCH_FACTOR=4
```

如果 36L 仍然显存利用不足，可以开机后直接改用 48L stress：

```bash
PRESET=fe-deep-48l-512 \
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_fe_fair_deep48 \
TOTAL_TOKENS=20000000 \
nohup bash experiments/run_0525_fe_fair_deep.sh \
  > /root/autodl-tmp/isoHC/results/0525_fe_fair_deep48/launcher.out 2>&1 &
```

## 监控命令

```bash
nvidia-smi
tail -n 80 /root/autodl-tmp/isoHC/results/0525_fe_fair_deep36/launcher.out
find /root/autodl-tmp/isoHC/results/0525_fe_fair_deep36 -maxdepth 2 -name 'run_summary.json' -o -name 'summary.json'
```

fair batch probe 文件：

```bash
/root/autodl-tmp/isoHC/results/0525_fe_fair_deep36/fe-deep-36l-512_fair_batch_probe.json
```

这个文件必须显示所有方法最终使用同一个 common batch，才能进入论文表格或机制图。

## 结果解读优先级

第一优先级不是 PPL 单点胜负，而是：

- `1^\perp` mean-zero energy curve：mHC 是否扩散/收缩，IsoHC 是否保留。
- stream cosine：mHC 是否更接近 collapse，IsoHC 是否保持 diversity。
- mixer complement singular values：IsoHC 是否集中在 1，mHC 是否小于 1。
- gradient profile：更深模型里 IsoHC 是否比 mHC/unconstrained 更稳。

只有在这些机制指标成立且 val loss 不差时，才考虑把 FE run 扩展到更长 token budget 或 A100 controlled run。

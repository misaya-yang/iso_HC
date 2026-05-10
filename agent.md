# IsoHC Stage 1 Agent Guide

## 服务器信息

| 项目 | 值 |
|------|-----|
| 主机 | connect.westd.seetacloud.com |
| 端口 | 19596 |
| 用户 | root |
| 认证 | 免密 SSH (已配置) |
| OS | Ubuntu 22.04.5 LTS |

## 代码位置

服务器路径: `/root/isoHC/`

```
/root/isoHC/
├── isohc/                          # 核心模块
│   ├── __init__.py
│   ├── projection.py               # Iso-NS 投影 (SVD + NS 双模式)
│   ├── layers.py                   # RMSNorm, Attention, MLP, IsoHC Mixing
│   └── transformer.py              # IsoHC / Baseline / Unconstrained Transformer
├── experiments/
│   ├── stage1_projection_sanity.py # 实验1: 投影正确性
│   ├── stage1_residual_only.py     # 实验2: 深层稳定性
│   └── stage1_tiny_smoke.py        # 实验3: Tiny Transformer 训练
├── run_stage1.py                   # 一键运行全部实验
└── agent.md                        # 本文件
```

本地路径: `/Users/yang/projects/isoHC/`

## Python 环境

| 项目 | 值 |
|------|-----|
| Python | 3.12.3 (miniconda3 base) |
| PyTorch | 2.8.0+cu128 |
| NumPy | 2.3.2 |
| CUDA | 12.8 (驱动已装，当前无卡模式) |
| Python 路径 | `/root/miniconda3/bin/python3` |

## 运行命令

```bash
# SSH 连接
ssh -p 19596 root@connect.westd.seetacloud.com

# 进入项目目录
cd /root/isoHC

# 一键运行全部实验 (CPU 模式)
/root/miniconda3/bin/python3 run_stage1.py --device cpu --quick

# 单独运行实验
/root/miniconda3/bin/python3 experiments/stage1_projection_sanity.py --device cpu
/root/miniconda3/bin/python3 experiments/stage1_residual_only.py --device cpu --dtype-tensor fp32
/root/miniconda3/bin/python3 experiments/stage1_tiny_smoke.py --model isohc --device cpu --precision fp32

# GPU 模式 (切换到有卡模式后)
/root/miniconda3/bin/python3 run_stage1.py --device cuda
```

## 核心设计

- **e0 = 1/sqrt(n)** (非 1/n)
- **U 构造**: 确定性 QR 分解，`torch.cat([e0, eye[:, 1:]])`
- **Polar decomposition**: SVD 模式用于高精度 forward-only 测试；NS 迭代用于训练 backward
- **H_raw 初始化**: identity，确保训练初期 NS 快速收敛
- **IsoHC 混合**: 仅替换 Attention 残差路径，MLP 保持标准残差

## 同步代码到服务器

```bash
rsync -avz --exclude='*.pdf' --exclude='*.md' -e 'ssh -p 19596' \
  /Users/yang/projects/isoHC/isohc \
  /Users/yang/projects/isoHC/experiments \
  /Users/yang/projects/isoHC/run_stage1.py \
  root@connect.westd.seetacloud.com:/root/isoHC/
```

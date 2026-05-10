# IsoHC 第一阶段验证计划（单卡 RTX 5090，≤2 小时）

## 目标

本阶段只验证 IsoHC 是否具备继续投入实验的最低可行性。  
不追求 benchmark，不做完整语言模型训练，不证明最终性能优势。

本阶段要回答三个问题：

1. **Iso-NS 数学投影是否正确。**
2. **IsoHC 是否在 residual-only 深层传播中保持能量和梯度。**
3. **IsoHC 是否能在一个极小 Transformer smoke test 中稳定完成前向、反向和短训练。**

如果这三个问题都通过，再进入第二阶段：真实小规模 LM 训练、深度扩展、mHC 对照和 ablation。

---

## 时间预算

总时间控制在 **2 小时以内**。

| 模块 | 预计时间 |
|---|---:|
| 环境检查与脚本启动 | 10 min |
| 实验 1：Iso-NS 投影正确性 | 15 min |
| 实验 2：Residual-only 深层稳定性 | 25 min |
| 实验 3：Tiny Transformer smoke test | 45-60 min |
| 汇总结果与判断 | 10 min |

如果实验 1 或实验 2 失败，直接停止，不进入实验 3。

---

## 硬件假设

- GPU: 单卡 RTX 5090
- 显存: 32GB
- Precision: bf16 mixed precision
- Iso-NS 内部建议使用 fp32

---

## Stage 1 总体通过标准

第一阶段通过的最低标准：

1. Iso-NS 输出满足：

   $$
   \|H^\top H-I\|_F < 10^{-3}
   $$

   $$
   \|H\mathbf 1-\mathbf 1\|_2 < 10^{-5}
   $$

2. Residual-only 深层传播中，IsoHC 在 $L=256$ 下保持：

   $$
   0.98 \leq \frac{\|X_L\|_F}{\|X_0\|_F} \leq 1.02
   $$

   $$
   0.98 \leq \frac{\|G_0\|_F}{\|G_L\|_F} \leq 1.02
   $$

3. Tiny Transformer smoke test 中：

   - 无 NaN；
   - 无 loss explosion；
   - global grad norm 有界；
   - IsoHC 的 projection error 在训练过程中没有漂移；
   - 能稳定跑完 300-500 steps。

如果以上全部满足，则进入下一阶段。

---

# 实验 1：Iso-NS 投影正确性

## 目的

验证 Iso-NS 是否真的把任意矩阵投影到：

$$
\mathcal M_{\mathrm{iso}}
=
\{H\in\mathbb R^{n\times n}: H^\top H=I,\ H\mathbf 1=\mathbf 1\}
$$

即同时满足正交性和固定均值方向。

---

## 配置

```yaml
experiment: projection_sanity
streams: [4, 8, 16]
ns_steps: [1, 2, 3, 5]
num_random_matrices: 128
dtype_projection: fp32
device: cuda
```

---

## 必须检查的实现细节

### 1. 均值方向

必须使用：

$$
e_0=\frac{\mathbf 1}{\sqrt n}
$$

不要写成：

$$
\frac{\mathbf 1}{n}
$$

### 2. 正交补空间基

构造：

$$
U\in\mathbb R^{n\times(n-1)}
$$

满足：

$$
U^\top U=I
$$

$$
U^\top e_0=0
$$

### 3. 子空间投影

给定 unconstrained matrix $\tilde H$，计算：

$$
A=U^\top \tilde H U
$$

然后对 $A$ 做 Newton-Schulz polar approximation：

$$
R_K\approx \operatorname{polar}(A)
$$

最后重构：

$$
H=e_0e_0^\top+UR_KU^\top
$$

---

## 记录指标

对每个 $n$ 和 $K$，记录均值和最大值：

$$
\epsilon_{\mathrm{orth}}
=
\|H^\top H-I\|_F
$$

$$
\epsilon_{\mathrm{fix}}
=
\|H\mathbf 1-\mathbf 1\|_2
$$

$$
\epsilon_{\mathrm{energy}}
=
\left|
\frac{\|HX\|_F}{\|X\|_F}-1
\right|
$$

其中：

$$
X\sim\mathcal N(0,1)
$$

---

## 通过标准

推荐使用 $K=5$。

实验 1 通过条件：

```text
for n in {4, 8, 16}:
  mean orth_error at K=5 < 1e-3
  max  orth_error at K=5 < 5e-3
  mean fix_error  at K=5 < 1e-6
  max  fix_error  at K=5 < 1e-5
  mean energy_error at K=5 < 1e-3
```

如果 $K=3$ 已经达到上述标准，后续 smoke test 可以优先使用 $K=3$。  
如果 $K=5$ 仍未通过，停止后续实验，优先修复 Iso-NS 实现。

---

# 实验 2：Residual-only 深层稳定性

## 目的

隔离 Transformer 的非线性模块，只测试 residual mixing 的深层乘积稳定性。

测试系统：

$$
X_{l+1}=H_lX_l
$$

反向传播对应：

$$
G_l=H_l^\top G_{l+1}
$$

IsoHC 理论上应保持 forward energy 和 backward gradient norm。

---

## 对照方法

本阶段只需要三个方法：

| 方法 | 说明 |
|---|---|
| Unconstrained HC | 不加约束的随机 residual mixing |
| mHC-lite | Sinkhorn doubly-stochastic projection |
| IsoHC | Iso-NS fixed-vector orthogonal projection |

如果时间紧，可以先只跑 Unconstrained HC 和 IsoHC。  
但推荐保留 mHC-lite，因为它能初步观察 mean-preserving diffusion 的收缩行为。

---

## 配置

```yaml
experiment: residual_only_depth
streams: [4, 8]
feature_dim: 512
depths: [32, 64, 128, 256]
num_trials: 32
ns_steps: 5
dtype_projection: fp32
dtype_tensor: bf16
device: cuda
```

---

## 输入设置

需要测试两类输入。

### 1. 普通随机输入

$$
X_0\sim\mathcal N(0,1)
$$

### 2. Mean-zero 输入

$$
X_0^\perp
=
X_0
-
\frac{1}{n}\mathbf 1\mathbf 1^\top X_0
$$

mean-zero 输入很重要，因为 mHC 的能量收缩通常在 $\mathbf 1^\perp$ 子空间上更明显。

---

## 记录指标

### Forward energy ratio

$$
r_L
=
\frac{\|X_L\|_F}{\|X_0\|_F}
$$

### Mean preservation error

$$
m_L
=
\left\|
\frac{1}{n}\mathbf 1^\top X_L
-
\frac{1}{n}\mathbf 1^\top X_0
\right\|_2
$$

### Backward gradient ratio

随机初始化：

$$
G_L\sim\mathcal N(0,1)
$$

然后反向传播：

$$
G_l=H_l^\top G_{l+1}
$$

记录：

$$
g_0=
\frac{\|G_0\|_F}{\|G_L\|_F}
$$

### Singular value diagnostics

对每层 $H_l$ 记录：

$$
\sigma_{\max}(H_l)
$$

$$
\sigma_{\min}(H_l)
$$

IsoHC 应接近：

$$
\sigma_{\max}(H_l)\approx 1
$$

$$
\sigma_{\min}(H_l)\approx 1
$$

---

## 通过标准

实验 2 通过条件：

```text
IsoHC at L=256:
  forward energy ratio in [0.98, 1.02]
  backward grad ratio in [0.98, 1.02]
  mean preservation error < 1e-5
  per-layer max singular value <= 1.01
  per-layer min singular value >= 0.99
```

对照方法预期：

```text
Unconstrained HC:
  likely explode or vanish as depth increases

mHC-lite:
  mean preserved
  energy may contract on mean-zero input
```

如果 IsoHC 在 residual-only 中失败，停止实验。  
这说明核心数学实现或数值稳定性还没过关。

---

# 实验 3：Tiny Transformer Smoke Test

## 目的

验证 IsoHC 能否插入一个极小 Transformer 并稳定完成短训练。

本实验不是性能实验，只是 smoke test。

---

## 模型配置

为了保证 2 小时内完成，模型必须小。

推荐配置：

```yaml
experiment: tiny_transformer_smoke
model:
  hidden_dim: 256
  num_layers: 16
  num_heads: 4
  mlp_ratio: 4
  streams: 4
  context_length: 256
  dropout: 0.0
  norm: rmsnorm
  residual_mixing: isohc
  ns_steps: 5

training:
  steps: 300
  eval_every: 100
  batch_size: 8
  grad_accum_steps: 4
  precision: bf16
  grad_clip: 1.0
  optimizer: adamw
  learning_rate: 3e-4
  weight_decay: 0.1
  warmup_steps: 50
```

如果实现很快，也可以跑：

```yaml
steps: 500
```

但第一目标是稳定跑完 300 steps。

---

## 数据

为了避免数据准备占用时间，优先使用 synthetic language modeling data。

### Synthetic LM Data

随机 token：

```text
vocab_size = 4096
seq_len = 256
```

训练目标：

```text
next-token prediction
```

这不是为了测真实性能，而是为了验证：

1. forward pass 是否正确；
2. backward pass 是否正确；
3. IsoHC 参数是否可学习；
4. projection error 是否随训练保持稳定；
5. loss 是否不会 NaN 或爆炸。

如果本地已经有 TinyStories / WikiText cache，也可以使用真实数据。  
但第一阶段不依赖真实数据。

---

## 对照设置

第一阶段 smoke test 至少跑两个版本：

```text
A. Standard residual Transformer
B. IsoHC Transformer
```

如果时间允许，再加：

```text
C. Unconstrained HC
```

第一阶段不强制跑 mHC，因为 mHC 对照可以留到第二阶段。

---

## 记录指标

每 20 steps 记录：

```text
train_loss
global_grad_norm
max_abs_param
max_abs_activation
gpu_memory_allocated
step_time
```

IsoHC 额外记录：

$$
\|H_l^\top H_l-I\|_F
$$

$$
\|H_l\mathbf 1-\mathbf 1\|_2
$$

$$
\frac{\|H_lX_l\|_F}{\|X_l\|_F}
$$

并统计所有层的：

```text
mean orth_error
max orth_error
mean fix_error
max fix_error
mean residual_energy_ratio
```

---

## 通过标准

实验 3 通过条件：

```text
IsoHC Transformer:
  completes 300 steps
  no NaN
  no Inf
  loss decreases or remains numerically stable
  global_grad_norm does not explode
  max orth_error < 5e-3
  max fix_error < 1e-5
  residual_energy_ratio mostly within [0.98, 1.02]
```

这里的 loss 不需要优于 baseline。  
只要能稳定训练，就是第一阶段通过。

---

# 推荐执行顺序

## Step 0：环境检查

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

---

## Step 1：运行 projection sanity

```bash
python experiments/stage1_projection_sanity.py \
  --streams 4 8 16 \
  --ns-steps 1 2 3 5 \
  --num-random-matrices 128 \
  --device cuda
```

输出文件：

```text
results/stage1/projection_sanity.json
results/stage1/projection_sanity.md
```

---

## Step 2：运行 residual-only depth test

```bash
python experiments/stage1_residual_only.py \
  --streams 4 8 \
  --depths 32 64 128 256 \
  --feature-dim 512 \
  --num-trials 32 \
  --ns-steps 5 \
  --device cuda
```

输出文件：

```text
results/stage1/residual_only.json
results/stage1/residual_only.md
results/stage1/residual_energy.png
results/stage1/residual_gradient.png
```

---

## Step 3：运行 tiny Transformer smoke test

```bash
python experiments/stage1_tiny_smoke.py \
  --model isohc \
  --layers 16 \
  --hidden-dim 256 \
  --heads 4 \
  --streams 4 \
  --context-length 256 \
  --steps 300 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --ns-steps 5 \
  --precision bf16
```

可选 baseline：

```bash
python experiments/stage1_tiny_smoke.py \
  --model baseline \
  --layers 16 \
  --hidden-dim 256 \
  --heads 4 \
  --context-length 256 \
  --steps 300 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --precision bf16
```

---

# Stage 1 结果汇总模板

实验完成后，用下面格式汇总。

```markdown
# Stage 1 Result Summary

## Projection Sanity

| n | K | mean orth error | max orth error | mean fix error | max fix error | pass |
|---|---:|---:|---:|---:|---:|---|
| 4 | 5 | | | | | |
| 8 | 5 | | | | | |
| 16 | 5 | | | | | |

## Residual-only Depth

| method | n | L | energy ratio | grad ratio | mean error | pass |
|---|---:|---:|---:|---:|---:|---|
| IsoHC | 4 | 256 | | | | |
| IsoHC | 8 | 256 | | | | |

## Tiny Transformer Smoke Test

| model | steps completed | final loss | max grad norm | max orth error | max fix error | pass |
|---|---:|---:|---:|---:|---:|---|
| baseline | | | | n/a | n/a | |
| IsoHC | | | | | | |

## Decision

Stage 1 decision:

- [ ] Pass: proceed to Stage 2
- [ ] Fail: fix implementation before further training

Notes:
-
-
-
```

---

# 失败时的优先排查顺序

## 如果 projection sanity 失败

按顺序检查：

1. $e_0$ 是否为 $\mathbf 1/\sqrt n$；
2. $U^\top U$ 是否接近 $I$；
3. $U^\top e_0$ 是否接近 0；
4. reconstruction 是否为 $e_0e_0^\top+URU^\top$；
5. Newton-Schulz scaling 是否保证谱半径合适；
6. 是否在 fp32 中执行 Iso-NS。

---

## 如果 residual-only 失败

按顺序检查：

1. 每一层 $H_l$ 的 orthogonality error；
2. 每一层 $H_l\mathbf 1-\mathbf 1$；
3. $R_K^\top R_K-I$ 是否已经很大；
4. bf16 是否导致误差，尝试全 fp32；
5. $K=5$ 是否不够，尝试 $K=8$；
6. random matrix 初始化是否过大。

---

## 如果 tiny smoke test 失败

按顺序检查：

1. learning rate 降到 $1e-4$；
2. grad clip 降到 0.5；
3. Iso-NS 内部强制 fp32；
4. $H_{\mathrm{raw}}$ 初始化为 identity-like；
5. layers 从 16 降到 8；
6. ns_steps 从 5 增加到 8；
7. 关闭 dropout；
8. 确认 loss explosion 不是普通 Transformer 代码问题。

---

# 第一阶段结论标准

只有当以下三项都通过时，才进入第二阶段：

```text
[ ] Projection sanity passed
[ ] Residual-only depth stability passed
[ ] Tiny Transformer smoke test passed
```

如果只通过前两项、第三项失败，不应否定数学理论。  
这时优先判断为工程集成问题。

如果第一项失败，说明 Iso-NS 实现错误。  
如果第二项失败，说明数值误差或近似正交性还不够。  
如果第三项失败，说明需要调整训练超参或 HC block 集成方式。

---

# 进入第二阶段的条件

进入第二阶段前，必须拿到以下最小证据：

1. 一张 projection error vs Newton-Schulz steps 的表或图；
2. 一张 residual-only energy ratio across depth 的图；
3. 一张 residual-only gradient ratio across depth 的图；
4. tiny Transformer smoke test 的 loss curve；
5. IsoHC 每层 projection error 的日志。

满足这些条件后，第二阶段再规划：

- mHC 正式对照；
- 32/48/64 层深度扩展；
- TinyStories 或 WikiText 真实数据；
- $K$ 步数 ablation；
- stream 数量 ablation；
- gradient profile 和 stream diversity 诊断。

# Diff-Protect

面向扩散模型图像生成的对抗扰动防护框架。

![](test_images/media/teaser.png)

## 概述

Diff-Protect 生成不可感知的对抗扰动，将其叠加到图像上后，可以破坏扩散模型的生成过程。本框架目前支持两个模型系列：

| 特性 | SD v1.4 (`diff_mist.py`) | SD 3.5 (`diff_mist_SD3.py`) |
|------|--------------------------|-----------------------------|
| 网络架构 | UNet + ε-预测 | MMDiT + v-预测 (Flow Matching) |
| 文本编码器 | CLIP × 1 | CLIP-L + CLIP-G + T5 |
| 注意力机制 | Cross-Attention | Joint Attention (双流) |
| 攻击损失 | 语义损失 / 纹理损失 / 联合损失 | 纹理损失 + 4 种 MMDiT 专用损失 |

---

## 快速配置

### SD v1.4 环境

```bash
conda env create -f env.yml
conda activate mist
pip install --force-reinstall pillow
```

下载 [Stable Diffusion v1.4 模型权重](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)：

```bash
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir ckpt
mv sd-v1-4.ckpt ckpt/model.ckpt
```

### SD 3.5 环境

使用 `dino` conda 环境（需安装 PyTorch 2.0+ 及 modelscope）：

```bash
conda activate dino
# modelscope, diffusers, transformers, accelerate 应已预装
```

SD 3.5 模型权重从 ModelScope 加载（`stabilityai/stable-diffusion-3.5-medium`），使用本地缓存 `~/.cache/modelscope/hub/`，首次运行时自动下载。


## 运行防护

### SD v1.4 攻击

配置文件位于 `configs/attack/base.yaml`：

```yaml
attack:
    epsilon: 16           # L_inf 扰动预算（像素单位）
    steps: 100            # 攻击迭代次数
    input_size: 512       # 图像分辨率
    mode: sds             # 攻击模式
    img_path: test_images/to_protect
    output_path: out/
    alpha: 1              # 步长
    g_mode: "+"           # 梯度方向
    device: 0             # GPU 编号
```

可用模式：

| 模式 | 损失函数 | 命令 |
|------|----------|------|
| **AdvDM** | 仅语义损失 | `python code/diff_mist.py attack.mode='advdm' attack.g_mode='+' attack.device="cuda:1"` |
| **PhotoGuard** | 仅纹理损失 | `python code/diff_mist.py attack.mode='texture_only' attack.g_mode='+' attack.device="cuda:1"` |
| **Mist** | 纹理损失 + 语义损失（联合） | `python code/diff_mist.py attack.mode='mist' attack.g_mode='+' attack.device="cuda:1"` |
| **SDS(+)** | SDS 加速，梯度上升 | `python code/diff_mist.py attack.mode='sds' attack.g_mode='+' attack.device="cuda:1"` |
| **SDS(-)** | SDS 加速，梯度下降 | `python code/diff_mist.py attack.mode='sds' attack.g_mode='-' attack.device="cuda:1"` |
| **SDST(-)** | SDS + 目标纹理损失 | `python code/diff_mist.py attack.mode='sds' attack.g_mode='-' attack.using_target=True attack.device="cuda:1"` |

输出文件：
- `[NAME]_attacked.png` — 添加对抗扰动后的图像
- `[NAME]_onestep.png` — 单步去噪预测结果
- `[NAME]_multistep.png` — 多步 SDEdit 结果

<img src="out/advdm_eps16_steps100_gmode+/to_protect/suzume_attacked.png" alt="drawing" width="200"/>  <img src="out/mist_eps16_steps100_gmode+/to_protect/suzume_attacked.png" alt="drawing" width="200"/> <img src="out/sds_eps16_steps100_gmode-/to_protect/suzume_attacked.png" alt="drawing" width="200"/>

*[从左到右]：AdvDM、Mist、SDS(-)，使用 eps=16。SDS 版本的效果明显优于前两种方法。*


### SD 3.5 MMDiT 攻击

配置文件位于 `configs/attack/base_sd3.yaml`：

```yaml
attack:
    epsilon: 16           # L_inf 扰动预算（像素单位）
    steps: 100            # 攻击迭代次数
    input_size: 1024      # 图像分辨率
    mode: A               # MMDiT 攻击模式：O/A/B/C/D
    img_path: test_images/to_protect
    output_path: out_sd3/
    alpha: 1              # 步长
    g_mode: "+"           # 梯度方向：'+' 或 '-'
    textual_weight: 1.0   # 纹理损失权重（VAE 潜空间推离）
    mmdit_weight: 1.0     # MMDiT 专用损失权重
    device: "cuda:1"      # GPU 设备
    model_name: "stabilityai/stable-diffusion-3.5-medium"  # ModelScope 模型 ID
```

#### MMDiT 攻击模式

联合损失遵循如下设计：**JOINT LOSS = TEXTURAL LOSS + λ · MMDiT LOSS**，其中 TEXTURAL LOSS 将 VAE 潜空间表示推向错误方向（与原始 Mist 相同），MMDiT LOSS 则利用 SD3 多模态 DiT 的架构特性进行攻击。

**g_mode='+'** 表示梯度上升（gradient ascent），即最大化损失：
- textural loss 越大越好：图像在 VAE latent 中远离目标图像
- mmdit loss 越大越好：语义差异越大，破坏越有效

| 模式 | 名称 | 攻击目标 | 原理 |
|------|------|----------|------|
| **O** | 纹理损失 + 语义损失（基线） | VAE 潜空间 + 去噪器 | 最大化 VAE latent 与目标的距离 + 最大化去噪器预测幅度（语义破坏） |
| **A** | 跨模态对齐破坏 | Joint Attention | 最大化 text→image 注意力权重的熵，使文本到图像的语义注入变得均匀/随机 |
| **B** | 注意力特征偏移 | MMDiT 中间层特征 | 最大化对抗样本与干净样本特征的余弦距离 + 最大化 Gram 矩阵 L1 距离（破坏纹理/风格统计量） |
| **C** | 时序一致性破坏 | Flow Matching 轨迹 | 最大化对抗样本与干净样本速度预测（v-prediction）之间的夹角，使 ODE 去噪轨迹发散 |
| **D** | 模态不平衡 | 双流架构 | 强制图像流特征方差异常增大 + 强制跨模态投影（Q_img, K_txt）正交化，使 Joint Attention 融合产生"排异反应" |

#### 使用方式

```bash
# 模式 O：仅纹理损失（基线）
python code/diff_mist_SD3.py attack.mode='O' attack.g_mode='+' attack.device="cuda:1"

# 模式 A：跨模态对齐破坏
python code/diff_mist_SD3.py attack.mode='A' attack.g_mode='+' attack.device="cuda:1"

# 模式 B：注意力特征偏移
python code/diff_mist_SD3.py attack.mode='B' attack.g_mode='+' attack.device="cuda:1"

# 模式 C：时序一致性破坏
python code/diff_mist_SD3.py attack.mode='C' attack.g_mode='+' attack.device="cuda:1"

# 模式 D：模态不平衡
python code/diff_mist_SD3.py attack.mode='D' attack.g_mode='+' attack.device="cuda:1"

# 调整损失权重：增大 MMDiT 损失权重，减小纹理损失权重
python code/diff_mist_SD3.py attack.mode='A' attack.textual_weight=0.5 attack.mmdit_weight=2.0 attack.device="cuda:1"

# 反向梯度方向
python code/diff_mist_SD3.py attack.mode='O' attack.g_mode='-' attack.device="cuda:1"
python code/diff_mist_SD3.py attack.mode='A' attack.g_mode='-' attack.device="cuda:1"
python code/diff_mist_SD3.py attack.mode='B' attack.g_mode='-' attack.device="cuda:1"
python code/diff_mist_SD3.py attack.mode='C' attack.g_mode='-' attack.device="cuda:1"
python code/diff_mist_SD3.py attack.mode='D' attack.g_mode='-' attack.device="cuda:1"
```

输出文件：
- `[NAME]_attacked.png` — 添加对抗扰动后的图像
- `[NAME]_sdedit_noise_0.1.png` — 噪声水平 0.1 的 SDEdit 去噪结果
- `[NAME]_sdedit_noise_0.3.png` — 噪声水平 0.3 的 SDEdit 去噪结果
- `[NAME]_sdedit_noise_0.5.png` — 噪声水平 0.5 的 SDEdit 去噪结果
- `[NAME]_loss.npy` — PGD 迭代过程中的损失曲线

## 损失曲线可视化

使用 `plot_loss.py` 可视化攻击过程中的损失曲线，帮助分析收敛行为和对比不同攻击模式。

### 基础用法

```bash
# 绘制指定图像的损失曲线（所有模式）
python code/plot_loss.py --root out_sd3 --image suzume

# 绘制所有图像的损失曲线
python code/plot_loss.py --root out_sd3 --all

# 仅绘制指定模式
python code/plot_loss.py --root out_sd3 --image suzume --modes A B C

# 绘制指定梯度模式（g_mode：+ 表示上升，- 表示下降）
python code/plot_loss.py --root out_sd3 --image suzume --gmodes +
python code/plot_loss.py --root out_sd3 --image suzume --gmodes + -
```

### 高级选项

```bash
# 使用移动平均平滑曲线
python code/plot_loss.py --root out_sd3 --image suzume --smooth 5

# 将曲线归一化到 [0, 1] 以便对比
python code/plot_loss.py --root out_sd3 --image suzume --normalize

# 使用对数坐标
python code/plot_loss.py --root out_sd3 --image suzume --log

# 高 DPI 输出用于论文发表
python code/plot_loss.py --root out_sd3 --image suzume --dpi 300

# 绘制组件分解（textual vs MMDiT 损失）
python code/plot_loss.py --root out_sd3 --image suzume --components
```

### 输出文件

脚本会在指定输出目录（默认为 `<root>/figures/`）生成以下图表：

| 文件 | 说明 |
|------|------|
| `{image}_loss_modes.png` | 各模式的总损失对比 |
| `summary_loss_modes.png` | 所有图像的平均损失（均值 ± 标准差） |
| `{image}_loss_components.png` | 每个模式的组件分解（textual / MMDiT / total） |
| `summary_textual_loss.png` | 跨图像平均的纹理损失 |
| `summary_mmdit_loss.png` | 跨图像平均的 MMDiT 损失 |

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--root` | 实验输出根目录 | `out_sd3` |
| `--image` | 指定图像名（不含 `_loss` 后缀） | `None`（需使用 `--all`） |
| `--all` | 绘制根目录下所有图像 | `False` |
| `--modes` | 绘制的模式：O, A, B, C, D | `all` |
| `--gmodes` | 梯度模式：`+`（上升），`-`（下降） | `all` |
| `--output` | 图表输出目录 | `<root>/figures/` |
| `--dpi` | 图表输出 DPI | `150` |
| `--smooth` | 移动平均窗口大小 | `None` |
| `--log` | 使用对数坐标 | `False` |
| `--normalize` | 将曲线归一化到 [0, 1] | `False` |
| `--components` | 绘制 textual/mmdit/total 组件分解 | `False` |

### 示例：完整对比

```bash
# 平滑、归一化后对比所有模式，并绘制组件分解
python code/plot_loss.py \
    --root out_sd3 \
    --all \
    --smooth 5 \
    --normalize \
    --dpi 300 \
    --components
```

这将生成涵盖所有测试图像、所有攻击模式的综合损失可视化图表。

## 项目结构

```
Diff-Protect/
├── code/
│   ├── diff_mist.py          # SD v1.4 攻击启动脚本
│   ├── diff_mist_SD3.py      # SD 3.5 MMDiT 攻击启动脚本
│   ├── attacks.py            # SD v1.4 PGD/SDS 攻击实现
│   ├── attacks_SD3.py        # SD 3.5 MMDiT 攻击 + 4 种损失函数
│   ├── utils.py              # 工具函数（图像读写、指标计算）
│   ├── eval_gen.py           # 生成质量评估
│   ├── eval.py               # 评估指标
│   ├── clip_similarity.py    # 基于 CLIP 的相似度
│   ├── inpaint.py            # 图像修复工具
│   ├── test_bench.py         # 测试基准
│   └── ldm/                  # 潜扩散模型（SD v1.4）
├── configs/
│   ├── attack/
│   │   ├── base.yaml         # SD v1.4 攻击配置
│   │   └── base_sd3.yaml     # SD 3.5 攻击配置
│   └── stable-diffusion/
│       └── v1-inference-attack.yaml
├── test_images/
│   ├── to_protect/           # 待保护图像
│   ├── target/               # 纹理损失的目标图像
│   └── media/                # README 素材
├── ckpt/                     # 模型权重（SD v1.4）
├── out/                      # SD v1.4 输出目录
├── out_sd3/                  # SD 3.5 输出目录
├── env.yml                   # Conda 环境配置（SD v1.4）
├── SD3对抗扰动设计.md         # SD3 对抗扰动设计文档
└── README.md
```


## 技术细节

### SD v1.4 损失函数

- **语义损失 (L_S)**：最大化去噪器的预测误差——将图像表示从扩散模型的语义空间中拉出。形式化：$\max_\delta \mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t', t)\|_2^2]$
- **纹理损失 (L_T)**：将图像在 VAE 潜空间中的表示推向错误目标方向。形式化：$\min_\delta \|\mathcal{E}(y) - \mathcal{E}(x+\delta)\|_2$
- **联合损失 (Mist)**：$\max_\delta (w \cdot L_S - L_T)$，以权重平衡因子联合优化两种损失。

### SD 3.5 MMDiT 损失函数

SD3 的 MMDiT 使用 Joint Attention，文本 token 和图像潜空间 token 拼接后共同进行自注意力计算，扰动可以在模态间传播。五种攻击模式如下：

0. **模式 O — 纹理损失 + 语义损失（基线）**：结合 VAE 纹理损失 $\max_\delta \|\mathcal{E}(y) - \mathcal{E}(x+\delta)\|_2$（将潜空间表示推离目标）与基于去噪器预测幅度的语义损失。语义损失实现为 $\max_\delta \|v_{\text{pred}}(x+\delta)\|_2$，更大的预测幅度意味着更强的语义破坏。两种损失均通过梯度上升（g_mode='+'）进行最大化。作为对比基线，用于评估 MMDiT 专用损失的额外增益。

1. **损失 A — 跨模态对齐破坏**：在 Joint Attention 中，text→image 注意力块（$\text{attn}[:N_i, N_i:]$）反映了语义注入强度。最大化其注意力熵使语义引导变得均匀/随机，导致生成结果偏离文本提示。

2. **损失 B — 注意力特征偏移**：Joint Attention 块输出的融合特征定义了图像在 MMDiT 特征流形中的表示。最大化对抗样本与干净样本特征之间的余弦距离（在高维 DiT 特征空间中方向比模长更重要，故选用余弦距离而非 L2），将表示推离干净流形。同时，最大化对抗样本与干净样本 Gram 矩阵之间的 L1 距离，破坏纹理/风格统计量。

3. **损失 C — 时序一致性破坏**：SD3 使用基于 Flow Matching 的 v-预测。通过最大化对抗样本与干净样本速度预测之间的夹角（$\min \cos(v_{adv}, v_{clean})$），使去噪轨迹发散，后续步骤的非线性将进一步放大扰动。

4. **损失 D — 模态不平衡**：MMDiT 拥有独立的文本流和图像流，仅在 Joint Attention 中交互。强制图像流特征的方差异常增大，同时将跨模态投影（Q_img, K_txt）推向正交，在联合融合中产生"排异反应"。

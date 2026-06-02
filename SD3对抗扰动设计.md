# SD3对抗扰动设计

## 目标

在/data/home/Boheng/cyd/Diff-Protect项目的基础上，完成针对SD3的对抗扰动生成，使用git做好版本管理。

思路为对于Stable Diffusion 3.5，参考mist的TEXTUAL LOSS，把SEMANTIC LOSS换成以下几种对于MMDiT攻击的loss，通过参数选择对应的MMDiT攻击的loss，JOINT LOSS=TEXTUAL LOSS + lambda*MMDiT LOSS。

## 实验环境

Stable Diffusion 3.5的推理pipeline参考/data/home/Boheng/cyd/test.py
python环境使用conda的dino环境
gpu尽可能使用cuda:1

## 针对vae的loss设计参考TEXTUAL LOSS

### Mist: Towards Improved Adversarial Examples for Diffusion Models

#### SEMANTIC LOSS 
> 让去噪器的预测误差最大

$$\begin{aligned} & \delta:=arg min _{\delta} \mathbb{E}_{x_{1: T}' \sim u\left(x_{1: T}'\right)} \mathcal{L}_{D M}\left(x', \theta\right), \\ & where x \sim q(x), x'=x+\delta . \end{aligned}$$

$$\mathbb{E}_{t, \epsilon \sim \mathcal{N}(0,1)} \mathbb{E}_{x_{t}' \sim u\left(x_{t}'\right)}\left[\left\| \epsilon-\epsilon_{\theta}\left(x_{t}', t\right)\right\| _{2}^{2}\right]$$


其中，期望通过蒙特卡洛方法估计。直观地说，这种损失试图将图像x的表示从扩散模型的语义空间中拉出来。我们的经验观察表明，这种损失的最大化会导致基于对抗样本生成的图像中出现混乱内容。因此，我们将这种损失称为语义损失。

#### TEXTUAL LOSS 
> 把图在潜空间中的表示“推”向一个错误的方向

$$\begin{aligned} \delta: & =arg min _{\delta} \mathcal{L}_{\mathcal{E}}(x, \delta, y) \\ & =arg min _{\delta}\| \mathcal{E}(y)-\mathcal{E}(x+\delta)\| _{2}, \end{aligned}$$



其中，ε表示潜在扩散模型的图像编码器，x代表输入图像，y是给定的目标图像。为了优化这一损失，我们采用了投影梯度下降（PGD）攻击。由此产生的扰动具有类似于背景上嵌入水印的特征。因此，我们将这种损失称为纹理损失。

####  JOINT LOSS


$$\begin{gathered} \delta:=arg max _{\delta}\left(w \mathbb{E}_{x_{1: T}' \sim u\left(x_{1: T}'\right)} \mathcal{L}_{D M}\left(x', \theta\right)-\mathcal{L}_{\mathcal{E}}(x, \delta, y)\right), \\ where x \sim q(x), x'=x+\delta . \end{gathered}$$



$$\begin{array} {c}\mathbb {E}_{t,\epsilon \sim \mathcal {N}(0,1)}\mathbb {E}_{x_{t}^{\prime }\sim u\left(x_{t}^{\prime }\right) }\left[ w\left\| \epsilon -\epsilon _{\theta }\left( x_{t}', t\right) \| _{2}^{2}\right. \right. \\ \left.-\| \mathcal {E}(y)-\mathcal {E}(x+\delta )\| _{2}\right] \end{array}$$

## 针对MMDiT的loss设计

针对Stable Diffusion 3 (SD3) 中 MMDiT (Multimodal DiT) 架构设计对抗扰动（Adversarial Perturbation）的 Loss，需要充分考虑 MMDiT 的**双流（Dual-Stream）机制**、**联合注意力（Joint Attention）**以及**文本-图像跨模态交互**的特性。

与传统的 UNet 不同，MMDiT 中文本 token 和图像 latent token 在 Joint Attention 层是拼接在一起进行自注意力的，这意味着扰动可以在模态间传播。

以下是设计的几种针对 SD3 MMDiT Attention 的对抗 Loss 策略，按攻击目标和原理分类：

### 1. 核心前置知识：MMDiT Attention 结构
在设计 Loss 前，需明确攻击点。SD3 MMDiT 的 Joint Attention 计算如下：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{[Q_t; Q_i][K_t; K_i]^T}{\sqrt{d}}\right) [V_t; V_i] $$
其中 $t$ 代表 text tokens，$i$ 代表 image latent tokens。对抗扰动 $\delta$ 可以加在输入端（Latent/Embedding），也可以直接针对 Attention Map 或 Feature Map 进行优化。

---

### 2. 推荐的对抗 Loss 设计方案

#### A. 跨模态对齐破坏 Loss (Cross-Modal Alignment Disruption)
**目标**：利用 MMDiT 的联合注意力特性，破坏文本条件对图像生成的语义引导，使生成结果偏离 prompt。
**原理**：在 Joint Attention 中，Text-to-Image 的注意力权重反映了语义注入强度。最大化该权重的熵或最小化特定关键 token 的注意力峰值。

```python
# 伪代码示例
def cross_modal_disruption_loss(attn_map, text_token_mask):
    """
    attn_map: [B, H, N_text + N_img, N_text + N_img]
    text_token_mask: 标记哪些是text token的位置
    """
    # 提取 Text -> Image 的注意力块
    # 假设前 N_t 个是 text，后 N_i 个是 image
    N_t = text_token_mask.sum()
    text_to_img_attn = attn_map[:, :, N_t:, :N_t]  # [B, H, N_img, N_text]
    
    # 策略1: 最大化注意力熵 (使语义注入变得均匀/随机)
    entropy_loss = -(text_to_img_attn * torch.log(text_to_img_attn + 1e-8)).sum(dim=-1).mean()
    
    # 策略2: 最小化关键Token的最大注意力权重 (削弱核心语义)
    # key_token_indices 由CLIP/T5 tokenizer获取
    max_attn_suppression = -text_to_img_attn[:, :, :, key_token_indices].max(dim=-1).values.mean()
    
    return entropy_loss + lambda_ * max_attn_suppression
```

#### B. Attention Feature 偏移 Loss (Feature Space Adversarial Loss)
**目标**：让受扰动后的 MMDiT 中间层特征远离干净样本的特征流形，导致解码器输出噪声或错误内容。
**原理**：MMDiT 的双流 Block 会不断融合 text/image feature。在融合后的特征空间做对抗最有效。

```python
def feature_divergence_loss(feat_adv, feat_clean):
    """
    针对 MMDiT Joint Block 输出的 combined feature
    """
    # 使用余弦距离而非L2，因为DiT特征在高维空间方向比模长更重要
    cos_sim = F.cosine_similarity(feat_adv.flatten(1), feat_clean.flatten(1), dim=1)
    
    # 最小化相似度 = 最大化对抗性
    loss = cos_sim.mean()
    
    # 可选: 加入 Gram Matrix 匹配破坏 (破坏纹理/风格统计量)
    gram_adv = gram_matrix(feat_adv)
    gram_clean = gram_matrix(feat_clean)
    style_loss = F.l1_loss(gram_adv, gram_clean)
    
    return loss + lambda_style * style_loss
```

#### C. 时序一致性破坏 Loss (Temporal/Step Consistency Break)
**目标**：SD3 是多步去噪过程。设计一个 Loss 使得扰动在当前步看起来无害，但在后续步被 MMDiT 的非线性放大。
**原理**：利用 ODE/SDE 轨迹的敏感性。

```python
def trajectory_divergence_loss(model, x_adv, x_clean, t, cond):
    """
    预测下一步的去噪方向差异
    """
    with torch.no_grad():
        v_clean = model(x_clean, t, **cond)  # SD3 使用 v-prediction
    
    v_adv = model(x_adv, t, **cond)
    
    # 让对抗样本的速度场指向错误方向
    # 注意: 这里不是简单的MSE，而是要结合scheduler的方向
    velocity_loss = -F.cosine_similarity(v_adv, v_clean, dim=[1,2,3]).mean()
    
    return velocity_loss
```

#### D. 针对 MMDiT 特有结构的 Modality Imbalance Loss
**目标**：MMDiT 有独立的 Text Stream 和 Image Stream，只在 Joint Attention 交互。打破两个流的内部自洽性。
**原理**：让 Image Stream 的 Self-Attention 产生高频噪声模式，同时保持 Text Stream 不变，使 Joint Attention 的融合产生"排异反应"。

```python
def modality_imbalance_loss(img_stream_feat, txt_stream_feat):
    # 强制 Image Stream 特征的方差异常增大
    img_var = img_stream_feat.var(dim=[2,3])
    var_explosion_loss = -img_var.mean()  # 最大化方差
    
    # 或者: 让 Image Stream 和 Text Stream 在 Joint Attention 前的 
    # Q/K 投影空间的夹角正交化
    q_img = W_q_img(img_stream_feat)
    k_txt = W_k_txt(txt_stream_feat)
    ortho_loss = (q_img @ k_txt.transpose(-2,-1)).abs().mean()
    
    return var_explosion_loss + lambda_ortho * ortho_loss
```

### 4. 关键注意事项

1.  **梯度传播路径**：确保 Loss 能通过 MMDiT 的 Joint Attention 反传到 Latent。
2.  **v-prediction 适配**：SD3 默认使用 v-prediction 而非 epsilon-prediction。所有基于去噪方向的 Loss 必须使用 `v_target` 而非 `eps_target`。
3.  **多分辨率感知**：MMDiT 支持多种分辨率，Attention Map 的尺寸随 latent size 变化

## 关于论文

1.前面的照着写，介绍mmdit
2.介绍四种方法
3.baseline选mist，Lt+Ls/Lt+ABCD做对比，说明Lt+Ls比我们差就行（解释原因：结构不一样，直接用噪声预测的结果作为优化的依据不能很好地优化）
4.图片（结构图/之前方法的图）

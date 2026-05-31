# Diff-Protect

Adversarial perturbation framework for protecting images against diffusion model-based generation.

![](test_images/media/teaser.png)

## Overview

Diff-Protect generates imperceptible adversarial perturbations that, when added to images, disrupt the generation process of diffusion models. The framework currently supports two model families:

| Feature | SD v1.4 (`diff_mist.py`) | SD 3.5 (`diff_mist_SD3.py`) |
|---------|--------------------------|-----------------------------|
| Architecture | UNet + ε-prediction | MMDiT + v-prediction (flow matching) |
| Text Encoder | CLIP × 1 | CLIP-L + CLIP-G + T5 |
| Attention | Cross-Attention | Joint Attention (dual-stream) |
| Attack Losses | Semantic / Textual / Joint | Textual + 4 MMDiT-specific losses |
| Conda Env | `mist` | `dino` |

---

## Quick Setup

### SD v1.4 Environment

```bash
conda env create -f env.yml
conda activate mist
pip install --force-reinstall pillow
```

Download the [Stable Diffusion v1.4 checkpoint](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original):

```bash
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir ckpt
mv sd-v1-4.ckpt ckpt/model.ckpt
```

### SD 3.5 Environment

Use the `dino` conda environment with PyTorch 2.0+ and diffusers:

```bash
conda activate dino
# diffusers, transformers, accelerate should already be installed
```

The SD 3.5 model weights are automatically downloaded from HuggingFace on first run (`stabilityai/stable-diffusion-3.5-medium`).

---

## Run the Protection

### SD v1.4 Attacks

Configs can be set in `configs/attack/base.yaml`:

```yaml
attack:
    epsilon: 16           # l_inf budget (pixel units)
    steps: 100            # attack iterations
    input_size: 512       # image resolution
    mode: sds             # attack mode
    img_path: test_images/to_protect
    output_path: out/
    alpha: 1              # step size
    g_mode: "+"           # gradient direction
    device: 0             # GPU id
```

Available modes:

| Mode | Loss | Command |
|------|------|---------|
| **AdvDM** | Semantic loss only | `python code/diff_mist.py attack.mode='advdm' attack.g_mode='+' attack.device="cuda:2"` |
| **PhotoGuard** | Textual loss only | `python code/diff_mist.py attack.mode='texture_only' attack.g_mode='+' attack.device="cuda:2"` |
| **Mist** | Textural + Semantic (joint) | `python code/diff_mist.py attack.mode='mist' attack.g_mode='+' attack.device="cuda:2"` |
| **SDS(+)** | SDS-accelerated, ascend gradient | `python code/diff_mist.py attack.mode='sds' attack.g_mode='+' attack.device="cuda:2"` |
| **SDS(-)** | SDS-accelerated, descend gradient | `python code/diff_mist.py attack.mode='sds' attack.g_mode='-' attack.device="cuda:2"` |
| **SDST(-)** | SDS + targeted textual loss | `python code/diff_mist.py attack.mode='sds' attack.g_mode='-' attack.using_target=True attack.device="cuda:2"` |

The output includes:
- `[NAME]_attacked.png` — adversarially perturbed image
- `[NAME]_onestep.png` — one-step denoising prediction
- `[NAME]_multistep.png` — multi-step SDEdit result

<img src="out/advdm_eps16_steps100_gmode+/to_protect/suzume_attacked.png" alt="drawing" width="200"/>  <img src="out/mist_eps16_steps100_gmode+/to_protect/suzume_attacked.png" alt="drawing" width="200"/> <img src="out/sds_eps16_steps100_gmode-/to_protect/suzume_attacked.png" alt="drawing" width="200"/>

*[From left to right]: AdvDM, Mist, SDS(-), using eps=16. SDS-version is much more effective.*

---

### SD 3.5 MMDiT Attacks

Configs can be set in `configs/attack/base_sd3.yaml`:

```yaml
attack:
    epsilon: 16           # l_inf budget (pixel units)
    steps: 100            # attack iterations
    input_size: 1024      # image resolution
    mode: A               # MMDiT attack mode: A/B/C/D
    img_path: test_images/to_protect
    output_path: out_sd3/
    alpha: 1              # step size
    g_mode: "+"           # gradient direction: '+' or '-'
    textual_weight: 1.0   # weight for textual loss (VAE latent push)
    mmdit_weight: 1.0     # weight for MMDiT-specific loss
    device: "cuda:2"      # GPU device
    model_name: "stabilityai/stable-diffusion-3.5-medium"
```

#### MMDiT Attack Modes

The joint loss follows the design: **JOINT LOSS = TEXTUAL LOSS + λ · MMDiT LOSS**, where TEXTUAL LOSS pushes the VAE latent toward a wrong direction (same as the original Mist), and the MMDiT LOSS exploits the architectural specifics of SD3's Multimodal DiT.

| Mode | Name | Target | Principle |
|------|------|--------|-----------|
| **A** | Cross-Modal Alignment Disruption | Joint Attention | Maximize entropy of text→image attention weights, making semantic injection from text uniform/random |
| **B** | Attention Feature Shift | MMDiT intermediate features | Maximize cosine distance between adversarial and clean features + disrupt Gram matrix (style statistics) |
| **C** | Temporal Consistency Break | Flow matching trajectory | Maximize angle between adversarial and clean velocity predictions (v-prediction), causing ODE trajectory to diverge |
| **D** | Modality Imbalance | Dual-stream architecture | Force image stream feature variance explosion + enforce cross-modal (Q_img, K_txt) orthogonality |

#### Usage

```bash
# Mode A: Cross-Modal Alignment Disruption
python code/diff_mist_SD3.py attack.mode='A' attack.g_mode='+' attack.device="cuda:2"

# Mode B: Attention Feature Shift
python code/diff_mist_SD3.py attack.mode='B' attack.g_mode='+' attack.device="cuda:2"

# Mode C: Temporal Consistency Break
python code/diff_mist_SD3.py attack.mode='C' attack.g_mode='+' attack.device="cuda:2"

# Mode D: Modality Imbalance
python code/diff_mist_SD3.py attack.mode='D' attack.g_mode='+' attack.device="cuda:2"

# Adjust loss weights: emphasize MMDiT loss over textual loss
python code/diff_mist_SD3.py attack.mode='A' attack.textual_weight=0.5 attack.mmdit_weight=2.0 attack.device="cuda:2"

# Reverse gradient direction
python code/diff_mist_SD3.py attack.mode='A' attack.g_mode='-' attack.device="cuda:1"
```

The output includes:
- `[NAME]_attacked.png` — adversarially perturbed image
- `[NAME]_sdedit_noise_0.1.png` — SDEdit denoising from noise level 0.1
- `[NAME]_sdedit_noise_0.3.png` — SDEdit denoising from noise level 0.3
- `[NAME]_sdedit_noise_0.5.png` — SDEdit denoising from noise level 0.5
- `[NAME]_loss.npy` — loss curve over PGD iterations

---

## Project Structure

```
Diff-Protect/
├── code/
│   ├── diff_mist.py          # SD v1.4 attack launch script
│   ├── diff_mist_SD3.py      # SD 3.5 MMDiT attack launch script
│   ├── attacks.py            # SD v1.4 PGD/SDS attack implementations
│   ├── attacks_SD3.py        # SD 3.5 MMDiT attack + 4 loss functions
│   ├── utils.py              # Utility functions (image I/O, metrics)
│   ├── eval_gen.py           # Generation quality evaluation
│   ├── eval.py               # Evaluation metrics
│   ├── clip_similarity.py    # CLIP-based similarity
│   ├── inpaint.py            # Inpainting utilities
│   ├── test_bench.py         # Testing benchmark
│   └── ldm/                  # Latent Diffusion Model (SD v1.4)
├── configs/
│   ├── attack/
│   │   ├── base.yaml         # SD v1.4 attack config
│   │   └── base_sd3.yaml     # SD 3.5 attack config
│   └── stable-diffusion/
│       └── v1-inference-attack.yaml
├── test_images/
│   ├── to_protect/           # Images to protect
│   ├── target/               # Target images for textual loss
│   └── media/                # README assets
├── ckpt/                     # Model checkpoints (SD v1.4)
├── out/                      # SD v1.4 output directory
├── out_sd3/                  # SD 3.5 output directory
├── env.yml                   # Conda environment (SD v1.4)
├── SD3对抗扰动设计.md         # SD3 adversarial design document
└── README.md
```

---

## Technical Details

### SD v1.4 Loss Functions

- **Semantic Loss (L_S)**: Maximizes the denoiser prediction error — pulls the image representation out of the diffusion model's semantic space. Formally: $\max_\delta \mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t', t)\|_2^2]$
- **Textural Loss (L_T)**: Pushes the image's latent representation toward a wrong target in VAE latent space. Formally: $\min_\delta \|\mathcal{E}(y) - \mathcal{E}(x+\delta)\|_2$
- **Joint Loss (Mist)**: $\max_\delta (w \cdot L_S - L_T)$, combining both losses with a balancing weight.

### SD 3.5 MMDiT Loss Functions

SD3's MMDiT uses Joint Attention where text tokens and image latent tokens are concatenated for self-attention, enabling cross-modal perturbation propagation. The four MMDiT losses exploit different aspects of this architecture:

1. **Loss A — Cross-Modal Alignment Disruption**: In Joint Attention, the text→image attention block ($\text{attn}[:N_i, N_i:]$) reflects how semantics are injected. Maximizing its entropy makes the semantic guidance uniform/random, causing generation to lose prompt alignment.

2. **Loss B — Attention Feature Shift**: The combined features after Joint Attention blocks define the image's representation in MMDiT's feature manifold. Maximizing cosine distance between adversarial and clean features (using cosine over L2, as direction matters more than magnitude in high-dimensional DiT features) pushes the representation off the clean manifold.

3. **Loss C — Temporal Consistency Break**: SD3 uses flow matching with v-prediction. By maximizing the angle between adversarial and clean velocity predictions ($\min \cos(v_{adv}, v_{clean})$), the denoising trajectory diverges, and the non-linearity of subsequent steps amplifies the perturbation.

4. **Loss D — Modality Imbalance**: MMDiT has separate text and image streams that interact only through Joint Attention. Forcing the image stream's feature variance to explode while pushing cross-modal projections (Q_img, K_txt) toward orthogonality creates a "rejection" effect in the joint fusion.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{xue2023mist,
  title={Mist: Towards Improved Adversarial Examples for Diffusion Models},
  author={Xue, Haotian and et al.},
  year={2023}
}
```

# @ Boheng 2026
# SD3 Adversarial Perturbation Generator
# Based on Diff-Protect framework, adapted for Stable Diffusion 3.5 (MMDiT architecture)
#
# Usage:
#   python code/diff_mist_SD3.py attack.mode='A' attack.g_mode='+' attack.device="cuda:2"
#   python code/diff_mist_SD3.py attack.mode='B' attack.g_mode='-' attack.device="cuda:1"
#   python code/diff_mist_SD3.py attack.mode='C' attack.g_mode='+' attack.device="cuda:2"
#   python code/diff_mist_SD3.py attack.mode='D' attack.g_mode='-' attack.device="cuda:1"
#
# Modes:
#   O: Textual Loss only (仅纹理损失，作为对比基线)
#   A: Cross-Modal Alignment Disruption (破坏跨模态对齐)
#   B: Attention Feature Shift (特征偏移)
#   C: Temporal Consistency Break (时序一致性破坏)
#   D: Modality Imbalance (模态不平衡)

import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import glob
import hydra

from utils import mp, si, cprint
from attacks_SD3 import SD3_Linf_PGD, AttentionMapHook, register_feature_hooks, restore_processors

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')


def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    """Load image from path and resize to input_size."""
    img = Image.open(image_path).resize((input_size, input_size), resample=PIL.Image.BICUBIC)
    return img


class identity_loss(nn.Module):
    """Identity loss for advertorch compatibility."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class SD3_target_model(nn.Module):
    """Wrapper model for SD3 that computes textual + MMDiT adversarial losses.

    This follows the same pattern as the original target_model in diff_mist.py,
    but adapted for SD3's pipeline architecture (MMDiT + Flow Matching).

    The forward() method is called by PGD to get the loss for gradient computation.
    """

    def __init__(self, pipe, condition: str, mode: str = 'A', g_mode: str = '+',
                 textual_weight: float = 1.0, mmdit_weight: float = 1.0,
                 device: str = "cuda:0"):
        """
        Args:
            pipe: StableDiffusion3Pipeline instance
            condition: text prompt for conditioning
            mode: MMDiT attack mode ('A', 'B', 'C', 'D')
            g_mode: gradient direction ('+' for maximize, '-' for minimize)
            textual_weight: weight for textual loss (VAE latent push)
            mmdit_weight: weight for MMDiT-specific loss
            device: target device
        """
        super().__init__()
        self.pipe = pipe
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")
        self.mode = mode
        self.g_mode = g_mode
        self.textual_weight = textual_weight
        self.mmdit_weight = mmdit_weight

        # Pre-compute and cache text embeddings
        self._encode_prompt(condition, device)

        # Target image for textual loss
        self.target_info = None

        # Clean features for loss B
        self.feat_clean_list = None

        print(f'[SD3_target_model] mode={mode}, g_mode={g_mode}, '
              f'textual_weight={textual_weight}, mmdit_weight={mmdit_weight}')

    def _encode_prompt(self, prompt, device):
        """Pre-compute text embeddings for SD3 (3 encoders: CLIP-L, CLIP-G, T5)."""
        print(f'Encoding prompt: "{prompt}"')
        with torch.no_grad():
            (
                self.prompt_embeds,
                self.negative_prompt_embeds,
                self.pooled_prompt_embeds,
                self.negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        print(f'  prompt_embeds shape: {self.prompt_embeds.shape}')
        print(f'  pooled_prompt_embeds shape: {self.pooled_prompt_embeds.shape}')

    def compute_clean_features(self, X_clean):
        """Pre-compute clean features through MMDiT for loss B comparison."""
        from attacks_SD3 import AttentionMapHook, register_feature_hooks

        hook = AttentionMapHook()
        transformer = self.pipe.transformer
        register_feature_hooks(transformer, hook, capture_attn=False)

        with torch.no_grad():
            z_clean = self.pipe.vae.encode(X_clean.to(self.pipe.vae.dtype)).latent_dist.mean
            z_clean = (z_clean - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            z_clean = z_clean.to(transformer.dtype)

            timestep = torch.tensor([0.5], device=X_clean.device, dtype=transformer.dtype)
            _ = transformer(
                hidden_states=z_clean,
                timestep=timestep,
                encoder_hidden_states=self.prompt_embeds.to(transformer.dtype),
                pooled_projections=self.pooled_prompt_embeds.to(transformer.dtype),
                return_dict=False,
            )

        self.feat_clean_list = [f.clone() for f in hook.img_stream_feats]
        hook.remove()
        restore_processors(transformer)

    def get_components(self, x):
        """Compute VAE latent encoding and textual loss.

        Returns:
            z_x: latent encoding of x
            textual_loss: MSE distance between z_x and z_target in latent space
        """
        z_x = self.pipe.vae.encode(x.to(self.pipe.vae.dtype)).latent_dist.mean
        z_x = (z_x - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        z_x = z_x.to(x.dtype)

        if self.target_info is not None:
            with torch.no_grad():
                z_y = self.pipe.vae.encode(self.target_info.to(self.pipe.vae.dtype)).latent_dist.mean
                z_y = (z_y - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
                z_y = z_y.to(x.dtype).detach()
            textual_loss = self.fn(z_x, z_y)
        else:
            textual_loss = torch.tensor(0.0, device=x.device)

        return z_x, textual_loss

    def forward(self, x):
        """Compute the joint loss for PGD attack.

        Mode O: Textual Loss only (baseline, no MMDiT forward pass)
        Modes A-D: JOINT LOSS = textual_weight * TEXTUAL LOSS + mmdit_weight * MMDiT LOSS

        The returned loss is what PGD will minimize. For g_mode='+', we negate
        to achieve maximization.
        """
        from attacks_SD3 import (
            cross_modal_disruption_loss,
            feature_divergence_loss,
            trajectory_divergence_loss,
            modality_imbalance_loss,
            AttentionMapHook,
            register_feature_hooks,
        )

        device = x.device
        g_dir = 1. if self.g_mode == '+' else -1.

        # ---- Textual Loss (VAE latent push) ----
        z_x, textual_loss = self.get_components(x)

        # ---- Mode O: Textual Loss only (baseline) ----
        if self.mode == 'O':
            return textual_loss * g_dir

        # ---- MMDiT Loss ----
        hook = AttentionMapHook()
        transformer = self.pipe.transformer
        need_attn = (self.mode == 'A')
        register_feature_hooks(transformer, hook, capture_attn=need_attn)

        # Sample random timestep for flow matching
        t_val = torch.rand(1, device=device).item()
        timestep = torch.tensor([t_val], device=device, dtype=transformer.dtype)

        # Forward through MMDiT
        v_pred = transformer(
            hidden_states=z_x.to(transformer.dtype),
            timestep=timestep,
            encoder_hidden_states=self.prompt_embeds.to(transformer.dtype),
            pooled_projections=self.pooled_prompt_embeds.to(transformer.dtype),
            return_dict=False,
        )[0]

        mmdit_loss = torch.tensor(0.0, device=device)

        if self.mode == 'A':
            # Cross-Modal Alignment Disruption
            if len(hook.attn_maps) > 0:
                # Estimate N_img from hidden states
                N_img = hook.hidden_states_list[0].shape[1] if len(hook.hidden_states_list) > 0 else 1024
                mmdit_loss = cross_modal_disruption_loss(hook.attn_maps, N_img)

        elif self.mode == 'B':
            # Feature Shift / Divergence
            if len(hook.img_stream_feats) > 0 and self.feat_clean_list is not None:
                mmdit_loss = feature_divergence_loss(hook.img_stream_feats, self.feat_clean_list)

        elif self.mode == 'C':
            # Trajectory Divergence
            with torch.no_grad():
                z_clean = self.pipe.vae.encode(
                    (x - x.grad if x.grad is not None else x).to(self.pipe.vae.dtype)
                ).latent_dist.mean if self.target_info is None else self.pipe.vae.encode(
                    self.target_info.to(self.pipe.vae.dtype)
                ).latent_dist.mean
                # Use the original clean image stored separately
                if hasattr(self, '_clean_x') and self._clean_x is not None:
                    z_clean = self.pipe.vae.encode(self._clean_x.to(self.pipe.vae.dtype)).latent_dist.mean
                    z_clean = (z_clean - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
                    z_clean = z_clean.to(transformer.dtype).detach()
                    v_clean = transformer(
                        hidden_states=z_clean,
                        timestep=timestep,
                        encoder_hidden_states=self.prompt_embeds.to(transformer.dtype),
                        pooled_projections=self.pooled_prompt_embeds.to(transformer.dtype),
                        return_dict=False,
                    )[0]
                    mmdit_loss = trajectory_divergence_loss(v_pred, v_clean)

        elif self.mode == 'D':
            # Modality Imbalance
            if len(hook.img_stream_feats) > 0 and len(hook.txt_stream_feats) > 0:
                mmdit_loss = modality_imbalance_loss(hook.img_stream_feats, hook.txt_stream_feats)

        # Clean up hooks
        hook.remove()
        restore_processors(transformer)

        # ---- Joint Loss ----
        # Following mist design: JOINT LOSS = TEXTUAL LOSS + lambda * MMDiT LOSS
        # g_dir controls gradient direction for PGD optimization
        joint_loss = self.textual_weight * textual_loss * g_dir + self.mmdit_weight * mmdit_loss * g_dir

        return joint_loss


def init(epsilon: int = 16, steps: int = 100, alpha: int = 1,
         input_size: int = 512, mode: str = 'A', g_mode: str = '+',
         device: str = "cuda:0", input_prompt: str = 'a photo',
         textual_weight: float = 1.0, mmdit_weight: float = 1.0,
         model_name: str = "stabilityai/stable-diffusion-3.5-medium"):
    """Initialize SD3 pipeline and attack configuration.

    Args:
        epsilon: L_inf perturbation budget (in pixel units, divided by 255 internally)
        steps: number of PGD iterations
        alpha: step size per PGD iteration
        input_size: image resolution (must be divisible by vae_scale_factor * patch_size)
        mode: MMDiT attack mode ('A', 'B', 'C', 'D')
        g_mode: gradient direction ('+' for maximize, '-' for minimize)
        device: target device (e.g., "cuda:2")
        input_prompt: text prompt for conditioning
        textual_weight: weight for textual loss component
        mmdit_weight: weight for MMDiT loss component
        model_name: ModelScope model ID for SD3

    Returns:
        dict with 'net', 'fn', 'parameters'
    """
    from modelscope import StableDiffusion3Pipeline

    print(f"Loading SD3 model from ModelScope: {model_name}")
    print(f"Target device: {device}")

    # Load SD3 pipeline from ModelScope (uses local cache)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)

    # Disable gradient checkpointing for attack (need gradients)
    pipe.transformer.gradient_checkpointing = False

    fn = identity_loss()

    # Create target model wrapper
    net = SD3_target_model(
        pipe, condition=input_prompt,
        mode=mode, g_mode=g_mode,
        textual_weight=textual_weight,
        mmdit_weight=mmdit_weight,
        device=device,
    )
    net.eval()

    # Attack parameters (mapping to [-1, 1] pixel range)
    parameters = {
        'epsilon': epsilon / 255.0 * (1 - (-1)),  # map to [-1,1] range
        'alpha': alpha / 255.0 * (1 - (-1)),
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'g_mode': g_mode,
        'textual_weight': textual_weight,
        'mmdit_weight': mmdit_weight,
    }

    return {'net': net, 'fn': fn, 'parameters': parameters}


def infer(img: PIL.Image.Image, config, tar_img: PIL.Image.Image = None,
          device: str = "cuda:0") -> tuple:
    """Generate adversarial perturbation for a single image.

    Args:
        img: input image to protect
        config: attack configuration dict from init()
        tar_img: target image for textual loss
        device: computation device

    Returns:
        (adversarial_image, sdedit_results) tuple
    """
    net = config['net']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters['epsilon']
    alpha = parameters['alpha']
    steps = parameters['steps']
    input_size = parameters['input_size']
    g_mode = parameters['g_mode']
    textual_weight = parameters['textual_weight']
    mmdit_weight = parameters['mmdit_weight']

    cprint(f'epsilon: {epsilon}', 'y')
    cprint(f'mode: {mode}, g_mode: {g_mode}', 'y')

    # Preprocess image to [-1, 1] range
    img = img.convert('RGB')
    img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
    img_np = img_np[:, :, :3]

    if tar_img is not None:
        tar_img = tar_img.convert('RGB')
        tar_np = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
        tar_np = tar_np[:, :, :3]

    trans = transforms.Compose([transforms.ToTensor()])

    data_source = torch.zeros([1, 3, input_size, input_size]).to(device)
    data_source[0] = trans(img_np).to(device)

    target_info = torch.zeros([1, 3, input_size, input_size]).to(device)
    if tar_img is not None:
        target_info[0] = trans(tar_np).to(device)
    else:
        target_info = None

    net.target_info = target_info
    net.mode = mode

    # For mode C (trajectory divergence), store clean image reference
    net._clean_x = data_source.clone().detach()

    # For mode B (feature divergence), pre-compute clean features
    if mode == 'B':
        cprint('Pre-computing clean features for loss B...', 'y')
        net.compute_clean_features(data_source)

    # Print initial loss
    with torch.no_grad():
        init_loss = net(data_source)
    cprint(f'Initial loss: {init_loss.item():.5f}', 'y')

    # ---- Run PGD Attack ----
    time_start_attack = time.time()

    attack = SD3_Linf_PGD(
        net=net,
        fn=config['fn'],
        epsilon=epsilon,
        steps=steps,
        eps_iter=alpha,
        clip_min=-1.0,
        clip_max=1.0,
        targeted=True,
        g_mode=g_mode,
        mmdit_mode=mode,
        capture_attn=(mode == 'A'),
        textual_weight=textual_weight,
        mmdit_weight=mmdit_weight,
    )

    attack_output, loss_all = attack.pgd_sd3(
        X=data_source,
        target_image=target_info,
    )

    print(f'Max perturbation: {torch.abs(attack_output - data_source).max().item():.6f}')
    print(f'Attack takes: {time.time() - time_start_attack:.2f}s')

    # ---- SD3 SDEdit evaluation ----
    sdedit_results = _sdedit_sd3(net, attack_output, device)

    # ---- Save adversarial image ----
    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()

    return grid_adv, sdedit_results, loss_all


def _sdedit_sd3(net, x_adv, device, t_list=None, num_steps=28):
    """Run SD3 denoising from various noise levels (SDEdit-style evaluation).

    This tests how the adversarial perturbation disrupts SD3's generation
    by adding noise and then denoising.

    Args:
        net: SD3_target_model
        x_adv: adversarial image tensor [1, 3, H, W] in [-1, 1]
        device: computation device
        t_list: list of noise levels (0 to 1) for SDEdit
        num_steps: number of denoising steps

    Returns:
        dict with denoised images at various noise levels
    """
    if t_list is None:
        t_list = [0.1, 0.3, 0.5]

    pipe = net.pipe
    results = {}

    # Encode adversarial image to latent
    with torch.no_grad():
        z_adv = pipe.vae.encode(x_adv.to(pipe.vae.dtype)).latent_dist.mean
        z_adv = (z_adv - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_adv = z_adv.to(pipe.transformer.dtype)

    for noise_level in t_list:
        with torch.no_grad():
            # Add noise at the specified level
            noise = torch.randn_like(z_adv)
            # In flow matching: z_t = (1-t) * z_0 + t * noise
            # sigma = noise_level corresponds to adding noise proportionally
            sigma = noise_level
            z_noisy = (1 - sigma) * z_adv + sigma * noise

            # Denoise from this noise level
            # We use the scheduler's timesteps but start from the noise level
            from diffusers import FlowMatchEulerDiscreteScheduler

            # Create a temporary scheduler with fewer steps for partial denoising
            result = _denoise_from_noise_level(
                pipe, z_noisy, net.prompt_embeds, net.pooled_prompt_embeds,
                noise_level, num_steps, device
            )

            results[f'noise_{noise_level}'] = result

    return results


def _denoise_from_noise_level(pipe, z_noisy, prompt_embeds, pooled_embeds,
                              start_noise_level, num_steps, device):
    """Denoise from a given noise level using SD3's flow matching scheduler.

    Args:
        pipe: SD3 pipeline
        z_noisy: noisy latent [1, C, H, W]
        prompt_embeds: text prompt embeddings
        pooled_embeds: pooled text embeddings
        start_noise_level: the noise level (0-1) where denoising starts
        num_steps: total inference steps
        device: computation device

    Returns:
        denoised PIL Image
    """
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Find the starting timestep closest to start_noise_level
    # In flow matching, sigmas correspond to noise levels
    sigmas = pipe.scheduler.sigmas
    start_idx = (sigmas - start_noise_level).abs().argmin().item()

    # Only use timesteps from start_idx onwards
    timesteps = timesteps[start_idx:]
    latents = z_noisy.to(pipe.transformer.dtype)

    for t in tqdm(timesteps, desc=f"Denoising from noise={start_noise_level:.2f}"):
        timestep = t.expand(latents.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.to(pipe.transformer.dtype),
            pooled_projections=pooled_embeds.to(pipe.transformer.dtype),
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Decode
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type='pil')

    return image[0]


@hydra.main(version_base=None, config_path="../configs/attack", config_name="base_sd3")
def main(cfg: DictConfig):
    """Main entry point for SD3 adversarial perturbation generation."""
    print(OmegaConf.to_yaml(cfg))
    time_start = time.time()

    args = cfg.attack

    epsilon = args.epsilon
    steps = args.steps
    input_size = args.input_size
    mode = args.mode
    alpha = args.alpha
    g_mode = args.g_mode
    output_path = args.output_path
    img_path = args.img_path
    device = args.device
    textual_weight = args.get('textual_weight', 1.0)
    mmdit_weight = args.get('mmdit_weight', 1.0)
    model_name = args.get('model_name', 'stabilityai/stable-diffusion-3.5-medium')

    # Convert device string
    if isinstance(device, int):
        device = f"cuda:{device}"
    elif isinstance(device, str) and device.isdigit():
        device = f"cuda:{device}"

    mode_name = f'{mode}_eps{epsilon}_steps{steps}_gmode{g_mode}'
    mode_name += f'_tw{textual_weight}_mw{mmdit_weight}'

    output_path = output_path + f'/{mode_name}/'
    mp(output_path)

    # Select input prompt based on image path
    if 'anime' in img_path:
        input_prompt = 'an anime picture'
    elif 'artwork' in img_path:
        input_prompt = 'an artwork painting'
    elif 'landscape' in img_path:
        input_prompt = 'a landscape photo'
    elif 'portrait' in img_path:
        input_prompt = 'a portrait photo'
    else:
        input_prompt = 'a photo'

    # Initialize
    config = init(
        epsilon=epsilon, alpha=alpha, steps=steps,
        mode=mode, g_mode=g_mode, device=device,
        input_prompt=input_prompt,
        textual_weight=textual_weight,
        mmdit_weight=mmdit_weight,
        input_size=input_size,
        model_name=model_name,
    )

    # Find images to protect
    img_paths = glob.glob(img_path + '/*.png') + glob.glob(img_path + '/*.jpg') + glob.glob(img_path + '/*.jpeg')

    max_exp_num = args.get('max_exp_num', 100)
    img_paths = img_paths[:max_exp_num]

    cprint(f'Found {len(img_paths)} images to process', 'y')

    for image_path in tqdm(img_paths):
        cprint(f'Processing: [{image_path}]', 'y')

        rsplit_image_path = image_path.rsplit('/')
        file_name = f"/{rsplit_image_path[-2]}/{rsplit_image_path[-1]}/"
        file_name = file_name.rsplit('.')[0]
        mp(output_path + file_name)

        target_image_path = 'test_images/target/MIST.png'
        img = load_image_from_path(image_path, input_size)
        tar_img = load_image_from_path(target_image_path, input_size) if os.path.exists(target_image_path) else None

        output_image, sdedit_results, loss_all = infer(img, config, tar_img, device=device)

        # Save adversarial image
        output = Image.fromarray(output_image.astype(np.uint8))
        output_name = output_path + f'{file_name}_attacked.png'
        output.save(output_name)

        # Save SDEdit results
        time_start_sdedit = time.time()
        for noise_key, pil_img in sdedit_results.items():
            sdedit_path = output_path + f'{file_name}_sdedit_{noise_key}.png'
            if isinstance(pil_img, PIL.Image.Image):
                pil_img.save(sdedit_path)
        print(f'SDEdit takes: {time.time() - time_start_sdedit:.2f}s')

        # Save loss curve
        if loss_all:
            loss_path = output_path + f'{file_name}_loss.npy'
            np.save(loss_path, np.array(loss_all))

        print(f'Total time: {time.time() - time_start:.2f}s')


if __name__ == '__main__':
    main()

# SD3 MMDiT Adversarial Attack Module
# Implements 4 MMDiT-specific loss strategies (A/B/C/D) for SD3 adversarial perturbation
# Based on SD3对抗扰动设计.md

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class AttentionMapHook:
    """Hook to capture attention maps and intermediate features from MMDiT JointTransformerBlocks."""

    def __init__(self):
        self.attn_maps = []
        self.hidden_states_list = []
        self.encoder_hidden_states_list = []
        self.img_stream_feats = []
        self.txt_stream_feats = []
        self.hooks = []

    def clear(self):
        self.attn_maps = []
        self.hidden_states_list = []
        self.encoder_hidden_states_list = []
        self.img_stream_feats = []
        self.txt_stream_feats = []

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class AttnMapCaptureProcessor:
    """Custom attention processor that captures the attention weights in Joint Attention.

    Replaces JointAttnProcessor2_0 temporarily to extract Q, K attention matrices
    for computing adversarial losses.
    """

    def __init__(self, hook: AttentionMapHook, block_idx: int, capture_attn: bool = True):
        self.hook = hook
        self.block_idx = block_idx
        self.capture_attn = capture_attn

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # sample projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # context projections
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Capture attention map before SDP attention
            N_img = query.shape[2]
            N_txt = encoder_hidden_states_query_proj.shape[2]

            full_query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            full_key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            full_value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

            if self.capture_attn:
                with torch.no_grad():
                    # Compute attention weights: [B, H, N_img+N_txt, N_img+N_txt]
                    attn_weights = torch.matmul(full_query, full_key.transpose(-2, -1)) / (head_dim ** 0.5)
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    self.hook.attn_maps.append(attn_weights.detach())

            hidden_states = F.scaled_dot_product_attention(
                full_query, full_key, full_value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # Split the attention outputs
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def register_feature_hooks(transformer, hook: AttentionMapHook, capture_attn: bool = True):
    """Register forward hooks on MMDiT transformer blocks to capture intermediate features.

    Args:
        transformer: SD3Transformer2DModel
        hook: AttentionMapHook instance to store captured data
        capture_attn: Whether to also capture attention maps (more memory intensive)
    """
    hook.remove()
    hook.clear()

    # Replace attention processors to capture attn maps
    if capture_attn:
        for idx, block in enumerate(transformer.transformer_blocks):
            block.attn.processor = AttnMapCaptureProcessor(hook, idx, capture_attn=True)

    # Register forward hooks on transformer blocks to capture hidden states
    def make_forward_hook(block_idx):
        def forward_hook(module, input, output):
            # output is (encoder_hidden_states, hidden_states)
            if isinstance(output, tuple) and len(output) == 2:
                enc_h, img_h = output
                hook.hidden_states_list.append(img_h.detach())
                hook.encoder_hidden_states_list.append(enc_h.detach() if enc_h is not None else None)
                # Image stream feature = hidden_states
                hook.img_stream_feats.append(img_h.detach())
                # Text stream feature = encoder_hidden_states
                hook.txt_stream_feats.append(enc_h.detach() if enc_h is not None else None)
        return forward_hook

    for idx, block in enumerate(transformer.transformer_blocks):
        h = block.register_forward_hook(make_forward_hook(idx))
        hook.hooks.append(h)

    return hook


def restore_processors(transformer):
    """Restore original JointAttnProcessor2_0 on all transformer blocks."""
    from diffusers.models.attention_processor import JointAttnProcessor2_0
    for block in transformer.transformer_blocks:
        block.attn.processor = JointAttnProcessor2_0()
        if hasattr(block, 'attn2') and block.attn2 is not None:
            block.attn2.processor = JointAttnProcessor2_0()


# ============================================================
# MMDiT Loss Functions (A/B/C/D)
# ============================================================

def cross_modal_disruption_loss(attn_maps, N_img_tokens):
    """Loss A: Cross-Modal Alignment Disruption Loss

    Maximize the entropy of text->image attention weights in Joint Attention,
    making the semantic injection from text to image uniform/random.

    Args:
        attn_maps: list of [B, H, N_img+N_txt, N_img+N_txt] attention weight tensors
        N_img_tokens: number of image tokens in the attention sequence

    Returns:
        loss scalar (to be minimized => attention entropy maximized)
    """
    total_loss = 0.0
    count = 0
    for attn_weights in attn_maps:
        # attn_weights: [B, H, N_total, N_total]
        # First N_img_tokens are image, rest are text
        N_total = attn_weights.shape[-1]
        N_txt = N_total - N_img_tokens

        # Extract text -> image attention block:
        # Image queries attending to text keys => attn[B, H, :N_img, N_img:]
        text_to_img_attn = attn_weights[:, :, :N_img_tokens, N_img_tokens:]  # [B, H, N_img, N_txt]

        # Strategy 1: Maximize attention entropy (minimize negative entropy)
        # This makes semantic injection uniform/random
        entropy = -(text_to_img_attn * torch.log(text_to_img_attn + 1e-8)).sum(dim=-1).mean()

        # We want to maximize entropy, so minimize negative entropy
        total_loss = total_loss - entropy
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=attn_maps[0].device if len(attn_maps) > 0 else 'cpu')
    return total_loss / count


def feature_divergence_loss(feat_adv_list, feat_clean_list):
    """Loss B: Attention Feature Shift Loss (Feature Space Adversarial Loss)

    Maximize cosine distance between adversarial and clean features in MMDiT blocks,
    and break Gram matrix similarity (style/texture statistics).

    Args:
        feat_adv_list: list of adversarial image stream features [B, N_img, D] per block
        feat_clean_list: list of clean image stream features [B, N_img, D] per block

    Returns:
        loss scalar (to be minimized via gradient ascent => feature divergence maximized)
    """
    total_loss = 0.0
    count = 0
    for feat_adv, feat_clean in zip(feat_adv_list, feat_clean_list):
        if feat_adv is None or feat_clean is None:
            continue

        # Cosine similarity: minimize to maximize divergence
        cos_sim = F.cosine_similarity(feat_adv.flatten(1), feat_clean.flatten(1), dim=1)
        cos_loss = cos_sim.mean()  # minimize this => maximize divergence

        # Gram matrix style disruption (maximize difference => minimize negative similarity)
        gram_adv = _gram_matrix(feat_adv)
        gram_clean = _gram_matrix(feat_clean)
        style_loss = -F.l1_loss(gram_adv, gram_clean)  # maximize L1 distance between Gram matrices

        # Combined: minimize cos_sim + maximize style divergence
        total_loss = total_loss + cos_loss + style_loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device='cpu')
    return total_loss / count


def _gram_matrix(x):
    """Compute Gram matrix for style representation.

    Args:
        x: [B, N, D] feature tensor
    Returns:
        [B, D, D] Gram matrix
    """
    B, N, D = x.shape
    feat = x.reshape(B, N, D)
    gram = torch.bmm(feat.transpose(1, 2), feat) / (N * D)
    return gram


def trajectory_divergence_loss(v_adv, v_clean):
    """Loss C: Temporal/Step Consistency Break Loss

    Maximize the divergence between adversarial and clean velocity predictions
    in the flow matching ODE trajectory.

    In SD3's flow matching framework, the model predicts velocity v.
    We maximize the angle (minimize cosine similarity) between v_adv and v_clean,
    causing the adversarial sample's denoising trajectory to diverge.

    Args:
        v_adv: velocity prediction for adversarial input [B, C, H, W]
        v_clean: velocity prediction for clean input [B, C, H, W] (no_grad)

    Returns:
        loss scalar (to be minimized => trajectory divergence maximized)
    """
    # Maximize angle between velocity directions
    # Minimize cosine similarity => trajectory points in wrong direction
    cos_sim = F.cosine_similarity(v_adv.flatten(1), v_clean.flatten(1), dim=1)
    loss = cos_sim.mean()

    return loss


def modality_imbalance_loss(img_stream_feats, txt_stream_feats):
    """Loss D: Modality Imbalance Loss

    Exploit MMDiT's dual-stream architecture:
    - Force image stream features to have abnormally high variance (noise injection)
    - Force Q_img and K_txt to be orthogonal (break cross-modal alignment)

    Args:
        img_stream_feats: list of image stream features [B, N_img, D] per block
        txt_stream_feats: list of text stream features [B, N_txt, D] per block

    Returns:
        loss scalar (to be minimized => modality imbalance maximized)
    """
    total_loss = 0.0
    count = 0

    for img_feat, txt_feat in zip(img_stream_feats, txt_stream_feats):
        if img_feat is None or txt_feat is None:
            continue

        # Strategy 1: Maximize image stream feature variance
        # Minimize negative variance => maximize variance
        img_var = img_feat.var(dim=[1, 2]).mean()
        var_loss = -img_var

        # Strategy 2: Force image and text features toward orthogonality
        # Minimize the mean absolute dot product
        # img_feat: [B, N_img, D], txt_feat: [B, N_txt, D]
        # Compute cross-correlation matrix: [B, N_img, N_txt]
        cross_corr = torch.bmm(
            F.normalize(img_feat, dim=-1),
            F.normalize(txt_feat, dim=-1).transpose(1, 2)
        )
        ortho_loss = cross_corr.abs().mean()

        # Combined: maximize variance + minimize correlation
        total_loss = total_loss + var_loss + 0.1 * ortho_loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device='cpu')
    return total_loss / count


def denoiser_prediction_loss(v_pred):
    """Loss O (Semantic): Denoiser's Prediction Error

    Semantic loss based on the denoiser's prediction magnitude.
    This measures the semantic difference as the norm of the predicted
    velocity in the flow matching framework. Maximizing this loss increases
    the prediction magnitude, indicating greater semantic disruption.

    Args:
        v_pred: velocity prediction from MMDiT [B, C, H, W]

    Returns:
        loss scalar (positive L2 norm)
        For g_mode='+': maximize this loss → increase prediction magnitude
        For g_mode='-': minimize this loss → decrease prediction magnitude
    """
    # Semantic loss: positive L2 norm of prediction
    # Larger norm = larger prediction = greater semantic difference
    # Ensure the loss matches the dtype of v_pred for proper gradients
    return v_pred.norm().to(v_pred.dtype)


# ============================================================
# SD3 PGD Attack with MMDiT losses
# ============================================================

class SD3_Linf_PGD:
    """PGD attack for SD3 with MMDiT-specific loss functions.

    Supports 5 attack modes:
        O: Textural Loss + Semantic Loss (baseline with denoiser prediction error)
        A: Cross-Modal Alignment Disruption
        B: Attention Feature Shift
        C: Temporal Consistency Break
        D: Modality Imbalance

    Modes A-D combine with TEXTURAL LOSS (VAE latent push):
        JOINT LOSS = textual_weight * TEXTURAL LOSS + mmdit_weight * MMDiT LOSS
    Mode O uses TEXTURAL LOSS + SEMANTIC LOSS as a baseline comparison.
    """

    def __init__(self, net, fn, epsilon, steps, eps_iter, clip_min=-1.0, clip_max=1.0,
                 targeted=True, g_mode='+', mmdit_mode='A', capture_attn=True,
                 textual_weight=1.0, mmdit_weight=1.0):
        """
        Args:
            net: SD3 target model wrapper (SD3_target_model)
            fn: identity loss module
            epsilon: L_inf perturbation budget
            steps: number of attack iterations
            eps_iter: step size per iteration
            clip_min, clip_max: pixel value range
            targeted: whether attack is targeted
            g_mode: gradient direction ('+' or '-')
            mmdit_mode: which MMDiT loss to use ('O', 'A', 'B', 'C', 'D')
            capture_attn: whether to capture attention maps (needed for loss A)
            textual_weight: weight for textual loss component
            mmdit_weight: weight for MMDiT loss component
        """
        self.net = net
        self.fn = fn
        self.eps = epsilon
        self.step_size = eps_iter
        self.iters = steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.g_mode = g_mode
        self.g_dir = 1. if g_mode == '+' else -1.
        self.mmdit_mode = mmdit_mode
        self.capture_attn = capture_attn
        self.textual_weight = textual_weight
        self.mmdit_weight = mmdit_weight

        self.cirt = nn.MSELoss(reduction="sum")

    def pgd_sd3(self, X, target_image=None, random_start=False):
        """Run PGD attack on SD3 with MMDiT-specific losses.

        Args:
            X: input image tensor [1, 3, H, W] in [-1, 1] range
            target_image: target image for textual loss [1, 3, H, W]
            random_start: whether to start from random perturbation

        Returns:
            X_adv: adversarial image tensor
            loss_history: dict of lists with keys 'total', 'textual', 'mmdit' per iteration
        """
        device = X.device

        if random_start:
            X_adv = X.clone().detach() + (torch.rand(*X.shape) * 2 * self.eps - self.eps).to(device)
        else:
            X_adv = X.clone().detach()

        loss_history = {'total': [], 'textual': [], 'mmdit': []}
        pbar = tqdm(range(self.iters), desc=f"SD3-PGD mode={self.mmdit_mode}")

        # Pre-compute clean features for loss B (feature divergence)
        feat_clean_list = None
        if self.mmdit_mode == 'B':
            feat_clean_list = self._compute_clean_features(X)

        for i in pbar:
            actual_step_size = self.step_size

            X_adv.requires_grad_(True)

            # Compute joint loss (returns dict with component losses)
            loss, components = self._compute_loss(X_adv, X, target_image, feat_clean_list, device)

            pbar.set_description(
                f"SD3-PGD mode={self.mmdit_mode} | total={loss.item():.3f} "
                f"textual={components['textual']:.3f} mmdit={components['mmdit']:.3f}"
            )

            loss.backward()
            grad = X_adv.grad.detach()

            # Update
            X_adv = X_adv.detach() + self.g_dir * grad.sign() * actual_step_size
            # Clip to epsilon ball
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            # Clip to valid range
            X_adv = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)

            loss_history['total'].append(loss.item())
            loss_history['textual'].append(components['textual'])
            loss_history['mmdit'].append(components['mmdit'])
            X_adv.grad = None

            torch.cuda.empty_cache()

        return X_adv, loss_history

    def _compute_clean_features(self, X_clean):
        """Compute clean image features through MMDiT for loss B comparison."""
        pipe = self.net.pipe
        hook = AttentionMapHook()
        transformer = pipe.transformer

        register_feature_hooks(transformer, hook, capture_attn=False)

        with torch.no_grad():
            z_clean = pipe.vae.encode(X_clean.to(pipe.vae.dtype)).latent_dist.sample()
            z_clean = (z_clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_clean = z_clean.to(pipe.transformer.dtype)

            timestep = torch.tensor([0.5], device=X_clean.device, dtype=pipe.transformer.dtype)
            prompt_embeds = self.net.prompt_embeds.to(pipe.transformer.dtype)
            pooled_embeds = self.net.pooled_prompt_embeds.to(pipe.transformer.dtype)

            _ = transformer(
                hidden_states=z_clean,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )

        feat_clean = [f.clone() for f in hook.img_stream_feats]
        hook.remove()
        restore_processors(transformer)

        return feat_clean

    def _compute_loss(self, X_adv, X_clean, target_image, feat_clean_list, device):
        """Compute the joint loss = textual_weight * TEXTUAL LOSS + mmdit_weight * MMDiT LOSS.

        This follows the design in SD3对抗扰动设计.md:
        JOINT LOSS = TEXTUAL LOSS + lambda * MMDiT LOSS

        Mode 'O': Textural Loss + Semantic Loss (baseline with denoiser prediction error)

        Returns:
            (loss_tensor, components_dict) where components_dict has keys:
                'textual': float, textual loss value
                'mmdit': float, mmdit loss value
        """
        pipe = self.net.pipe

        # ---- Mode O: Textural Loss + Semantic Loss (baseline) ----
        if self.mmdit_mode == 'O':
            transformer = pipe.transformer

            # Encode to latent space
            z_adv = pipe.vae.encode(X_adv.to(pipe.vae.dtype)).latent_dist.mean
            z_adv = (z_adv - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_adv = z_adv.to(pipe.transformer.dtype)

            # Sample timestep for flow matching
            t_val = torch.rand(1, device=device).item()
            timestep = torch.tensor([t_val], device=device, dtype=pipe.transformer.dtype)

            # Get prompt embeddings
            prompt_embeds = self.net.prompt_embeds.to(pipe.transformer.dtype)
            pooled_embeds = self.net.pooled_prompt_embeds.to(pipe.transformer.dtype)

            # Forward through MMDiT for semantic loss
            v_pred = transformer(
                hidden_states=z_adv,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

            # Semantic loss: denoiser's prediction error
            semantic_loss = denoiser_prediction_loss(v_pred)

            # Textural loss: VAE latent push toward target
            textual_loss_val = 0.0
            textual_loss = torch.tensor(0.0, device=device, dtype=pipe.transformer.dtype)
            if target_image is not None:
                with torch.no_grad():
                    z_target = pipe.vae.encode(target_image.to(pipe.vae.dtype)).latent_dist.mean
                    z_target = (z_target - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                    z_target = z_target.to(pipe.transformer.dtype).detach()
                textual_loss = self.cirt(z_adv, z_target)
                textual_loss_val = textual_loss.item()

            # Joint loss: Textural + Semantic
            joint_loss = self.textual_weight * textual_loss + self.mmdit_weight * semantic_loss

            components = {
                'textual': textual_loss_val,
                'mmdit': semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else float(semantic_loss),
            }
            return joint_loss, components

        # ---- Modes A/B/C/D: Joint Loss with MMDiT ----
        hook = AttentionMapHook()
        transformer = pipe.transformer

        need_attn = (self.mmdit_mode == 'A')
        register_feature_hooks(transformer, hook, capture_attn=need_attn)

        # Encode to latent space (with gradient through VAE)
        z_adv = pipe.vae.encode(X_adv.to(pipe.vae.dtype)).latent_dist.mean
        z_adv = (z_adv - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_adv = z_adv.to(pipe.transformer.dtype)

        # Sample timestep for the attack
        # Flow matching: t in [0, 1], sample uniformly
        t_val = torch.rand(1, device=device).item()
        timestep = torch.tensor([t_val], device=device, dtype=pipe.transformer.dtype)

        # Get prompt embeddings
        prompt_embeds = self.net.prompt_embeds.to(pipe.transformer.dtype)
        pooled_embeds = self.net.pooled_prompt_embeds.to(pipe.transformer.dtype)

        # Forward through MMDiT transformer
        v_pred = transformer(
            hidden_states=z_adv,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

        # ---- Compute MMDiT loss ----
        mmdit_loss = torch.tensor(0.0, device=device)

        if self.mmdit_mode == 'A':
            # Loss A: Cross-Modal Alignment Disruption
            if len(hook.attn_maps) > 0:
                # Determine N_img_tokens from hidden states
                N_img = hook.hidden_states_list[0].shape[1] if len(hook.hidden_states_list) > 0 else z_adv.shape[2] * z_adv.shape[3] // (pipe.transformer.config.patch_size ** 2)
                mmdit_loss = cross_modal_disruption_loss(hook.attn_maps, N_img)

        elif self.mmdit_mode == 'B':
            # Loss B: Feature Shift / Divergence
            if len(hook.img_stream_feats) > 0 and feat_clean_list is not None:
                mmdit_loss = feature_divergence_loss(hook.img_stream_feats, feat_clean_list)

        elif self.mmdit_mode == 'C':
            # Loss C: Trajectory Divergence
            with torch.no_grad():
                z_clean = pipe.vae.encode(X_clean.to(pipe.vae.dtype)).latent_dist.mean
                z_clean = (z_clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                z_clean = z_clean.to(pipe.transformer.dtype)
                v_clean = transformer(
                    hidden_states=z_clean,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]
            mmdit_loss = trajectory_divergence_loss(v_pred, v_clean.detach())

        elif self.mmdit_mode == 'D':
            # Loss D: Modality Imbalance
            if len(hook.img_stream_feats) > 0 and len(hook.txt_stream_feats) > 0:
                mmdit_loss = modality_imbalance_loss(hook.img_stream_feats, hook.txt_stream_feats)

        # ---- Compute Textual Loss (VAE latent push) ----
        textual_loss = torch.tensor(0.0, device=device)
        if target_image is not None:
            with torch.no_grad():
                z_target = pipe.vae.encode(target_image.to(pipe.vae.dtype)).latent_dist.mean
                z_target = (z_target - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                z_target = z_target.to(pipe.transformer.dtype).detach()
            textual_loss = self.cirt(z_adv, z_target)

        # ---- Joint Loss ----
        # JOINT LOSS = TEXTUAL LOSS + lambda * MMDiT LOSS
        joint_loss = self.textual_weight * textual_loss + self.mmdit_weight * mmdit_loss

        # Clean up hooks
        hook.remove()
        restore_processors(transformer)

        components = {
            'textual': textual_loss.item(),
            'mmdit': mmdit_loss.item() if isinstance(mmdit_loss, torch.Tensor) else float(mmdit_loss),
        }
        return joint_loss, components

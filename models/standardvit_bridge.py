"""Standard (open_clip-native) ViT-B/16 vision backbone with HiVG's
Multi-layer Adaptive Cross-modal Bridge.

Drop-in replacement for CLIP_Vision_Model_with_Crossmodal_Bridge / LoopViTVisionWithBridge
(see HiVG.py) when args.model == "StandardViT-Distilled": same forward() signature,
same returned dict shape, so HiVG.forward() needs no changes downstream of vision
encoding.

Lets a clip-kd-snn CLIP-KD checkpoint of the *plain* (non-LoopViT) ViT-B/16 student
-- open_clip's native VisionTransformer, the architecture actually trained by
e.g. `kd_vit_b16_to_b16_cc3m12m` -- be dropped into HiVG fine-tuning, so it can be
compared head-to-head against the LoopViT arm (see loopvit_bridge.py) on identical
data/recipe/bridge architecture, differing only in the vision backbone.

Because clip-kd-snn's standard ViT-B/16 student *is* open_clip's own ViT-B-16
class (CLIPWrapper just wraps it, see src/models/factory.py:build_student_model),
loading its checkpoint here needs no key remapping at all -- `self.visual` below
is the exact same class, so `load_submodule_from_lightning_ckpt` matches
'model.visual.*' keys 1:1.

LoRA: open_clip's ResidualAttentionBlock uses a fused nn.MultiheadAttention, so
hilora.py's `_patch_openclip_block` swaps it for `_UnfusedOpenCLIPAttention`,
which splits in_proj_weight into q/k/v LoRALinear plus out_proj and reimplements
forward via F.scaled_dot_product_attention. The adapted scope is therefore
q_proj/k_proj/v_proj/out_proj -- identical to the HF-CLIP path and to HiVG's
authors' target_modules, not MLP-only. Unlike LoopViT (loop_core_depth=1, one shared block,
so HiVG's hierarchical 3-stage curriculum is degenerate), this backbone has 12
distinct blocks, so the genuine cumulative HiLoRA staging (5/8/12 blocks at
stages 1/2/3) applies via hilora.py's blocks_for_stage -- same helper the native
src/downstream/ grounding pipeline already uses.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import open_clip
import torch
import torch.nn as nn

from src.downstream.models.hilora import blocks_for_stage, patch_block_with_lora, set_lora_trainable
from src.downstream.models.macb import MACB
from src.downstream.utils.checkpoint_io import load_submodule_from_lightning_ckpt

from .loopvit_bridge import _BRIDGE_EXECS, _EXEC_SLOT, _VisionConfig


def load_visual_proj_from_standard_ckpt(ckpt_path: str, target_linear: nn.Linear) -> None:
    """Loads model.visual.proj from a CLIP-KD Lightning ckpt (open_clip-native
    ViT-B-16 student) into target_linear.

    open_clip's VisionTransformer bakes its own 768->512 head into `visual.proj`,
    a raw nn.Parameter used as `pooled @ proj` (shape (768, 512)) -- the
    transpose of nn.Linear.weight's (out_features, in_features) convention,
    hence the .t() below.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    raw = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
    for key in ("student.model.visual.proj", "model.visual.proj"):
        if key in raw:
            target_linear.weight.data.copy_(raw[key].t())
            return
    raise RuntimeError(
        f"No 'model.visual.proj' (optionally 'student.'-prefixed) found in {ckpt_path}"
    )


class StandardViTVisionWithBridge(nn.Module):
    """Standard open_clip ViT-B/16 visual backbone + 4 externally-held MACB
    cross-modal bridges.

    Args:
        args: parsed CLI namespace. Reads standardvit_checkpoint.
        extract_text_layer: 1-indexed text layer ids feeding the bridges
                             (same list HiVG.__init__ computes per dataset).
        text_hidden_size: CLIP text tower hidden size (512 for ViT-B/16).
    """

    def __init__(self, args, extract_text_layer: list[int], text_hidden_size: int) -> None:
        super().__init__()
        embed_dim = 768

        # arch is configurable so a known-good CLIP can be pushed through this exact
        # code path as a control. clip-kd-snn's students are plain-GELU "ViT-B-16";
        # OpenAI/DFN2B weights are QuickGELU and need "ViT-B-16-quickgelu", which has
        # identical parameter shapes but a different activation -- loading one into
        # the other is silent and merely wrong.
        arch = getattr(args, "standardvit_arch", "ViT-B-16")
        self.visual = open_clip.create_model(arch, pretrained=None).visual
        self.config = _VisionConfig(embed_dim)

        self.bridges = nn.ModuleList([
            MACB(
                visual_dim=embed_dim,
                text_dim=text_hidden_size,
                n_text_layers=len(extract_text_layer),
                text_seq_len=77,
                num_heads=8,
                # opt-in: start the bridge as a no-op so it cannot perturb the
                # frozen backbone's residual stream before it has learned anything
                zero_init=bool(getattr(args, "macb_zero_init", False)),
                text_ln=bool(getattr(args, "macb_text_ln", False)),
                # opt-in: reproduce HiVG.py:452's transposed cross-attention input,
                # so the port can be run in the regime the paper's numbers came from
                hivg_layout=bool(getattr(args, "macb_hivg_layout", False)),
                text_const=bool(getattr(args, "macb_text_const", False)),
            )
            for _ in range(len(_BRIDGE_EXECS))
        ])

        # Ablation: skip the bridge injection entirely, the equivalent of HiVG's
        # "MACB off" row (an empty adapt_layer). The modules are still constructed but
        # never called, so they receive no gradient and contribute nothing -- neither
        # capacity nor cross-modal signal.
        self.macb_disable = bool(getattr(args, "macb_disable", False))
        if self.macb_disable:
            print("[StandardViT-Distilled] MACB DISABLED -- no cross-modal bridge injection")

        standardvit_checkpoint = getattr(args, "standardvit_checkpoint", "")
        if standardvit_checkpoint:
            load_submodule_from_lightning_ckpt(standardvit_checkpoint, self.visual, "visual", strict=True)

    def iter_blocks(self):
        yield from self.visual.transformer.resblocks

    def patch_lora_stage(self, stage: int, rank: int, alpha: float) -> None:
        """Cumulative HiLoRA staging (0 = no LoRA, 1/2/3 = 5/8/12 blocks active).

        Mirrors src/downstream/lightning/grounding_module.py's setup(): patches
        adapters onto the active blocks (idempotent) and turns their gradients
        on. Blocks outside the active set stay frozen and unpatched.
        """
        if stage == 0:
            return
        active_blocks = blocks_for_stage(self, stage)
        for block in active_blocks:
            patch_block_with_lora(block, rank=rank, alpha=alpha, dropout=0.1)
        set_lora_trainable(active_blocks, True)

    def forward(
        self,
        adapt_layer,
        text_states,
        reg_src,
        pixel_values: torch.Tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Matches CLIP_Vision_Model_with_Crossmodal_Bridge.forward's signature
        and return shape exactly (adapt_layer/reg_src/output_attentions are
        accepted but unused -- adapt_layer is fixed by construction to the 4
        bridge slots; reg_src is consumed by HiVG.forward, not the backbone;
        attentions are not exposed -- nn.MultiheadAttention's call here uses
        need_weights=False internally via ResidualAttentionBlock.attention()).
        """
        hidden_states = self.visual._embeds(pixel_values)
        encoder_states = (hidden_states,)

        exec_idx = 0
        for block in self.visual.transformer.resblocks:
            exec_idx += 1
            if exec_idx in _EXEC_SLOT and not self.macb_disable:
                slot = _EXEC_SLOT[exec_idx]
                attn_out = block.ls_1(block.attention(q_x=block.ln_1(hidden_states)))
                hidden_states = hidden_states + attn_out
                hidden_states = hidden_states + self.bridges[slot](hidden_states, text_states)
                mlp_out = block.ls_2(block.mlp(block.ln_2(hidden_states)))
                hidden_states = hidden_states + mlp_out
            else:
                hidden_states = block(hidden_states)
            encoder_states = encoder_states + (hidden_states,)

        last_hidden_state = hidden_states
        # open_clip's own _pool() applies ln_post to the full sequence before
        # slicing the CLS token; LayerNorm is per-token, so slicing first and
        # normalising only the CLS token (as done here) is equivalent.
        pooled_output = self.visual.ln_post(last_hidden_state[:, 0, :])

        if not return_dict:
            return (last_hidden_state, pooled_output, encoder_states, None)

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_states,
            "attentions": None,
        }

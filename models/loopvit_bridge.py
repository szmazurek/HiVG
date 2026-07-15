"""LoopViT vision backbone with HiVG's Multi-layer Adaptive Cross-modal Bridge.

Drop-in replacement for CLIP_Vision_Model_with_Crossmodal_Bridge (see HiVG.py)
when args.model == "LoopViT": same forward() signature, same returned dict
shape, so HiVG.forward() needs no changes downstream of vision encoding.

loop_core_depth / max_loop_steps / loop_mode are configurable (default
depth=1, mode="global", matching clip-kd-snn's configs/downstream/model/
bvit_d1.yaml). Regardless of depth, block-execution count is always
max_loop_steps * loop_core_depth = 12 by construction across the three target
configs (1x12, 3x4, 6x2), so the four cross-modal bridges always inject/extract
at the same execution indices {1,4,8,12}. Because distinct loop iterations can
reuse the *same* physical block object (loop_core_depth < max_loop_steps), the
bridges cannot live on the block itself (that would collapse multiple distinct
bridges into one, the way HiVG's literal CLIPEncoderLayer_with_Crossmodal_Bridge
attaches one bridge per distinct CLIP layer) -- they are instead held externally
in an nn.ModuleList keyed by injection slot, and applied by this wrapper's
block-execution loop. Mirrors the pattern already validated in clip-kd-snn's
src/downstream/models/hivg_loopvit.py.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from src.downstream.models.hilora import blocks_for_stage, patch_block_with_lora, set_lora_trainable
from src.downstream.models.macb import MACB
from src.downstream.utils.checkpoint_io import load_submodule_from_lightning_ckpt
from src.models.visual_encoders.loopvit import LoopViT

# Block-execution indices where the bridge injects / features get extracted.
# Same 4 points HiVG.py uses for extract_vision_layer=[1,4,8,12] /
# adapt_layer=[0,3,7,11] -- pre- vs post-layer numbering for the same spots.
_BRIDGE_EXECS = (1, 4, 8, 12)
_EXEC_SLOT = {exec_idx: slot for slot, exec_idx in enumerate(_BRIDGE_EXECS)}


def load_visual_projection_from_lightning_ckpt(ckpt_path: str, target_linear: nn.Linear) -> None:
    """Loads model.visual_proj.weight from a CLIP-KD Lightning ckpt into target_linear.

    Stands in for CLIPModel.visual_projection (the OpenAI-pretrained 768->512
    head), which is not meaningful for a LoopViT-produced CLS embedding --
    LoopViTCLIPModel trains its own visual_proj (same Linear(768, 512,
    bias=False) shape, see src/models/loopvit_clip.py) during CLIP-KD.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    raw = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
    for key in ("student.model.visual_proj.weight", "model.visual_proj.weight"):
        if key in raw:
            target_linear.weight.data.copy_(raw[key])
            return
    raise RuntimeError(
        f"No 'model.visual_proj.weight' (optionally 'student.'-prefixed) found in {ckpt_path}"
    )


class _VisionConfig:
    """Minimal stand-in for CLIPVisionConfig -- HiVG.py only reads .hidden_size."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size


class LoopViTVisionWithBridge(nn.Module):
    """LoopViT visual backbone + 4 externally-held MACB cross-modal bridges.

    Args:
        args: parsed CLI namespace. Reads imsize, loopvit_checkpoint,
              loopvit_max_loop_steps.
        extract_text_layer: 1-indexed text layer ids feeding the bridges
                             (same list HiVG.__init__ computes per dataset).
        text_hidden_size: CLIP text tower hidden size (512 for ViT-B/16).
    """

    def __init__(self, args, extract_text_layer: list[int], text_hidden_size: int) -> None:
        super().__init__()
        embed_dim = 768
        loop_core_depth = int(getattr(args, "loopvit_loop_core_depth", 1))
        max_loop_steps = int(getattr(args, "loopvit_max_loop_steps", 12))
        loop_mode = str(getattr(args, "loopvit_loop_mode", "global"))
        loop_schedule = getattr(args, "loopvit_loop_schedule", None)

        self.loopvit = LoopViT(
            img_size=args.imsize,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=embed_dim,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.0,
            loop_core_depth=loop_core_depth,
            max_loop_steps=max_loop_steps,
            min_loop_steps=1,
            add_step_embeddings=False,
            use_exit_gate=False,
            swiglu=False,
            loop_mode=loop_mode,
            loop_schedule=loop_schedule,
        )
        self.config = _VisionConfig(embed_dim)

        self.bridges = nn.ModuleList([
            MACB(
                visual_dim=embed_dim,
                text_dim=text_hidden_size,
                n_text_layers=len(extract_text_layer),
                text_seq_len=77,
                num_heads=8,
            )
            for _ in range(len(_BRIDGE_EXECS))
        ])

        loopvit_checkpoint = getattr(args, "loopvit_checkpoint", "")
        if loopvit_checkpoint:
            load_submodule_from_lightning_ckpt(loopvit_checkpoint, self.loopvit, "visual", strict=True)

    def iter_blocks(self):
        yield from self.loopvit.iter_blocks()

    def patch_lora_stage(self, stage: int, rank: int, alpha: float) -> None:
        """Cumulative HiLoRA staging: one adapter per unique physical block,
        unlocked progressively across stages 1/2/3 (0 = no LoRA).

        Mirrors StandardViTVisionWithBridge.patch_lora_stage exactly, via the
        same blocks_for_stage() generalization (dedups blocks by identity in
        first-execution order, splits cumulatively -- 5/8/12 for 12 unique
        blocks, an even cumulative split otherwise, e.g. 1/2/3 for
        loop_core_depth=3, 2/4/6 for loop_core_depth=6). At loop_core_depth=1
        this is behavior-preserving relative to the old single-block
        patch_lora(): blocks_for_stage() returns the same one block at every
        stage >= 1. Text tower is handled by set_HiLoRA in HiVG.py after this
        call.
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
        attentions are not exposed -- LoopViT's attention runs through
        F.scaled_dot_product_attention, which does not return weights).
        """
        hidden_states = self.loopvit.image_tokens(pixel_values)
        encoder_states = (hidden_states,)

        exec_idx = 0
        for block in self.loopvit.iter_blocks():
            exec_idx += 1
            if exec_idx in _EXEC_SLOT:
                slot = _EXEC_SLOT[exec_idx]
                hidden_states = hidden_states + block.self_attn_sublayer(hidden_states)
                hidden_states = hidden_states + self.bridges[slot](hidden_states, text_states)
                hidden_states = hidden_states + block.mlp_sublayer(hidden_states)
            else:
                hidden_states = block(hidden_states)
            encoder_states = encoder_states + (hidden_states,)

        last_hidden_state = hidden_states
        pooled_output = self.loopvit.head_norm(last_hidden_state[:, 0, :])

        if not return_dict:
            return (last_hidden_state, pooled_output, encoder_states, None)

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_states,
            "attentions": None,
        }

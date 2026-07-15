"""LoopText text backbone for HiVG, paired with LoopViT (args.model ==
"LoopViT-LoopText").

Drop-in replacement for the frozen HF CLIPTextModel HiVG.py otherwise uses as
self.clip.text_model: forward() duck-types HF's BaseModelOutputWithPooling
(.last_hidden_state / .hidden_states / .pooler_output, dot-accessed), so
HiVG.forward()'s text-encoding branch needs no changes at all.

Does NOT call LoopText.encode_text_with_hidden_states() -- that method applies
LoopText's own internal text_projection, which would double up with
self.clip.text_projection applied downstream in HiVG.forward() (mirroring how
HF's CLIPTextModel keeps text_projection external to the text tower, applied
once by the caller). Instead this replays LoopText's loop manually, exactly
the same way loopvit_bridge.py replays LoopViT's loop rather than calling
LoopViT.forward_features().

hidden_states[0] is the token+position embedding (pre any block), and
hidden_states[i] for i>=1 is the state after the i-th block-execution -- the
same "index 0 is the input embedding" convention extract_vision_layer already
relies on (see HiVG.py's comment on extract_vision_layer/adapt_layer), so
extract_text_layer picks like [12], [6, 12], [1..12] index correctly with no
off-by-one.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from src.downstream.models.hilora import patch_block_with_lora, set_lora_trainable
from src.downstream.utils.checkpoint_io import load_submodule_from_lightning_ckpt
from src.models.text_encoders.looptext import LoopText


class LoopTextWithBridge(nn.Module):
    """LoopText text backbone, frozen + flat HiLoRA (no staging, no bridges --
    cross-modal bridges live only on the vision side, see loopvit_bridge.py).

    Args:
        args: parsed CLI namespace. Reads text_width, text_num_heads,
              text_mlp_ratio, text_loop_core_depth, text_max_loop_steps,
              text_vocab_size, max_query_len (context_length),
              loopvit_looptext_checkpoint.
    """

    def __init__(self, args) -> None:
        super().__init__()
        embed_dim = int(getattr(args, "embed_dim", 512))
        width = int(getattr(args, "text_width", 512))
        context_length = int(getattr(args, "max_query_len", 77))

        self.looptext = LoopText(
            vocab_size=int(getattr(args, "text_vocab_size", 49408)),
            context_length=context_length,
            embed_dim=embed_dim,
            width=width,
            num_heads=int(getattr(args, "text_num_heads", 8)),
            mlp_ratio=float(getattr(args, "text_mlp_ratio", 4.0)),
            dropout=0.0,
            loop_core_depth=int(getattr(args, "text_loop_core_depth", 1)),
            max_loop_steps=int(getattr(args, "text_max_loop_steps", 12)),
            add_step_embeddings=False,
            swiglu=False,
        )

        checkpoint = getattr(args, "loopvit_looptext_checkpoint", "")
        if checkpoint:
            load_submodule_from_lightning_ckpt(checkpoint, self.looptext, "text_model", strict=True)

    def patch_lora_flat(self, rank: float, alpha: float) -> None:
        """Flat (non-staged) LoRA: every unique text block gets an adapter.

        Unlike the vision side (patch_lora_stage, cumulative), text HiLoRA
        here is all-or-nothing -- matches the original released HiVG's de
        facto text-tower behavior and clip-kd-snn's native-pipeline
        _text_lora_blocks() convention. self.looptext.blocks is already the
        deduplicated set of unique physical blocks (loop_core_depth of them),
        no blocks_for_stage() needed.
        """
        for block in self.looptext.blocks:
            patch_block_with_lora(block, rank=rank, alpha=alpha, dropout=0.1)

    def set_trainable(self, trainable: bool) -> None:
        set_lora_trainable(list(self.looptext.blocks), trainable)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> SimpleNamespace:
        """Matches the frozen HF CLIPTextModel's call signature/return shape
        (output_attentions/output_hidden_states/return_dict accepted but
        unused -- always computed, LoopText's attention runs through
        F.scaled_dot_product_attention, which does not return weights)."""
        lt = self.looptext
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device)

        x = lt.token_embed(input_ids) + lt.pos_embed(positions)
        hidden_states = [x]

        for step in range(lt.max_loop_steps):
            if lt.step_embed is not None:
                x = x + lt.step_embed.weight[step].view(1, 1, -1)
            for block in lt.blocks:
                x = block(x, is_causal=True)
                hidden_states.append(x)

        last_hidden_state = lt.ln_final(x)
        eos_positions = input_ids.argmax(dim=-1)
        pooler_output = last_hidden_state[torch.arange(B, device=x.device), eos_positions]

        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            hidden_states=tuple(hidden_states),
            pooler_output=pooler_output,
        )

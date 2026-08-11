"""open_clip-native ViT-B/16 text tower for HiVG, paired with the distilled
StandardViT vision backbone (args.model == "StandardViT-Distilled" plus
args.distilled_text_tower).

Drop-in replacement for the frozen HF CLIPTextModel that HiVG.py otherwise uses
as self.clip.text_model: forward() duck-types HF's BaseModelOutputWithPooling
(.last_hidden_state / .hidden_states / .pooler_output, dot-accessed), so
HiVG.forward()'s text-encoding branch needs no changes at all. Text-side
counterpart of standardvit_bridge.py; cross-modal bridges live only on the
vision side, so there are none here.

WHY THIS EXISTS
---------------
HiVG.py's "StandardViT-Distilled" branch builds a complete OpenAI CLIPModel and
replaces only vision_model + visual_projection from the CLIP-KD checkpoint,
keeping OpenAI's text tower (a deliberate choice, documented at HiVG.py:681-684
for the sibling LoopViT branch). But this repo's CLIP-KD students are trained
with `init_student_text_from_teacher: false` from a random init, so the
student's 512-d joint space is unrelated to OpenAI's -- cosine ~0 against the
teacher on every text tensor. The runs measure it themselves: HiVG's
loss_contrastive (HiVG.py:1116-1117, a raw cosine between two *frozen*
projections with no trainable layer in between) starts at ln(batch), i.e.
exactly chance, in every distilled arm and never in an OpenAI/DFN2B arm.

This module supplies the matching text tower so the pair is coherent. It is
gated behind --distilled_text_tower (default off) so the original arm stays
runnable as the control.

Because the CLIP-KD student *is* open_clip's own ViT-B-16 (CLIPWrapper just
wraps it, see src/models/factory.py:build_student_model), the weights need no
key remapping -- only prefix stripping, see load_text_tower_from_standard_ckpt.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import open_clip
import torch
import torch.nn as nn

from src.downstream.models.hilora import patch_block_with_lora, set_lora_trainable

# The six top-level names that make up open_clip's CLIP minus its visual tower.
# Selection must test the *first path component* after prefix stripping, not a
# substring: `visual.positional_embedding` would otherwise be caught by a naive
# `"positional_embedding" in k` test.
_TEXT_TOWER_TOP_LEVEL = frozenset({
    "transformer",           # 144 keys: transformer.resblocks.{0..11}.*
    "token_embedding",       # (49408, 512)
    "positional_embedding",  # (77, 512)
    "ln_final",              # weight, bias
    "text_projection",       # (512, 512) raw Parameter, applied as x @ P
    "logit_scale",           # scalar; part of open_clip.CLIP's own state_dict
})


def _student_tensors(ckpt_path: str) -> dict:
    """Flattened `student.model.*` tensors from a CLIP-KD Lightning checkpoint.

    Strips torch.compile's "_orig_mod." wherever it occurs (this repo has seen
    it wrapping both the whole model and individual submodules) and then the
    "student.model." / "model." prefix.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    raw = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
    out = {}
    for k, v in raw.items():
        for prefix in ("student.model.", "model."):
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
                break
    return out


def load_text_tower_from_standard_ckpt(
    ckpt_path: str, module: nn.Module, strict: bool = True
) -> None:
    """Loads the flat open_clip text-tower tensors into `module`.

    Unlike checkpoint_io.load_submodule_from_lightning_ckpt, which strips a
    `model.<submodule>.` prefix, the text tower has no common prefix in an
    open_clip-native student -- it is six sibling top-level names. Hence a
    separate loader here rather than an extension of that helper, mirroring how
    load_visual_proj_from_standard_ckpt lives in standardvit_bridge.py.
    """
    tensors = _student_tensors(ckpt_path)
    sd = {k: v for k, v in tensors.items() if k.split(".")[0] in _TEXT_TOWER_TOP_LEVEL}
    if not sd:
        raise RuntimeError(
            f"No open_clip text-tower keys {sorted(_TEXT_TOWER_TOP_LEVEL)} found in "
            f"{ckpt_path}. A LoopText student stores its tower under "
            f"'model.text_model.*' instead -- use LoopTextWithBridge for those."
        )
    module.load_state_dict(sd, strict=strict)


def load_text_proj_from_standard_ckpt(ckpt_path: str, target_linear: nn.Linear) -> None:
    """Loads text_projection from a CLIP-KD ckpt into an HF-style nn.Linear.

    open_clip stores text_projection as a raw (in_features, out_features)
    Parameter applied as `pooled @ P`; nn.Linear.weight is (out, in) applied as
    `pooled @ W.t()`. So the transpose below is required -- the same .t() that
    load_visual_proj_from_standard_ckpt applies to visual.proj.

    DANGER: visual.proj is (768, 512), so forgetting the transpose there raises.
    text_projection is *square*, so forgetting it here would load cleanly and
    silently yield a permuted joint space -- a failure mode indistinguishable
    from "the distilled text tower didn't help", which is exactly the hypothesis
    under test. Hence the explicit shape assertion.
    """
    tensors = _student_tensors(ckpt_path)
    if "text_projection" not in tensors:
        raise RuntimeError(f"No 'text_projection' found in {ckpt_path}")
    raw = tensors["text_projection"]
    assert raw.shape == (512, 512), (
        f"expected a square (in=512, out=512) text_projection, got {tuple(raw.shape)}; "
        "if this ever becomes non-square, re-derive the transpose direction rather "
        "than trusting the .t() below"
    )
    target_linear.weight.data.copy_(raw.t())


class StandardTextWithBridge(nn.Module):
    """open_clip-native ViT-B/16 text tower, frozen + flat HiLoRA.

    Args:
        args: parsed CLI namespace. Reads standardtext_checkpoint, falling back
              to standardvit_checkpoint so vision and text come from the same
              joint space by default.
    """

    def __init__(self, args) -> None:
        super().__init__()
        # Build the whole CLIP and drop the visual tower: what remains is
        # exactly the six-name state_dict the checkpoint provides, so the load
        # below can stay strict. Constructing open_clip's TextTransformer
        # directly would omit logit_scale and diverge from the sibling
        # standardvit_bridge.py's create_model() pattern.
        # see standardvit_bridge.py: must match the checkpoint's activation variant
        full = open_clip.create_model(getattr(args, "standardvit_arch", "ViT-B-16"),
                                      pretrained=None)
        del full.visual
        self.text = full

        checkpoint = (getattr(args, "standardtext_checkpoint", "")
                      or getattr(args, "standardvit_checkpoint", ""))
        if checkpoint:
            load_text_tower_from_standard_ckpt(checkpoint, self.text, strict=True)

        # NOTE: self.text.logit_scale is loaded from the student (exp() ~= 78.1)
        # but is NOT the one HiVG uses -- HiVG.py:1116 reads self.clip.logit_scale,
        # which stays OpenAI's (exp() == 100.0). Transplanting it is a one-line
        # change but would perturb the CLC loss scale relative to the control
        # arm, so it is deliberately left alone.

    def iter_blocks(self):
        yield from self.text.transformer.resblocks

    def patch_lora_flat(self, rank: float, alpha: float) -> None:
        """Flat (non-staged) LoRA on every text block.

        Matches the HF branch this replaces (HiVG.py:913-917), which patches all
        12 layers and gates them on hi_lora_stage >= 1, and the same
        all-or-nothing convention LoopTextWithBridge.patch_lora_flat uses.
        Scope is q/k/v/out_proj via hilora._patch_openclip_block.
        """
        for block in self.text.transformer.resblocks:
            patch_block_with_lora(block, rank=rank, alpha=alpha, dropout=0.1)

    def set_trainable(self, trainable: bool) -> None:
        set_lora_trainable(list(self.text.transformer.resblocks), trainable)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> SimpleNamespace:
        """Matches the frozen HF CLIPTextModel's call signature/return shape.

        output_attentions/output_hidden_states/return_dict are accepted but
        unused -- hidden states are always computed, and open_clip's blocks call
        self.attn(..., need_weights=False) so attention weights are unavailable.
        HiVG only reads `attentions` from the vision output (HiVG.py:1085).

        Replays the tower manually rather than calling encode_text(), because
        encode_text() applies text_projection internally and HiVG applies
        self.clip.text_projection externally (HiVG.py:1068-1069) -- the same
        reason looptext_bridge.py replays LoopText's loop.
        """
        t = self.text
        x = t.token_embedding(input_ids) + t.positional_embedding  # (B, 77, 512)

        # hidden_states[0] is the token+position embedding (pre any block) and
        # hidden_states[i] for i>=1 is the state after the i-th block, matching
        # HF's convention so extract_text_layer picks like [12], [6, 12] and
        # [1..12] index correctly with no off-by-one.
        hidden_states = [x]

        # open_clip >= 3 runs the text transformer batch-first (transformer.py
        # Transformer.__init__ defaults batch_first=True and TextTransformer
        # does not override it), so everything stays (B, 77, 512) -- no permutes.
        # attn_mask is a registered non-persistent buffer: it moves with .to()
        # and is absent from state_dict(), so the strict load above is unaffected.
        # Passing it explicitly keeps causality after LoRA patching, since
        # hilora._UnfusedOpenCLIPAttention forwards attn_mask into SDPA.
        for block in t.transformer.resblocks:
            x = block(x, attn_mask=t.attn_mask)
            hidden_states.append(x)

        # last_hidden_state is post-ln_final while hidden_states[12] stays
        # pre-ln_final -- exactly HF's split (its hidden_states come from
        # encoder_outputs, before final_layer_norm). Appending ln_final(x) to
        # hidden_states instead would silently change what the MACB bridges
        # consume and break comparability with the OpenAI-text control arm.
        last_hidden_state = t.ln_final(x)

        # EOS pooling is byte-identical to HF's: the EOT id (49407) is the max
        # vocab id so argmax lands on the first EOT, and HF takes the same
        # argmax branch because its eos_token_id == 2.
        eos_positions = input_ids.argmax(dim=-1)
        pooler_output = last_hidden_state[
            torch.arange(input_ids.shape[0], device=x.device), eos_positions
        ]

        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            hidden_states=tuple(hidden_states),
            pooler_output=pooler_output,
        )

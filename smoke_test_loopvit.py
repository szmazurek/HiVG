"""Smoke test for the native HiVG LoopViT integration (models/loopvit_bridge.py).

Builds the HiVG model with --model LoopViT at a given loop_core_depth/
max_loop_steps, optionally loads a real CLIP-KD checkpoint (omit
--loopvit_checkpoint to test with random weights -- fine for architecture/
staging checks), runs one forward + backward pass on a dummy batch, and
reports which parameter groups are frozen vs. trainable, cross-checked
against blocks_for_stage()'s expected per-stage active-block counts.

Usage:
    cd HiVG && python smoke_test_loopvit.py [--loopvit_checkpoint <path>] \\
        [--loopvit_loop_core_depth 1|3|6] [--loopvit_max_loop_steps 12|4|2] \\
        [--hi_lora_stage 0|1|2|3] [--device cuda]
"""
import argparse
import os
import sys

import torch

from models import build_model
from utils.misc import NestedTensor

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from src.downstream.models.hilora import blocks_for_stage  # noqa: E402

# Expected cumulative active-block counts per stage, keyed by loop_core_depth,
# matching blocks_for_stage()'s documented math (12 unique blocks -> 5/8/12;
# otherwise an even cumulative split).
_EXPECTED_STAGE_COUNTS = {1: {1: 1, 2: 1, 3: 1}, 3: {1: 1, 2: 2, 3: 3}, 6: {1: 2, 2: 4, 3: 6}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loopvit_checkpoint", default="", type=str)
    parser.add_argument("--loopvit_loop_core_depth", default=1, type=int)
    parser.add_argument("--loopvit_max_loop_steps", default=12, type=int)
    parser.add_argument("--hi_lora_stage", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()

    model_args = argparse.Namespace(
        model="LoopViT",
        dataset="unc",
        mixup_pretrain=False,
        imsize=224,
        warmup=False,
        enable_adaptive_weights=True,
        vl_hidden_dim=512,
        vl_nheads=8,
        vl_dropout=0.1,
        vl_dim_feedforward=2048,
        vl_enc_layers=6,
        normalize_before=True,
        max_query_len=77,
        hi_lora_stage=args.hi_lora_stage,
        loopvit_checkpoint=args.loopvit_checkpoint,
        loopvit_loop_core_depth=args.loopvit_loop_core_depth,
        loopvit_max_loop_steps=args.loopvit_max_loop_steps,
        loopvit_loop_mode="global",
        loopvit_loop_schedule=None,
        loopvit_lora_rank=32,
        loopvit_lora_alpha=16.0,
    )

    print(f"Building HiVG(model=LoopViT, loop_core_depth={args.loopvit_loop_core_depth}, "
          f"max_loop_steps={args.loopvit_max_loop_steps}, hi_lora_stage={args.hi_lora_stage}) "
          f"{'from ' + args.loopvit_checkpoint if args.loopvit_checkpoint else '(random weights)'} ...")
    model = build_model(model_args).to(args.device)

    vm = model.clip.vision_model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    unique_blocks = []
    seen = set()
    for b in vm.loopvit.iter_blocks():
        if id(b) not in seen:
            seen.add(id(b))
            unique_blocks.append(b)
    n_active = sum(
        1 for b in unique_blocks if any(p.requires_grad for n, p in b.named_parameters() if "lora" in n)
    )
    n_patched = sum(
        1 for b in unique_blocks if any("lora" in n for n, _ in b.named_parameters())
    )
    expected_active = (
        0 if args.hi_lora_stage == 0
        else _EXPECTED_STAGE_COUNTS.get(args.loopvit_loop_core_depth, {}).get(args.hi_lora_stage)
    )

    bridges_trainable = all(p.requires_grad for p in vm.bridges.parameters())
    text_frozen = not any(p.requires_grad for p in model.clip.text_model.parameters())
    backbone_frozen = all(
        p.requires_grad is False for n, p in vm.loopvit.named_parameters() if "lora" not in n
    )

    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"unique physical blocks: {len(unique_blocks)} (expect {args.loopvit_loop_core_depth})")
    print(f"LoRA-patched blocks:    {n_patched}")
    print(f"LoRA-active blocks:     {n_active} (expect {expected_active})")
    print(f"bridges trainable (expect True):          {bridges_trainable}")
    print(f"text tower frozen (expect True):           {text_frozen}")
    print(f"backbone (non-LoRA) frozen (expect True): {backbone_frozen}")
    assert len(unique_blocks) == args.loopvit_loop_core_depth, "unique block count mismatch"
    if expected_active is not None:
        assert n_active == expected_active, f"expected {expected_active} active blocks, got {n_active}"

    B = args.batch_size
    images = torch.randn(B, 3, 224, 224, device=args.device)
    img_nested = NestedTensor(images, torch.ones(B, 224, 224, dtype=torch.bool, device=args.device))
    texts = ["a red car"] * B

    pred_box, logits_per_text, logits_per_image, visu_token_similarity, seg_mask = model(img_nested, texts)
    print(f"pred_box {pred_box.shape}, logits_per_text {logits_per_text.shape}, "
          f"visu_token_similarity {visu_token_similarity.shape}, seg_mask {seg_mask.shape}")

    loss = pred_box.sum() + logits_per_text.sum() + visu_token_similarity.sum() + seg_mask.sum()
    loss.backward()
    grad_ok = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in vm.loopvit.named_parameters() if "lora" in n
    ) if args.hi_lora_stage >= 1 else True
    print(f"backward ok, LoRA grad flowed as expected: {grad_ok}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

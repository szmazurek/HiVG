"""Smoke test for the HiVG LoopViT-LoopText integration (models/looptext_bridge.py).

Builds the HiVG model with --model LoopViT-LoopText at matched vision/text
loop_core_depth/steps, optionally loads a real joint CLIP-KD checkpoint (omit
--loopvit_looptext_checkpoint to test with random weights), runs one
forward + backward pass on a dummy batch, and reports which parameter groups
are frozen vs. trainable: vision LoRA staged (per blocks_for_stage), text LoRA
flat (all unique text blocks on/off together, no staging).

Usage:
    cd HiVG && python smoke_test_looptext.py [--loopvit_looptext_checkpoint <path>] \\
        [--loop_core_depth 1|3|6] [--max_loop_steps 12|4|2] \\
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

_EXPECTED_STAGE_COUNTS = {1: {1: 1, 2: 1, 3: 1}, 3: {1: 1, 2: 2, 3: 3}, 6: {1: 2, 2: 4, 3: 6}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loopvit_looptext_checkpoint", default="", type=str)
    parser.add_argument("--loop_core_depth", default=3, type=int,
                        help="shared vision/text loop_core_depth for this symmetric config")
    parser.add_argument("--max_loop_steps", default=4, type=int,
                        help="shared vision/text max_loop_steps for this symmetric config")
    parser.add_argument("--hi_lora_stage", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()

    model_args = argparse.Namespace(
        model="LoopViT-LoopText",
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
        loopvit_looptext_checkpoint=args.loopvit_looptext_checkpoint,
        loopvit_loop_core_depth=args.loop_core_depth,
        loopvit_max_loop_steps=args.max_loop_steps,
        loopvit_loop_mode="global",
        loopvit_loop_schedule=None,
        loopvit_lora_rank=32,
        loopvit_lora_alpha=16.0,
        embed_dim=512,
        text_width=512,
        text_loop_core_depth=args.loop_core_depth,
        text_max_loop_steps=args.max_loop_steps,
        text_num_heads=8,
        text_mlp_ratio=4.0,
        text_vocab_size=49408,
        looptext_lora_rank=32,
        looptext_lora_alpha=16.0,
    )

    print(f"Building HiVG(model=LoopViT-LoopText, loop_core_depth={args.loop_core_depth}, "
          f"max_loop_steps={args.max_loop_steps}, hi_lora_stage={args.hi_lora_stage}) "
          f"{'from ' + args.loopvit_looptext_checkpoint if args.loopvit_looptext_checkpoint else '(random weights)'} ...")
    model = build_model(model_args).to(args.device)

    vm = model.clip.vision_model
    tm = model.clip.text_model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Vision: staged, mirrors smoke_test_loopvit.py ---
    unique_v_blocks = []
    seen = set()
    for b in vm.loopvit.iter_blocks():
        if id(b) not in seen:
            seen.add(id(b))
            unique_v_blocks.append(b)
    n_v_active = sum(
        1 for b in unique_v_blocks if any(p.requires_grad for n, p in b.named_parameters() if "lora" in n)
    )
    expected_v_active = (
        0 if args.hi_lora_stage == 0
        else _EXPECTED_STAGE_COUNTS.get(args.loop_core_depth, {}).get(args.hi_lora_stage)
    )

    # --- Text: flat, all unique blocks on/off together ---
    text_blocks = list(tm.looptext.blocks)
    n_t_active = sum(
        1 for b in text_blocks if any(p.requires_grad for n, p in b.named_parameters() if "lora" in n)
    )
    expected_t_active = len(text_blocks) if args.hi_lora_stage >= 1 else 0

    bridges_trainable = all(p.requires_grad for p in vm.bridges.parameters())
    v_backbone_frozen = all(p.requires_grad is False for n, p in vm.loopvit.named_parameters() if "lora" not in n)
    t_backbone_frozen = all(p.requires_grad is False for n, p in tm.looptext.named_parameters() if "lora" not in n)
    proj_matches = torch.equal(model.clip.text_projection.weight.data, tm.looptext.text_projection.weight.data)

    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"[vision] unique blocks: {len(unique_v_blocks)} (expect {args.loop_core_depth})")
    print(f"[vision] LoRA-active blocks: {n_v_active} (expect {expected_v_active})")
    print(f"[text]   unique blocks: {len(text_blocks)} (expect {args.loop_core_depth})")
    print(f"[text]   LoRA-active blocks: {n_t_active} (expect {expected_t_active}, flat not staged)")
    print(f"bridges trainable (expect True):            {bridges_trainable}")
    print(f"vision backbone (non-LoRA) frozen (expect True): {v_backbone_frozen}")
    print(f"text backbone (non-LoRA) frozen (expect True):   {t_backbone_frozen}")
    print(f"text_projection transplanted correctly (expect True): {proj_matches}")

    assert len(unique_v_blocks) == args.loop_core_depth, "vision unique block count mismatch"
    assert len(text_blocks) == args.loop_core_depth, "text unique block count mismatch"
    if expected_v_active is not None:
        assert n_v_active == expected_v_active, f"expected {expected_v_active} active vision blocks, got {n_v_active}"
    assert n_t_active == expected_t_active, f"expected {expected_t_active} active text blocks (flat), got {n_t_active}"
    assert proj_matches, "text_projection was not transplanted from LoopText"

    B = args.batch_size
    images = torch.randn(B, 3, 224, 224, device=args.device)
    img_nested = NestedTensor(images, torch.ones(B, 224, 224, dtype=torch.bool, device=args.device))
    texts = ["a red car"] * B

    pred_box, logits_per_text, logits_per_image, visu_token_similarity, seg_mask = model(img_nested, texts)
    print(f"pred_box {pred_box.shape}, logits_per_text {logits_per_text.shape}, "
          f"visu_token_similarity {visu_token_similarity.shape}, seg_mask {seg_mask.shape}")

    loss = pred_box.sum() + logits_per_text.sum() + visu_token_similarity.sum() + seg_mask.sum()
    loss.backward()
    v_grad_ok = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in vm.loopvit.named_parameters() if "lora" in n
    ) if args.hi_lora_stage >= 1 else True
    t_grad_ok = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in tm.looptext.named_parameters() if "lora" in n
    ) if args.hi_lora_stage >= 1 else True
    print(f"backward ok, vision LoRA grad flowed as expected: {v_grad_ok}")
    print(f"backward ok, text LoRA grad flowed as expected:   {t_grad_ok}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

"""Smoke test for the native HiVG LoopViT integration (models/loopvit_bridge.py).

Builds the HiVG model with --model LoopViT, loads a real CLIP-KD checkpoint,
runs one forward + backward pass on a dummy batch, and reports which
parameter groups are frozen vs. trainable. No dataset/GPU required.

Usage:
    cd HiVG && python smoke_test_loopvit.py --loopvit_checkpoint <path-to-ckpt> [--hi_lora_stage 0|1] [--device cuda]
"""
import argparse

import torch

from models import build_model
from utils.misc import NestedTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loopvit_checkpoint", required=True, type=str)
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
        loopvit_max_loop_steps=12,
        loopvit_lora_rank=32,
        loopvit_lora_alpha=16.0,
    )

    print(f"Building HiVG(model=LoopViT, hi_lora_stage={args.hi_lora_stage}) "
          f"from {args.loopvit_checkpoint} ...")
    model = build_model(model_args).to(args.device)

    vm = model.clip.vision_model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_trainable = any(p.requires_grad for n, p in vm.loopvit.named_parameters() if "lora" in n)
    bridges_trainable = all(p.requires_grad for p in vm.bridges.parameters())
    text_frozen = not any(p.requires_grad for p in model.clip.text_model.parameters())
    backbone_frozen = all(
        p.requires_grad is False for n, p in vm.loopvit.named_parameters() if "lora" not in n
    )

    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"LoRA trainable (expect {args.hi_lora_stage >= 1}): {lora_trainable}")
    print(f"bridges trainable (expect True):                   {bridges_trainable}")
    print(f"text tower frozen (expect True):                   {text_frozen}")
    print(f"backbone (non-LoRA) frozen (expect True):          {backbone_frozen}")

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

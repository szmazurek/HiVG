#!/bin/bash
#
# data_root/split_root/checkpoint/output-root paths come from HiVG/.env
# (see _load_env.sh) -- copy HiVG/.env.example to HiVG/.env and fill it in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_load_env.sh"



CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set val    --output_dir $OUT_ROOT/mixup_pretraining_base/unc;
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set testA  --output_dir $OUT_ROOT/mixup_pretraining_base/unc;
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set testB  --output_dir $OUT_ROOT/mixup_pretraining_base/unc;
#

#
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc+          --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set val    --output_dir $OUT_ROOT/mixup_pretraining_base/unc+;
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc+          --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set testA  --output_dir $OUT_ROOT/mixup_pretraining_base/unc+;
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc+          --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set testB  --output_dir $OUT_ROOT/mixup_pretraining_base/unc+;

#
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset gref_umd      --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set val    --output_dir $OUT_ROOT/mixup_pretraining_base/gref_umd;
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset gref_umd      --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set test   --output_dir $OUT_ROOT/mixup_pretraining_base/gref_umd;
##

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset referit       --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set val    --output_dir $OUT_ROOT/mixup_pretraining_base/referit;
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset referit       --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set test   --output_dir $OUT_ROOT/mixup_pretraining_base/referit;
##
#
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset flickr        --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set val    --output_dir $OUT_ROOT/mixup_pretraining_base/flickr;
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset flickr        --imsize 224 --max_query_len 77 --normalize_before --use_mask_loss  --hi_lora_stage 3 --save_hilora_clip --mixup_pretrain --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/mixup_pretraining_base/mixup/best_checkpoint.pth --eval_set test   --output_dir $OUT_ROOT/mixup_pretraining_base/flickr;
##

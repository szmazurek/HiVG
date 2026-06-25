# RefCOCO (unc), single-dataset fine-tuning with the LoopViT backbone (--model
# LoopViT) instead of CLIP ViT-B/16. loop_core_depth=1 means there is only one
# physical transformer block (reused for every loop step), so there is no
# HiVG-style 3-stage hierarchical LoRA curriculum here -- just a warmup phase
# (no LoRA, bridges + new heads only) followed by one fine-tuning phase with
# the single LoRA adapter active (--hi_lora_stage 1). See models/HiVG.py's
# set_HiLoRA / models/loopvit_bridge.py for the implementation, and
# clip-kd-snn/configs/downstream/model/bvit_d1.yaml for why depth=1 collapses
# the hierarchical scheme to a single adapter.
#
# Reuses the same data_root/split_root our own pipeline already prepared:
#   data_root            = $SCRATCH/grounding_data            (data_root/other/images/mscoco/images/train2014)
#   split_root           = $SCRATCH/grounding_data/data        (split_root/unc/unc_{train,val,testA,testB}.pth)
#   loopvit_checkpoint   = a Lightning .ckpt from this repo's CLIP-KD training of LoopViT
#                          (bvit_d1.yaml architecture: loop_core_depth=1, max_loop_steps=12)
#   output_dir           = $SCRATCH/clip-kd-snn/HiVG_outputs/{output_lv0..lv1}/unc

DATA_ROOT=$SCRATCH/grounding_data
SPLIT_ROOT=$SCRATCH/grounding_data/data
LOOPVIT_CKPT=$SCRATCH/clip-kd-snn/outputs/16323316_kd_loopvit_vitb/checkpoints/best-epoch\=031-top1\=0.5075.ckpt
OUT_ROOT=$SCRATCH/clip-kd-snn/HiVG_outputs

echo -e "\n\n\n\n\n\n\n==================== unc LoopViT warmup (no LoRA) ==========================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 60 --batch_size 80 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate   --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --dataset unc      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --data_root $DATA_ROOT --split_root $SPLIT_ROOT --model LoopViT --loopvit_checkpoint $LOOPVIT_CKPT --hi_lora_stage 0 --output_dir $OUT_ROOT/output_lv0/unc --sup_type full;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --model LoopViT --hi_lora_stage 0 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_lv0/unc/best_checkpoint.pth --eval_set val      --output_dir $OUT_ROOT/output_lv0/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --model LoopViT --hi_lora_stage 0 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_lv0/unc/best_checkpoint.pth --eval_set testA    --output_dir $OUT_ROOT/output_lv0/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --model LoopViT --hi_lora_stage 0 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_lv0/unc/best_checkpoint.pth --eval_set testB    --output_dir $OUT_ROOT/output_lv0/unc;
##
# single LoRA-adapter fine-tuning phase
echo -e "\n\n\n\n\n\n\n==================== unc LoopViT stage 1 (single LoRA adapter) ==========================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 20  --batch_size 60 --lr 0.00010   --lr_scheduler cosine --aug_crop --aug_scale --aug_translate  --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --dataset unc      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT --model LoopViT --loopvit_checkpoint $LOOPVIT_CKPT --hi_lora_stage 1 --hi_lora_retrain $OUT_ROOT/output_lv0/unc/best_checkpoint.pth --output_dir $OUT_ROOT/output_lv1/unc      --sup_type full;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model LoopViT --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_lv1/unc/best_checkpoint.pth      --eval_set val    --output_dir $OUT_ROOT/output_lv1/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model LoopViT --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_lv1/unc/best_checkpoint.pth      --eval_set testA  --output_dir $OUT_ROOT/output_lv1/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model LoopViT --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_lv1/unc/best_checkpoint.pth      --eval_set testB  --output_dir $OUT_ROOT/output_lv1/unc;

# RefCOCO (unc), single-dataset fine-tuning + HiLoRA, using the real MSCOCO-debiased CLIP checkpoint.
#
# hivg_train.py/hivg_eval.py append a fresh timestamped subfolder to whatever
# --output_dir they're given on every invocation, so the checkpoint/clip file a
# stage just wrote never lives at the flat --output_dir path itself -- source
# _resolve_run_dir.sh and use resolve_run_dir() to find the real path after each
# step, instead of guessing a fixed one (that mismatch previously caused every
# stage past the first to crash with FileNotFoundError).
#
# GPU count / batch size are parametrized so the same script runs on fewer GPUs:
#   NPROC_PER_NODE (default 8) -- how many GPUs/processes to use
#   BATCH_MULT     (default 1) -- per-GPU batch size multiplier
# Keep NPROC_PER_NODE * BATCH_MULT == 8 to preserve the original global batch
# size (and thus not need to retune the learning rate): e.g. on 4 GPUs, set
# NPROC_PER_NODE=4 BATCH_MULT=2.
#
# data_root/split_root/checkpoints/output root all come from HiVG/.env (see _load_env.sh)
#  clip_model  = the debiased clip_b_ml_cascade_maskrcnn_model_224.pth checkpoint

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_resolve_run_dir.sh"
source "$SCRIPT_DIR/_load_env.sh"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
BATCH_MULT=${BATCH_MULT:-1}
CUDA_VIS=$(seq -s, 0 $((NPROC_PER_NODE - 1)))


echo "[gpu/batch config] NPROC_PER_NODE=$NPROC_PER_NODE BATCH_MULT=$BATCH_MULT CUDA_VISIBLE_DEVICES=$CUDA_VIS -- per-GPU batch sizes this run will use: $((80 * BATCH_MULT))/$((60 * BATCH_MULT))/$((60 * BATCH_MULT))/$((40 * BATCH_MULT)) (global batch = NPROC_PER_NODE * per-GPU batch, should match the 8-GPU/BATCH_MULT=1 baseline)"

echo -e "\n\n\n\n\n\n\n==================== unc warmup (hilora_real) ==========================="
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 60 --batch_size $((80 * BATCH_MULT)) --lr 0.00025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss --dataset unc --clip_model $CLIP_CKPT --data_root $DATA_ROOT --split_root $SPLIT_ROOT --output_dir $OUT_ROOT/output_v100/unc --sup_type full;
S0_TRAIN_DIR=$(resolve_run_dir "$OUT_ROOT/output_v100/unc" best_checkpoint.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((64 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S0_TRAIN_DIR/best_checkpoint.pth --eval_set val --output_dir $OUT_ROOT/output_v100/unc;
S0_CLIP_DIR=$(resolve_run_dir "$OUT_ROOT/output_v100/unc" clip_lora_stage_with_bridge.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((64 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S0_TRAIN_DIR/best_checkpoint.pth --eval_set testA --output_dir $OUT_ROOT/output_v100/unc;
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((64 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S0_TRAIN_DIR/best_checkpoint.pth --eval_set testB --output_dir $OUT_ROOT/output_v100/unc;

echo -e "\n\n\n\n\n\n\n==================== unc stage 1 (hilora_real) ==========================="
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 20 --batch_size $((60 * BATCH_MULT)) --lr 0.00010 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss --dataset unc --hi_lora_stage 1 --hi_lora_retrain $S0_TRAIN_DIR/best_checkpoint.pth --save_hilora_clip --hi_lora_clip $S0_CLIP_DIR/clip_lora_stage_with_bridge.pth --data_root $DATA_ROOT --split_root $SPLIT_ROOT --output_dir $OUT_ROOT/output_v101/unc --sup_type full;
S1_TRAIN_DIR=$(resolve_run_dir "$OUT_ROOT/output_v101/unc" best_checkpoint.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 1 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S1_TRAIN_DIR/best_checkpoint.pth --eval_set val --output_dir $OUT_ROOT/output_v101/unc;
S1_CLIP_DIR=$(resolve_run_dir "$OUT_ROOT/output_v101/unc" clip_lora_stage_with_bridge.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 1 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S1_TRAIN_DIR/best_checkpoint.pth --eval_set testA --output_dir $OUT_ROOT/output_v101/unc;
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 1 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S1_TRAIN_DIR/best_checkpoint.pth --eval_set testB --output_dir $OUT_ROOT/output_v101/unc;

echo -e "\n\n\n\n\n\n\n==================== unc stage 2 (hilora_real) ==========================="
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 20 --batch_size $((60 * BATCH_MULT)) --lr 0.00005 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss --dataset unc --hi_lora_stage 2 --hi_lora_retrain $S1_TRAIN_DIR/best_checkpoint.pth --save_hilora_clip --hi_lora_clip $S1_CLIP_DIR/clip_lora_stage_with_bridge.pth --data_root $DATA_ROOT --split_root $SPLIT_ROOT --output_dir $OUT_ROOT/output_v102/unc --sup_type full;
S2_TRAIN_DIR=$(resolve_run_dir "$OUT_ROOT/output_v102/unc" best_checkpoint.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 2 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S2_TRAIN_DIR/best_checkpoint.pth --eval_set val --output_dir $OUT_ROOT/output_v102/unc;
S2_CLIP_DIR=$(resolve_run_dir "$OUT_ROOT/output_v102/unc" clip_lora_stage_with_bridge.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 2 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S2_TRAIN_DIR/best_checkpoint.pth --eval_set testA --output_dir $OUT_ROOT/output_v102/unc;
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 2 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S2_TRAIN_DIR/best_checkpoint.pth --eval_set testB --output_dir $OUT_ROOT/output_v102/unc;

echo -e "\n\n\n\n\n\n\n==================== unc stage 3 (hilora_real) ==========================="
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 20 --batch_size $((40 * BATCH_MULT)) --lr 0.000025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss --dataset unc --hi_lora_stage 3 --hi_lora_retrain $S2_TRAIN_DIR/best_checkpoint.pth --save_hilora_clip --hi_lora_clip $S2_CLIP_DIR/clip_lora_stage_with_bridge.pth --data_root $DATA_ROOT --split_root $SPLIT_ROOT --output_dir $OUT_ROOT/output_v103/unc --sup_type full;
S3_TRAIN_DIR=$(resolve_run_dir "$OUT_ROOT/output_v103/unc" best_checkpoint.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 3 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S3_TRAIN_DIR/best_checkpoint.pth --eval_set val --output_dir $OUT_ROOT/output_v103/unc;
S3_CLIP_DIR=$(resolve_run_dir "$OUT_ROOT/output_v103/unc" clip_lora_stage_with_bridge.pth)
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 3 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S3_TRAIN_DIR/best_checkpoint.pth --eval_set testA --output_dir $OUT_ROOT/output_v103/unc;
CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size $((60 * BATCH_MULT)) --dataset unc --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --hi_lora_stage 3 --save_hilora_clip --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $S3_TRAIN_DIR/best_checkpoint.pth --eval_set testB --output_dir $OUT_ROOT/output_v103/unc;

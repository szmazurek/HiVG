# RefCOCO (unc), single-dataset fine-tuning with the standard (open_clip-native)
# ViT-B/16 backbone (--model StandardViT-Distilled) loaded from a clip-kd-snn
# CLIP-KD checkpoint -- the direct comparison arm for train_unc_loopvit.sh's
# LoopViT backbone: same bridges/recipe/text tower, only the vision backbone
# (and its checkpoint) differs, so val/testA/testB numbers are directly
# comparable between the two scripts.
#
# Full HiVG curriculum: warmup (no LoRA) + stages 1/2/3 (5/8/12 of the 12
# distinct blocks cumulatively LoRA-adapted, via src/downstream/models/
# hilora.py's blocks_for_stage) -- same per-stage epochs/batch_size/lr
# schedule as HiVG's own published curriculum (see train_unc_hilora_real.sh,
# which runs this same 4-phase schedule for the real OpenAI-CLIP backbone).
#
# Unlike that script, no --hi_lora_clip / --save_hilora_clip is needed here:
# those exist only because HiVG's non-LoopViT branches re-wrap self.clip in a
# fresh PeftModel every stage (peft's get_peft_model), requiring a separate
# same-stage-shaped reload for self.clip.model. StandardViT-Distilled never
# wraps self.clip in peft (its target_modules, q_proj/k_proj/v_proj/out_proj,
# don't match open_clip's fused nn.MultiheadAttention anyway -- see
# HiVG.py's set_HiLoRA), so the single --hi_lora_retrain full-model reload
# already covers vision_model/bridges/LoRA, exactly like train_unc_loopvit.sh.
#
# Reuses the same data_root/split_root our own pipeline already prepared:
#   data_root              = $SCRATCH/grounding_data            (data_root/other/images/mscoco/images/train2014)
#   split_root             = $SCRATCH/grounding_data/data        (split_root/unc/unc_{train,val,testA,testB}.pth)
#   standardvit_checkpoint = a Lightning .ckpt from this repo's CLIP-KD training of the standard
#                            (non-loop) ViT-B/16 student, e.g. kd_vit_b16_to_b16_cc3m12m
#   output_dir             = $SCRATCH/clip-kd-snn/HiVG_outputs/{output_sv0..sv3}/unc

DATA_ROOT=$SCRATCH/grounding_data
SPLIT_ROOT=$SCRATCH/grounding_data/data
STANDARDVIT_CKPT=$SCRATCH/clip-kd-snn/outputs/15938442_kd_vit_b16_to_b16_cc3m12m/checkpoints/best-epoch\=031-top1\=0.6518.ckpt
OUT_ROOT=$SCRATCH/clip-kd-snn/HiVG_outputs

echo -e "\n\n\n\n\n\n\n==================== unc StandardViT-Distilled warmup (no LoRA) ==========================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 60 --batch_size 80 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate   --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --dataset unc      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --data_root $DATA_ROOT --split_root $SPLIT_ROOT --model StandardViT-Distilled --standardvit_checkpoint $STANDARDVIT_CKPT --hi_lora_stage 0 --output_dir $OUT_ROOT/output_sv0/unc --sup_type full;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --model StandardViT-Distilled --hi_lora_stage 0 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv0/unc/best_checkpoint.pth --eval_set val      --output_dir $OUT_ROOT/output_sv0/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --model StandardViT-Distilled --hi_lora_stage 0 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv0/unc/best_checkpoint.pth --eval_set testA    --output_dir $OUT_ROOT/output_sv0/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 64  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss --model StandardViT-Distilled --hi_lora_stage 0 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv0/unc/best_checkpoint.pth --eval_set testB    --output_dir $OUT_ROOT/output_sv0/unc;
##
# stage 1 (5 of 12 blocks LoRA-adapted)
echo -e "\n\n\n\n\n\n\n==================== unc StandardViT-Distilled stage 1 ==========================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 20  --batch_size 60 --lr 0.00010   --lr_scheduler cosine --aug_crop --aug_scale --aug_translate  --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --dataset unc      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT --model StandardViT-Distilled --standardvit_checkpoint $STANDARDVIT_CKPT --hi_lora_stage 1 --hi_lora_retrain $OUT_ROOT/output_sv0/unc/best_checkpoint.pth --output_dir $OUT_ROOT/output_sv1/unc      --sup_type full;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv1/unc/best_checkpoint.pth      --eval_set val    --output_dir $OUT_ROOT/output_sv1/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv1/unc/best_checkpoint.pth      --eval_set testA  --output_dir $OUT_ROOT/output_sv1/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 1 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv1/unc/best_checkpoint.pth      --eval_set testB  --output_dir $OUT_ROOT/output_sv1/unc;
#
# stage 2 (8 of 12 blocks LoRA-adapted)
echo -e "\n\n\n\n\n\n\n==================== unc StandardViT-Distilled stage 2 ==========================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 20  --batch_size 60 --lr 0.00005   --lr_scheduler cosine --aug_crop --aug_scale --aug_translate  --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --dataset unc      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT --model StandardViT-Distilled --standardvit_checkpoint $STANDARDVIT_CKPT --hi_lora_stage 2 --hi_lora_retrain $OUT_ROOT/output_sv1/unc/best_checkpoint.pth --output_dir $OUT_ROOT/output_sv2/unc      --sup_type full;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv2/unc/best_checkpoint.pth      --eval_set val    --output_dir $OUT_ROOT/output_sv2/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv2/unc/best_checkpoint.pth      --eval_set testA  --output_dir $OUT_ROOT/output_sv2/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 2 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv2/unc/best_checkpoint.pth      --eval_set testB  --output_dir $OUT_ROOT/output_sv2/unc;
#
# stage 3 (12 of 12 blocks LoRA-adapted)
echo -e "\n\n\n\n\n\n\n==================== unc StandardViT-Distilled stage 3 ==========================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28887 --use_env hivg_train.py --num_workers 4 --epochs 20  --batch_size 40 --lr 0.000025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate  --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --dataset unc      --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss  --data_root $DATA_ROOT  --split_root $SPLIT_ROOT --model StandardViT-Distilled --standardvit_checkpoint $STANDARDVIT_CKPT --hi_lora_stage 3 --hi_lora_retrain $OUT_ROOT/output_sv2/unc/best_checkpoint.pth --output_dir $OUT_ROOT/output_sv3/unc      --sup_type full;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv3/unc/best_checkpoint.pth      --eval_set val    --output_dir $OUT_ROOT/output_sv3/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv3/unc/best_checkpoint.pth      --eval_set testA  --output_dir $OUT_ROOT/output_sv3/unc;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env hivg_eval.py --num_workers 2 --batch_size 60  --dataset unc           --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  --model StandardViT-Distilled --hi_lora_stage 3 --data_root $DATA_ROOT --split_root $SPLIT_ROOT --eval_model $OUT_ROOT/output_sv3/unc/best_checkpoint.pth      --eval_set testB  --output_dir $OUT_ROOT/output_sv3/unc;

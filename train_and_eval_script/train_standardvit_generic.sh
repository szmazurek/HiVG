# Generic single-dataset HiLoRA training+eval run for the standard
# (open_clip-native, non-recurrent) ViT-B/16 backbone (StandardViT-Distilled)
# -- generalizes train_unc_standardvit.sh (hardcoded to "unc") to any of the
# 5 grounding datasets. Always runs the full warmup -> stage1 -> stage2 ->
# stage3 HiLoRA curriculum (12 distinct blocks, no depth-based stage
# capping like the loopvit sweep -- see train_bvit_generic.sh).
#
# Required env vars:
#   DATASET            -- unc | unc+ | gref_umd | referit | flickr
#   STANDARDVIT_CKPT   -- path to the Lightning .ckpt to fine-tune (passed
#                         directly, no lookup convention -- this is meant for
#                         "I have one specific checkpoint I want to evaluate
#                         across all 5 datasets", unlike the bvit/b2vit sweep)
#
# Optional (defaults assume a 4-GPU box; see train_unc_standardvit.sh's
# header for the global-batch-preserving relationship between the two):
#   NPROC_PER_NODE (default 4), BATCH_MULT (default 2)
#   MASTER_PORT (default 28887, eval uses MASTER_PORT+1)
#   RUN_TAG (default "standardvit") -- namespaces $OUT_ROOT/<RUN_TAG>/<DATASET>/...,
#   set this to something checkpoint-specific if you'll evaluate more than
#   one standardvit checkpoint so their outputs don't land in the same dir.
#
# data_root/split_root/output root come from HiVG/.env (see _load_env.sh)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_resolve_run_dir.sh"
source "$SCRIPT_DIR/_load_env.sh"

: "${DATASET:?set DATASET (unc|unc+|gref_umd|referit|flickr)}"
: "${STANDARDVIT_CKPT:?set STANDARDVIT_CKPT (path to the checkpoint to fine-tune)}"

if [ ! -f "$STANDARDVIT_CKPT" ]; then
    echo "ERROR: STANDARDVIT_CKPT not found: $STANDARDVIT_CKPT" >&2
    exit 1
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
BATCH_MULT=${BATCH_MULT:-2}
MASTER_PORT=${MASTER_PORT:-28887}
CUDA_VIS=$(seq -s, 0 $((NPROC_PER_NODE - 1)))

case "$DATASET" in
    unc|"unc+")        EPOCHS_WARMUP=60; EPOCHS_STAGE=20; EVAL_SPLITS=(val testA testB) ;;
    gref_umd|referit)  EPOCHS_WARMUP=60; EPOCHS_STAGE=20; EVAL_SPLITS=(val test) ;;
    flickr)            EPOCHS_WARMUP=30; EPOCHS_STAGE=5;  EVAL_SPLITS=(val test) ;;
    *) echo "ERROR: unknown DATASET $DATASET (expected unc|unc+|gref_umd|referit|flickr)" >&2; exit 1 ;;
esac

RUN_TAG="${RUN_TAG:-standardvit}"
echo "[config] dataset=$DATASET standardvit_ckpt=$STANDARDVIT_CKPT run_tag=$RUN_TAG"
echo "[gpu/batch config] NPROC_PER_NODE=$NPROC_PER_NODE BATCH_MULT=$BATCH_MULT CUDA_VISIBLE_DEVICES=$CUDA_VIS"

# Fixed across all datasets (only warmup/stage epoch counts above vary, and
# only for flickr) -- see HiVG/train_and_eval_script/train_unc_standardvit.sh.
BATCH_BY_PHASE=(80 60 60 40)
LR_BY_PHASE=(0.00025 0.00010 0.00005 0.000025)

PREV_TRAIN_DIR=""
for PHASE in 0 1 2 3; do
    if [ "$PHASE" -eq 0 ]; then
        PHASE_NAME="warmup"
        EPOCHS=$EPOCHS_WARMUP
    else
        PHASE_NAME="stage${PHASE}"
        EPOCHS=$EPOCHS_STAGE
    fi
    BATCH=$(( ${BATCH_BY_PHASE[$PHASE]} * BATCH_MULT ))
    LR=${LR_BY_PHASE[$PHASE]}
    OUT_DIR="$OUT_ROOT/${RUN_TAG}/${DATASET}/output_v10${PHASE}"

    echo -e "\n\n\n==================== $DATASET $PHASE_NAME (standardvit) ==========================="

    RETRAIN_ARGS=()
    if [ -n "$PREV_TRAIN_DIR" ]; then
        RETRAIN_ARGS=(--hi_lora_retrain "$PREV_TRAIN_DIR/best_checkpoint.pth")
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch \
        --nproc_per_node=$NPROC_PER_NODE --master_port $MASTER_PORT --use_env hivg_train.py \
        --num_workers 4 --epochs "$EPOCHS" --batch_size "$BATCH" --lr "$LR" \
        --lr_scheduler cosine --aug_crop --aug_scale --aug_translate \
        --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before \
        --enable_adaptive_weights --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss \
        --dataset "$DATASET" --model StandardViT-Distilled --standardvit_checkpoint "$STANDARDVIT_CKPT" \
        --hi_lora_stage "$PHASE" "${RETRAIN_ARGS[@]}" \
        --data_root "$DATA_ROOT" --split_root "$SPLIT_ROOT" --output_dir "$OUT_DIR" --sup_type full

    TRAIN_DIR=$(resolve_run_dir "$OUT_DIR" best_checkpoint.pth)

    for SPLIT in "${EVAL_SPLITS[@]}"; do
        CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch \
            --nproc_per_node=$NPROC_PER_NODE --master_port $((MASTER_PORT + 1)) --use_env hivg_eval.py \
            --num_workers 2 --batch_size "$BATCH" --dataset "$DATASET" \
            --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before \
            --enable_adaptive_weights --use_mask_loss --model StandardViT-Distilled \
            --hi_lora_stage "$PHASE" \
            --data_root "$DATA_ROOT" --split_root "$SPLIT_ROOT" \
            --eval_model "$TRAIN_DIR/best_checkpoint.pth" --eval_set "$SPLIT" --output_dir "$OUT_DIR"
    done

    PREV_TRAIN_DIR=$TRAIN_DIR
done

echo "[done] standardvit / $DATASET -- final checkpoint: $PREV_TRAIN_DIR/best_checkpoint.pth"

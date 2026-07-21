# Generic single (dataset, backbone) HiLoRA training+eval run for the
# loopvit depth/iteration sweep (bvit_<D>b<S>i / b2vit_<D>b<S>i backbones,
# see _resolve_bvit_checkpoint.sh). Generalizes train_unc_loopvit.sh (which
# hardcoded depth=1/warmup+stage1 only) to any (depth, steps) pair and the
# full warmup -> stage1 -> stage2 -> stage3 HiLoRA curriculum, capped at
# min(depth, 3) stages -- past that, an additional stage can't unlock any
# new distinct block, so it would just repeat the previous one for nothing.
# "control" ViT-B is depth=12/steps=1 through this exact same script (a
# LoopViT with 12 distinct once-executed blocks is architecturally a
# standard ViT-B/16) -- no separate control codepath.
#
# Required env vars:
#   DATASET     -- unc | unc+ | gref_umd | referit | flickr
#   MODEL_KIND  -- bvit (--model LoopViT) | b2vit (--model LoopViT-LoopText)
#   DEPTH       -- loop_core_depth (distinct blocks), e.g. 1/2/3/4/6/12
#   STEPS       -- max_loop_steps (iterations per block); DEPTH*STEPS should be 12
#
# Optional (defaults assume a 4-GPU box; see train_unc_loopvit.sh's header
# for the global-batch-preserving relationship between the two):
#   NPROC_PER_NODE (default 4), BATCH_MULT (default 2)
#   MASTER_PORT (default 28887, eval uses MASTER_PORT+1) -- set this to a
#   distinct value per invocation if you're launching multiple combos
#   concurrently on the SAME node (e.g. several kfp pods sharing a host),
#   otherwise their torch.distributed rendezvous ports collide. Not needed
#   across separate nodes/SLURM jobs/pods with their own network namespace.
#
# data_root/split_root/output root/BVIT_CKPT_ROOT come from HiVG/.env (see _load_env.sh)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_resolve_run_dir.sh"
source "$SCRIPT_DIR/_resolve_bvit_checkpoint.sh"
source "$SCRIPT_DIR/_load_env.sh"

: "${DATASET:?set DATASET (unc|unc+|gref_umd|referit|flickr)}"
: "${MODEL_KIND:?set MODEL_KIND (bvit|b2vit)}"
: "${DEPTH:?set DEPTH (loop_core_depth)}"
: "${STEPS:?set STEPS (max_loop_steps)}"

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
BATCH_MULT=${BATCH_MULT:-2}
MASTER_PORT=${MASTER_PORT:-28887}
CUDA_VIS=$(seq -s, 0 $((NPROC_PER_NODE - 1)))

case "$MODEL_KIND" in
    bvit)  MODEL_ARG="LoopViT";          CKPT_FLAG="--loopvit_checkpoint" ;;
    b2vit) MODEL_ARG="LoopViT-LoopText"; CKPT_FLAG="--loopvit_looptext_checkpoint" ;;
    *) echo "ERROR: MODEL_KIND must be bvit or b2vit, got $MODEL_KIND" >&2; exit 1 ;;
esac

BACKBONE_CKPT=$(resolve_bvit_checkpoint "$MODEL_KIND" "$DEPTH" "$STEPS") || exit 1

TEXT_LOOP_ARGS=()
if [ "$MODEL_KIND" = "b2vit" ]; then
    # Matched vision/text recursion depth, per how these backbones were
    # pretrained (configs/model/loopvit_looptext_vitb.yaml overrides).
    TEXT_LOOP_ARGS=(--text_loop_core_depth "$DEPTH" --text_max_loop_steps "$STEPS")
fi

case "$DATASET" in
    unc|"unc+")        EPOCHS_WARMUP=60; EPOCHS_STAGE=20; EVAL_SPLITS=(val testA testB) ;;
    gref_umd|referit)  EPOCHS_WARMUP=60; EPOCHS_STAGE=20; EVAL_SPLITS=(val test) ;;
    flickr)            EPOCHS_WARMUP=30; EPOCHS_STAGE=5;  EVAL_SPLITS=(val test) ;;
    *) echo "ERROR: unknown DATASET $DATASET (expected unc|unc+|gref_umd|referit|flickr)" >&2; exit 1 ;;
esac

NUM_STAGES=$DEPTH
[ "$NUM_STAGES" -gt 3 ] && NUM_STAGES=3

RUN_TAG="${MODEL_KIND}_${DEPTH}b${STEPS}i"
echo "[config] dataset=$DATASET model_kind=$MODEL_KIND depth=$DEPTH steps=$STEPS num_stages=$NUM_STAGES"
echo "[config] backbone_ckpt=$BACKBONE_CKPT"
echo "[gpu/batch config] NPROC_PER_NODE=$NPROC_PER_NODE BATCH_MULT=$BATCH_MULT CUDA_VISIBLE_DEVICES=$CUDA_VIS"

# Fixed across all datasets/backbones (only warmup/stage epoch counts above
# vary, and only for flickr) -- see HiVG/train_and_eval_script/train_*_loopvit.sh.
BATCH_BY_PHASE=(80 60 60 40)
LR_BY_PHASE=(0.00025 0.00010 0.00005 0.000025)

PREV_TRAIN_DIR=""
for PHASE in $(seq 0 "$NUM_STAGES"); do
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

    echo -e "\n\n\n==================== $DATASET $PHASE_NAME ($RUN_TAG) ==========================="

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
        --dataset "$DATASET" --model "$MODEL_ARG" "$CKPT_FLAG" "$BACKBONE_CKPT" \
        --loopvit_loop_core_depth "$DEPTH" --loopvit_max_loop_steps "$STEPS" \
        "${TEXT_LOOP_ARGS[@]}" --hi_lora_stage "$PHASE" "${RETRAIN_ARGS[@]}" \
        --data_root "$DATA_ROOT" --split_root "$SPLIT_ROOT" --output_dir "$OUT_DIR" --sup_type full

    TRAIN_DIR=$(resolve_run_dir "$OUT_DIR" best_checkpoint.pth)

    for SPLIT in "${EVAL_SPLITS[@]}"; do
        CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch \
            --nproc_per_node=$NPROC_PER_NODE --master_port $((MASTER_PORT + 1)) --use_env hivg_eval.py \
            --num_workers 2 --batch_size "$BATCH" --dataset "$DATASET" \
            --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before \
            --enable_adaptive_weights --use_mask_loss --model "$MODEL_ARG" \
            --loopvit_loop_core_depth "$DEPTH" --loopvit_max_loop_steps "$STEPS" \
            "${TEXT_LOOP_ARGS[@]}" --hi_lora_stage "$PHASE" \
            --data_root "$DATA_ROOT" --split_root "$SPLIT_ROOT" \
            --eval_model "$TRAIN_DIR/best_checkpoint.pth" --eval_set "$SPLIT" --output_dir "$OUT_DIR"
    done

    PREV_TRAIN_DIR=$TRAIN_DIR
done

echo "[done] $RUN_TAG / $DATASET -- final checkpoint: $PREV_TRAIN_DIR/best_checkpoint.pth"

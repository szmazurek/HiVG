#!/bin/bash
# 1-epoch (warmup-stage only) smoke test of the real hivg_train.py training
# path across every (dataset, model checkpoint) combo, to catch a broken
# combo (bad path, crashing flag combination, OOM, ...) in minutes instead
# of hours into a real SLURM run.
#
# Run on an interactively-allocated node with 8 GPUs (e.g. via salloc/srun),
# from within HiVG/:
#   cd HiVG && bash smoke_test_all_training.sh
#
# Override epoch count or GPU/batch scaling the same way the real scripts do:
#   SMOKE_EPOCHS=2 NPROC_PER_NODE=4 BATCH_MULT=2 bash smoke_test_all_training.sh
#
# Prints a PASS/FAIL dataset x model_kind matrix at the end; per-combo logs
# land in $SMOKE_OUT_ROOT/logs/<dataset>_<kind>.log for debugging failures.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/train_and_eval_script/_resolve_run_dir.sh"
source "$SCRIPT_DIR/train_and_eval_script/_load_env.sh"

SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_MULT="${BATCH_MULT:-1}"
CUDA_VIS=$(seq -s, 0 $((NPROC_PER_NODE - 1)))

# DATA_ROOT, SPLIT_ROOT, STANDARDVIT_CKPT, LOOPVIT_CKPT, SMOKE_OUT_ROOT come
# from HiVG/.env (see _load_env.sh) -- copy HiVG/.env.example to HiVG/.env
# and fill in your paths.
OUT_ROOT="$SMOKE_OUT_ROOT"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

DATASETS="unc unc+ gref_umd referit flickr"
MODEL_KINDS="oaiclip standardvit loopvit"

declare -A RESULT

run_one() {
    local dataset="$1" kind="$2"
    local tag="${dataset}_${kind}"
    local out_dir="$OUT_ROOT/$tag"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    local log="$LOG_DIR/${tag}.log"

    local extra=""
    case "$kind" in
        oaiclip)     extra="" ;;
        standardvit) extra="--model StandardViT-Distilled --standardvit_checkpoint $STANDARDVIT_CKPT --hi_lora_stage 0" ;;
        loopvit)     extra="--model LoopViT --loopvit_checkpoint $LOOPVIT_CKPT --hi_lora_stage 0" ;;
    esac

    echo "=== [$tag] training, $SMOKE_EPOCHS epoch(s), $NPROC_PER_NODE GPU(s), batch $((80 * BATCH_MULT)) ==="
    CUDA_VISIBLE_DEVICES=$CUDA_VIS python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port 28887 \
        --use_env hivg_train.py --num_workers 4 --epochs "$SMOKE_EPOCHS" --batch_size $((80 * BATCH_MULT)) --lr 0.00025 --lr_scheduler cosine \
        --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --imsize 224 --max_query_len 77 --normalize_before \
        --enable_adaptive_weights --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss \
        --dataset "$dataset" $extra --data_root "$DATA_ROOT" --split_root "$SPLIT_ROOT" \
        --output_dir "$out_dir" --sup_type full > "$log" 2>&1
    local rc=$?

    if [ $rc -eq 0 ] && ls "$out_dir"/*/best_checkpoint.pth >/dev/null 2>&1; then
        RESULT["$tag"]="PASS"
    else
        RESULT["$tag"]="FAIL(rc=$rc)"
    fi
    echo "  -> ${RESULT[$tag]}  (log: $log)"
}

for dataset in $DATASETS; do
    for kind in $MODEL_KINDS; do
        run_one "$dataset" "$kind"
    done
done

echo
echo "=================== SMOKE TEST MATRIX (train, ${SMOKE_EPOCHS} epoch) ==================="
printf "%-12s" "dataset"
for kind in $MODEL_KINDS; do printf "%-16s" "$kind"; done
echo
for dataset in $DATASETS; do
    printf "%-12s" "$dataset"
    for kind in $MODEL_KINDS; do
        printf "%-16s" "${RESULT[${dataset}_${kind}]}"
    done
    echo
done
echo "============================================================================"
echo "Logs: $LOG_DIR"

for v in "${RESULT[@]}"; do
    if [[ "$v" != "PASS" ]]; then
        exit 1
    fi
done

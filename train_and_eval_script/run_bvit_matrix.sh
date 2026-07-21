# Matrix driver: runs train_bvit_generic.sh across every (dataset, backbone)
# combination in the loopvit depth/iteration sweep -- sequentially, since
# each combo uses all $NPROC_PER_NODE GPUs via DDP (no point parallelizing
# combos across the same 4 GPUs).
#
# Backbone configs: (depth, steps) pairs with depth*steps==12, for both
# LoopViT (bvit) and LoopViT-LoopText (b2vit) -- 12 backbones total,
# including bvit_12b1i / b2vit_12b1i as the "control" ViT-B entries
# (depth=12, steps=1 -- mathematically a standard, non-recurrent ViT-B/16,
# trained through the same LoopViT code path -- see
# _resolve_bvit_checkpoint.sh's header). No separate control codepath.
#
# Usage (CLI flags -- for one job/pod = one architecture across all datasets,
# so separate SLURM jobs / kfp pods can each claim a disjoint slice of the
# matrix and run concurrently on their own GPU allocation):
#   run_bvit_matrix.sh --arch bvit --depth 1 --iters 12
#   run_bvit_matrix.sh --arch b2vit --depth 2 --iters 6 --datasets "unc gref_umd"
#   run_bvit_matrix.sh --dry-run --arch bvit --depth 3 --iters 4
#   run_bvit_matrix.sh --help
#
# Usage (env-var interface -- what the flags above translate into; still
# works directly if you prefer it):
#   bash run_bvit_matrix.sh                                    # full 5-dataset x 12-backbone matrix
#   DATASETS="unc gref_umd" CONFIGS="1,12 3,4" KINDS="bvit" \
#       bash run_bvit_matrix.sh                                # restrict to a subset
#   DRY_RUN=1 bash run_bvit_matrix.sh                           # print combos + resolved checkpoints, run nothing
#
# Resumable at combo granularity: a combo is skipped if its manifest marker
# ($OUT_ROOT/.matrix_manifest/<kind>_<depth>b<steps>i__<dataset>.done)
# already exists -- delete the marker (or just the combo's own output dir,
# then the marker) to force a rerun. This is NOT phase-level: an
# interrupted combo restarts from its warmup phase on the next matrix run,
# it does not resume mid-curriculum. Markers/logs are namespaced per
# (kind, depth, steps, dataset), so concurrent invocations targeting
# different archs/depths/datasets don't collide with each other.
#
# NOTE for concurrent runs sharing a node: --master-port (default 28887,
# eval uses +1) is fixed per invocation -- pass a distinct --master-port to
# any jobs that might land on the same node at the same time, or they'll
# collide on torch.distributed's rendezvous port.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_load_env.sh"

usage() {
    sed -n '1,/^set -uo pipefail/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
}

ARCH_LIST=()
DEPTH_OVERRIDE=""
ITERS_OVERRIDE=""
DATASETS_OVERRIDE="${DATASETS:-}"
MASTER_PORT_OVERRIDE="${MASTER_PORT:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)            ARCH_LIST+=("$2"); shift 2 ;;
        --depth)           DEPTH_OVERRIDE="$2"; shift 2 ;;
        --iters)           ITERS_OVERRIDE="$2"; shift 2 ;;
        --datasets)        DATASETS_OVERRIDE="$2"; shift 2 ;;
        --nproc-per-node)  NPROC_PER_NODE="$2"; shift 2 ;;
        --batch-mult)      BATCH_MULT="$2"; shift 2 ;;
        --master-port)     MASTER_PORT_OVERRIDE="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)" >&2; exit 1 ;;
    esac
done

if [ -n "$DEPTH_OVERRIDE" ] || [ -n "$ITERS_OVERRIDE" ]; then
    if [ -z "$DEPTH_OVERRIDE" ] || [ -z "$ITERS_OVERRIDE" ]; then
        echo "ERROR: --depth and --iters must be given together" >&2
        exit 1
    fi
    if [ "$((DEPTH_OVERRIDE * ITERS_OVERRIDE))" -ne 12 ]; then
        echo "WARNING: --depth $DEPTH_OVERRIDE --iters $ITERS_OVERRIDE doesn't multiply to 12" \
             "-- every backbone in bvit_configs follows that convention, double-check the checkpoint exists" >&2
    fi
    CONFIGS="${DEPTH_OVERRIDE},${ITERS_OVERRIDE}"
fi

for A in "${ARCH_LIST[@]:-}"; do
    case "$A" in
        bvit|b2vit) ;;
        *) echo "ERROR: --arch must be bvit or b2vit, got $A" >&2; exit 1 ;;
    esac
done
[ ${#ARCH_LIST[@]} -gt 0 ] && KINDS="${ARCH_LIST[*]}"

DATASETS="$DATASETS_OVERRIDE"
MASTER_PORT="${MASTER_PORT_OVERRIDE:-28887}"

ALL_DATASETS=(unc "unc+" gref_umd referit flickr)
ALL_CONFIGS=("1,12" "2,6" "3,4" "4,3" "6,2" "12,1")
ALL_KINDS=(bvit b2vit)

read -r -a DATASETS_ARR <<< "${DATASETS:-${ALL_DATASETS[*]}}"
read -r -a CONFIGS_ARR  <<< "${CONFIGS:-${ALL_CONFIGS[*]}}"
read -r -a KINDS_ARR    <<< "${KINDS:-${ALL_KINDS[*]}}"

MANIFEST_DIR="$OUT_ROOT/.matrix_manifest"
mkdir -p "$MANIFEST_DIR"

echo "[matrix] datasets: ${DATASETS_ARR[*]}"
echo "[matrix] configs (depth,steps): ${CONFIGS_ARR[*]}"
echo "[matrix] kinds: ${KINDS_ARR[*]}"
echo "[matrix] total combos: $(( ${#DATASETS_ARR[@]} * ${#CONFIGS_ARR[@]} * ${#KINDS_ARR[@]} ))"

for KIND in "${KINDS_ARR[@]}"; do
    for CFG in "${CONFIGS_ARR[@]}"; do
        DEPTH=${CFG%,*}
        STEPS=${CFG#*,}
        for DATASET in "${DATASETS_ARR[@]}"; do
            RUN_TAG="${KIND}_${DEPTH}b${STEPS}i"
            MARKER="$MANIFEST_DIR/${RUN_TAG}__${DATASET}.done"

            if [ -f "$MARKER" ]; then
                echo "[skip] $RUN_TAG / $DATASET (marker present: $MARKER)"
                continue
            fi

            if [ "${DRY_RUN:-0}" = "1" ]; then
                source "$SCRIPT_DIR/_resolve_bvit_checkpoint.sh"
                if CKPT=$(resolve_bvit_checkpoint "$KIND" "$DEPTH" "$STEPS"); then
                    echo "[dry-run] $RUN_TAG / $DATASET -- ckpt=$CKPT"
                else
                    echo "[dry-run] $RUN_TAG / $DATASET -- CHECKPOINT MISSING, would fail"
                fi
                continue
            fi

            echo "[run] $RUN_TAG / $DATASET"
            DATASET="$DATASET" MODEL_KIND="$KIND" DEPTH="$DEPTH" STEPS="$STEPS" \
                NPROC_PER_NODE="${NPROC_PER_NODE:-4}" BATCH_MULT="${BATCH_MULT:-2}" \
                MASTER_PORT="$MASTER_PORT" \
                bash "$SCRIPT_DIR/train_bvit_generic.sh" \
                2>&1 | tee -a "$MANIFEST_DIR/${RUN_TAG}__${DATASET}.log"

            if [ "${PIPESTATUS[0]}" -eq 0 ]; then
                touch "$MARKER"
                echo "[done] $RUN_TAG / $DATASET"
            else
                echo "[FAILED] $RUN_TAG / $DATASET -- see $MANIFEST_DIR/${RUN_TAG}__${DATASET}.log" >&2
            fi
        done
    done
done

echo "[matrix] all combos processed."

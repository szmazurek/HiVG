# Runs train_standardvit_generic.sh (control ViT-B/16, StandardViT-Distilled)
# across all 5 grounding datasets for one given checkpoint -- sequentially,
# since each dataset run uses all $NPROC_PER_NODE GPUs via DDP.
#
# Usage:
#   run_standardvit_matrix.sh --checkpoint /path/to/standardvit.ckpt
#   run_standardvit_matrix.sh --checkpoint /path/to/standardvit.ckpt --datasets "unc gref_umd"
#   run_standardvit_matrix.sh --checkpoint /path/to/standardvit.ckpt --dry-run
#   run_standardvit_matrix.sh --help
#
# --tag namespaces $OUT_ROOT/<tag>/<dataset>/... -- defaults to the
# checkpoint's grandparent dir name (e.g. .../15938442_kd_vit_b16_to_b16_cc3m12m
# /checkpoints/best-....ckpt -> tag "15938442_kd_vit_b16_to_b16_cc3m12m"), so
# evaluating a second, different standardvit checkpoint later doesn't
# overwrite/mix with the first one's outputs. Pass --tag explicitly to
# control this yourself.
#
# Resumable at dataset granularity: a dataset is skipped if its manifest
# marker ($OUT_ROOT/.matrix_manifest/<tag>__<dataset>.done) already exists --
# delete the marker to force a rerun. This is NOT phase-level: an
# interrupted dataset restarts from its warmup phase on the next run.
#
# NOTE for concurrent runs sharing a node: --master-port (default 28887,
# eval uses +1) is fixed per invocation -- pass a distinct --master-port to
# any jobs that might land on the same node at the same time.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_load_env.sh"

usage() {
    sed -n '1,/^set -uo pipefail/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
}

CHECKPOINT=""
TAG=""
DATASETS_OVERRIDE="${DATASETS:-}"
MASTER_PORT_OVERRIDE="${MASTER_PORT:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint)      CHECKPOINT="$2"; shift 2 ;;
        --tag)             TAG="$2"; shift 2 ;;
        --datasets)        DATASETS_OVERRIDE="$2"; shift 2 ;;
        --nproc-per-node)  NPROC_PER_NODE="$2"; shift 2 ;;
        --batch-mult)      BATCH_MULT="$2"; shift 2 ;;
        --master-port)     MASTER_PORT_OVERRIDE="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)" >&2; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: --checkpoint is required" >&2
    usage >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [ -z "$TAG" ]; then
    TAG="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")"
fi

DATASETS="$DATASETS_OVERRIDE"
MASTER_PORT="${MASTER_PORT_OVERRIDE:-28887}"

ALL_DATASETS=(unc "unc+" gref_umd referit flickr)
read -r -a DATASETS_ARR <<< "${DATASETS:-${ALL_DATASETS[*]}}"

MANIFEST_DIR="$OUT_ROOT/.matrix_manifest"
mkdir -p "$MANIFEST_DIR"

echo "[matrix] checkpoint: $CHECKPOINT"
echo "[matrix] tag: $TAG"
echo "[matrix] datasets: ${DATASETS_ARR[*]}"

for DATASET in "${DATASETS_ARR[@]}"; do
    MARKER="$MANIFEST_DIR/${TAG}__${DATASET}.done"

    if [ -f "$MARKER" ]; then
        echo "[skip] $TAG / $DATASET (marker present: $MARKER)"
        continue
    fi

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] $TAG / $DATASET -- ckpt=$CHECKPOINT"
        continue
    fi

    echo "[run] $TAG / $DATASET"
    DATASET="$DATASET" STANDARDVIT_CKPT="$CHECKPOINT" RUN_TAG="$TAG" \
        NPROC_PER_NODE="${NPROC_PER_NODE:-4}" BATCH_MULT="${BATCH_MULT:-2}" \
        MASTER_PORT="$MASTER_PORT" \
        bash "$SCRIPT_DIR/train_standardvit_generic.sh" \
        2>&1 | tee -a "$MANIFEST_DIR/${TAG}__${DATASET}.log"

    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        touch "$MARKER"
        echo "[done] $TAG / $DATASET"
    else
        echo "[FAILED] $TAG / $DATASET -- see $MANIFEST_DIR/${TAG}__${DATASET}.log" >&2
    fi
done

echo "[matrix] all datasets processed."

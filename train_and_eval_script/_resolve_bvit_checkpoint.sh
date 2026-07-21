# Sourced by train_bvit_generic.sh / run_bvit_matrix.sh.
#
# Backbone checkpoints for the loopvit depth/iteration sweep live under
# $BVIT_CKPT_ROOT (see HiVG/.env), one directory per (arch, depth, steps):
#   $BVIT_CKPT_ROOT/bvit_<depth>b<steps>i/checkpoints/   -- LoopViT   (--model LoopViT)
#   $BVIT_CKPT_ROOT/b2vit_<depth>b<steps>i/checkpoints/  -- LoopViT-LoopText (--model LoopViT-LoopText)
# e.g. bvit_3b4i = loop_core_depth=3, max_loop_steps=4 (depth*steps==12 always).
# "control" ViT-B is just bvit_12b1i (depth=12, steps=1 -- mathematically a
# standard, non-recurrent ViT-B/16, trained through the same LoopViT code
# path) -- no special-casing needed anywhere in the launcher for it.
#
# Each checkpoint dir holds several Lightning ModelCheckpoint outputs
# (save_top_k=3 "best-epoch=NNN-top1=X.XXXX.ckpt" files plus last.ckpt); we
# always use the last/final-epoch weights, not the best-val one. A dir
# resumed mid-training gets last-v1.ckpt (Lightning doesn't overwrite an
# existing last.ckpt across a resume) written AFTER last.ckpt, so last.ckpt
# alone can be stale -- pick whichever last*.ckpt sorts highest
# (last.ckpt < last-v1.ckpt < last-v2.ckpt ...).
resolve_bvit_checkpoint() {
    local kind="$1" depth="$2" steps="$3"
    local tag="${depth}b${steps}i"
    local ckpt_dir="$BVIT_CKPT_ROOT/${kind}_${tag}/checkpoints"
    if [ ! -d "$ckpt_dir" ]; then
        echo "ERROR: no checkpoint dir at $ckpt_dir" >&2
        return 1
    fi
    local last
    last=$(ls -v "$ckpt_dir"/last*.ckpt 2>/dev/null | tail -1)
    if [ -z "$last" ]; then
        echo "ERROR: no last*.ckpt found in $ckpt_dir" >&2
        return 1
    fi
    echo "$last"
}

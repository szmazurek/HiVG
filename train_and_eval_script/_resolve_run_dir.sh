# Sourced by the train_*.sh scripts in this directory.
#
# hivg_train.py/hivg_eval.py append a fresh datetime.now()-based subfolder to
# whatever --output_dir they're given on every single invocation (see
# hivg_train.py/hivg_eval.py main()). That means the checkpoint/clip file a
# stage just wrote never lives at the flat --output_dir path we passed in --
# it's one level deeper, in a subfolder we can't predict in advance. This
# resolves that actual subfolder after the fact by finding the most recently
# written file with the expected name under the given root, so the next
# stage can be pointed at the real path instead of a guessed fixed one.
resolve_run_dir() {
    # $1 = the --output_dir root we passed to hivg_train.py/hivg_eval.py
    # $2 = filename to look for inside its timestamped subfolders (e.g. best_checkpoint.pth)
    local root="$1" fname="$2" f
    f=$(ls -t "$root"/*/"$fname" 2>/dev/null | head -1)
    if [ -z "$f" ]; then
        echo "ERROR: no $fname found under any $root/<timestamp>/ -- previous step didn't produce one, aborting." >&2
        exit 1
    fi
    dirname "$f"
}

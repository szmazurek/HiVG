# Sourced by the train_*.sh/eval_*.sh scripts in this directory and by
# smoke_test_all_training.sh to load machine-specific paths (data/split
# roots, checkpoints, output roots) from HiVG/.env, so none of those scripts
# hardcode a path themselves. First time setup:
#   cp HiVG/.env.example HiVG/.env   # then edit HiVG/.env
_ENV_LOADER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_HIVG_ENV_FILE="$_ENV_LOADER_DIR/../.env"
if [ ! -f "$_HIVG_ENV_FILE" ]; then
    echo "ERROR: $_HIVG_ENV_FILE not found -- copy HiVG/.env.example to HiVG/.env and fill in your paths." >&2
    exit 1
fi
set -a
source "$_HIVG_ENV_FILE"
set +a

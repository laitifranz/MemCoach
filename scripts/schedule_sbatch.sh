#!/usr/bin/env bash
# Author: @laitifranz
set -o errexit
set -o nounset
set -o pipefail


if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

print_usage() {
    cat <<'USAGE' >&2
Usage:
  bash scripts/schedule_sbatch.sh <config> [KEY=VALUE ...] <command ...>

Examples:
  bash scripts/schedule_sbatch.sh leonardo_gpu 'uv run runner.py --config_path .../runner.yaml'
  bash scripts/schedule_sbatch.sh leonardo_gpu NUM_JOBS=8 SLURM_MEM=128GB \
       'uv run runner.py --config_path ... --shard_id ${SLURM_ARRAY_TASK_ID}'

Config variables loaded from scripts/slurm_configs/<config>.sh (overridable via KEY=VALUE):
  SLURM_PARTITION   → #SBATCH --partition
  SLURM_ACCOUNT     → #SBATCH --account
  SLURM_NODES       → #SBATCH --nodes
  SLURM_TASKS       → #SBATCH --ntasks-per-node
  SLURM_CPU         → #SBATCH --cpus-per-task
  SLURM_MEM         → #SBATCH --mem
  SLURM_TIME        → #SBATCH --time
  SLURM_NAME        → #SBATCH --job-name and log directory name
  SLURM_GPU         → optional #SBATCH --gres=gpu:<count> (omit or set to 0 for CPU jobs)
  SLURM_QOS         → optional #SBATCH --qos
  SLURM_SIGNAL      → optional #SBATCH --signal=<sig>[@time]
  MAIL_TYPE         → optional #SBATCH --mail-type
  MAIL_USER         → optional #SBATCH --mail-user
  NUM_JOBS          → optional #SBATCH --array=0-(NUM_JOBS-1), switches log suffix to %A_%a
  BEFORE_CODE_BLOCK → array of shell lines prepended before your command
  AFTER_CODE_BLOCK  → array of shell lines appended after your command

Runtime details:
  • Logs: logs/slurm/${SLURM_NAME}/%j.out|err (arrays use %A_%a)
  • User command: everything after overrides; runs inside the sbatch script verbatim
  • Overrides: specify as KEY=VALUE before the command; they eval into the sourced config
      e.g. SLURM_MEM=64GB NUM_JOBS=4 SLURM_SIGNAL="TERM@120"

Pass -h or --help to print this message.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

if [[ -t 1 ]]; then
    RED="\033[31m"
    GREEN="\033[32m"
    YELLOW="\033[33m"
    BLUE="\033[34m"
    BOLD="\033[1m"
    RESET="\033[0m"
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    BOLD=""
    RESET=""
fi

# cd to the root of the project
cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

# helper function to emit a block of code
emit_hook_block() {
    local -n hook_ref=$1   # nameref to BEFORE_CODE_BLOCK or AFTER_CODE_BLOCK
    local lines=()
    for entry in "${hook_ref[@]}"; do
        lines+=("$entry")
    done
    printf '%s\n' "${lines[@]}"
}

# read config name from first argument 
CONFIG_NAME=${1:-}
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Usage: $0 <config> <command...>" >&2
    echo "Available configs:" >&2
    ls ./scripts/slurm_configs/ | sed 's/\.sh$//' | sed 's/^/    /' >&2
    echo "Example: bash schedule_sbatch.sh leonardo_gpu 'uv run runner.py --config_path <config_path>'" >&2
    exit 1
fi

echo -e "${BOLD}${BLUE}[INFO] Sourcing config${RESET}: $CONFIG_NAME" >&2
source "./scripts/slurm_configs/$CONFIG_NAME.sh"
shift

readarray -t VALID_OVERRIDE_KEYS <<'EOF'
SLURM_PARTITION
SLURM_ACCOUNT
SLURM_NODES
SLURM_TASKS
SLURM_CPU
SLURM_MEM
SLURM_TIME
SLURM_NAME
SLURM_GPU
SLURM_QOS
SLURM_SIGNAL
MAIL_TYPE
MAIL_USER
NUM_JOBS
BEFORE_CODE_BLOCK
AFTER_CODE_BLOCK
EOF

is_valid_override_key() {
    local candidate=$1
    for allowed in "${VALID_OVERRIDE_KEYS[@]}"; do
        if [[ "$allowed" == "$candidate" ]]; then
            return 0
        fi
    done
    return 1
}

# override config values with command line arguments
while [[ $# -gt 0 && "$1" == *=* ]]; do
    override_pair="$1"
    override_key="${override_pair%%=*}"
    if ! is_valid_override_key "$override_key"; then
        echo -e "${BOLD}${RED}[ERROR] Unknown override key${RESET}: $override_key" >&2
        echo -e "${BOLD}${YELLOW}[INFO] Valid keys${RESET}: ${VALID_OVERRIDE_KEYS[*]}" >&2
        exit 1
    fi
    echo -e "${BOLD}${YELLOW}[WARNING] Overriding config value${RESET}: $override_pair" >&2
    eval "$override_pair"   # e.g. SLURM_MEM=64GB overrides the sourced value
    shift
done

if [[ $# -eq 0 ]]; then
    echo -e "${BOLD}${RED}[ERROR] No command provided${RESET}" >&2
    exit 1
fi

# user command is the rest of the arguments
USER_CMD=("$@")
echo -e "${BOLD}${BLUE}[INFO] User command${RESET}: ${USER_CMD[@]}" >&2

# default: no array unless NUM_JOBS is set to a positive integer
ARRAY_DIRECTIVE=""
if [[ -n "${NUM_JOBS:-}" ]]; then
    ARRAY_DIRECTIVE="#SBATCH --array=0-$((NUM_JOBS-1))"
fi

# default: no GPUs unless SLURM_GPU > 0
GRES_DIRECTIVE=""
if [[ "${SLURM_GPU:-0}" != "0" ]]; then
    GRES_DIRECTIVE="#SBATCH --gres=gpu:${SLURM_GPU}"
fi

QOS_DIRECTIVE=""
if [[ -n "${SLURM_QOS:-}" ]]; then
    QOS_DIRECTIVE="#SBATCH --qos=${SLURM_QOS}"
fi

MAIL_TYPE_DIRECTIVE=""
if [[ -n "${MAIL_TYPE:-}" ]]; then
    MAIL_TYPE_DIRECTIVE="#SBATCH --mail-type=${MAIL_TYPE}"
fi

MAIL_USER_DIRECTIVE=""
if [[ -n "${MAIL_USER:-}" ]]; then
    MAIL_USER_DIRECTIVE="#SBATCH --mail-user=${MAIL_USER}"
fi

SIGNAL_DIRECTIVE=""
if [[ -n "${SLURM_SIGNAL:-}" ]]; then
    SIGNAL_DIRECTIVE="#SBATCH --signal=${SLURM_SIGNAL}"
fi

# build pre and post blocks from config
PRE_BLOCK=""
if declare -p BEFORE_CODE_BLOCK &>/dev/null; then
    PRE_BLOCK="$(emit_hook_block BEFORE_CODE_BLOCK)"
fi

POST_BLOCK=""
if declare -p AFTER_CODE_BLOCK &>/dev/null; then
    POST_BLOCK="$(emit_hook_block AFTER_CODE_BLOCK)"
fi

TASK_SUFFIX="%j"   # default: job ID only
if [[ -n "${NUM_JOBS:-}" ]]; then
    TASK_SUFFIX="%A_%a"   # array master + task ID when arrays are enabled
fi

LOG_ROOT="./logs/slurm/${SLURM_NAME}"

SLURM_OUTPUT="${LOG_ROOT}/${TASK_SUFFIX}.out"
SLURM_ERROR="${LOG_ROOT}/${TASK_SUFFIX}.err"

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --nodes=$SLURM_NODES
$ARRAY_DIRECTIVE
#SBATCH --ntasks-per-node=$SLURM_TASKS
#SBATCH --cpus-per-task=$SLURM_CPU
$GRES_DIRECTIVE
#SBATCH --mem=$SLURM_MEM
$SIGNAL_DIRECTIVE
#SBATCH --time=$SLURM_TIME
#SBATCH --job-name=$SLURM_NAME
#SBATCH --output=$SLURM_OUTPUT
$QOS_DIRECTIVE
#SBATCH --error=$SLURM_ERROR
$MAIL_TYPE_DIRECTIVE
$MAIL_USER_DIRECTIVE

${PRE_BLOCK}

${USER_CMD[@]}

${POST_BLOCK}

EOT
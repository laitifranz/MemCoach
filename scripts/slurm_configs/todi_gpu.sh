SLURM_NODES='1'
SLURM_TASKS='1'
SLURM_CPU='8'
SLURM_GPU='1'
SLURM_MEM='32GB'
SLURM_TIME='24:00:00'
SLURM_NAME='default_todi_gpu'

SLURM_PARTITION='gpupart'
SLURM_ACCOUNT='staff'

MAIL_TYPE='ALL'
MAIL_USER='francesco.laiti@unitn.it'

# Telegram job notifier: https://gist.github.com/laitifranz/5c6c8c4c8d0469d50c4afad5f3079874
BEFORE_CODE_BLOCK=(
    'bash /home/francesco.laiti/data/telegram-job-notifier.sh "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "BEGIN"'
)

AFTER_CODE_BLOCK=(
    'bash /home/francesco.laiti/data/telegram-job-notifier.sh "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "END" "$?"'
)
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SEED=${SEED:-1}
GPU_POOL=${GPU_POOL:-""}
DATASET=${DATASET:-"imagenet"}
POSITION=${POSITION:-"up"}
EVAL_DATASETS=${EVAL_DATASETS:-"imagenet-a,imagenet-r,imagenet-v2,imagenet-sketch,objectnet"}
RHOS=${RHOS:-"0.3,0.4"}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT=${RESULT_ROOT:-"/home/dingzijin/results/vitb16_r248_asam_fishersam_focalsam_${TIMESTAMP}"}
SAVE_ROOT=${SAVE_ROOT:-"/home/dingzijin/models/vitb16_r248_asam_fishersam_focalsam_${TIMESTAMP}"}

mkdir -p logs "$RESULT_ROOT" "$SAVE_ROOT"

SUMMARY_FILE="logs/sam_summary.tsv"
echo -e "opt\tbackbone\trank\trho\tstart\tend\tduration_min\tstatus\tlog_file" > "$SUMMARY_FILE"

declare -a RANKS=(2)
declare -a OPTS=(sam)
IFS=',' read -r -a RHO_LIST <<< "$RHOS"

declare -A LR_BY_RANK=(
  [2]=1e-4
)

if [[ -z "$GPU_POOL" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found, please set GPU_POOL manually, e.g. GPU_POOL=0,1,2,3"
    exit 1
  fi
  GPU_POOL=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_POOL"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no GPU found. Set GPU_POOL manually, e.g. GPU_POOL=0,1,2,3"
  exit 1
fi

echo "============================================"
echo "GPU pool:      ${GPU_POOL}"
echo "Rhos:          ${RHOS}"
echo "Summary:       ${SUMMARY_FILE}"
echo "============================================"

declare -a TASKS=()
for rank in "${RANKS[@]}"; do
  for opt in "${OPTS[@]}"; do
    for rho in "${RHO_LIST[@]}"; do
      TASKS+=("${rank}|${opt}|${rho}")
    done
  done
done

declare -a FREE_GPUS=("${GPU_LIST[@]}")
declare -A PID_TO_GPU
declare -A PID_TO_META
declare -A PID_TO_START_EPOCH
declare -A PID_TO_START_HUMAN

launch_job() {
  local gpu="$1"
  local rank="$2"
  local opt="$3"
  local rho="$4"
  local lr
  lr="${LR_BY_RANK[$rank]}"

  local rho_tag
  rho_tag=${rho//./p}
  local run_name="${opt}_vitb16_r${rank}_rho${rho_tag}"
  local log_file="logs/${run_name}.log"
  local result_root="${RESULT_ROOT}/rho${rho_tag}"
  local save_root="${SAVE_ROOT}/rho${rho_tag}"

  mkdir -p "$result_root" "$save_root"

  local run_start
  run_start=$(date +%s)
  local run_start_human
  run_start_human=$(date +"%F %T")

  echo "[START] ${run_name} on GPU ${gpu}"
  echo "[TIME]  ${run_start_human}"
  echo "[LOG]   ${log_file}"

  (
    set +e
    python main.py \
      --opt "$opt" \
      --dataset "$DATASET" \
      --backbone "ViT-B/16" \
      --r "$rank" \
      --lr "$lr" \
        --rho "$rho" \
      --gpu_ids "$gpu" \
      --position "$POSITION" \
        --result_path "$result_root" \
        --save_path "$save_root" \
      --seed "$SEED" \
      "${extra_args[@]}" \
      > "$log_file" 2>&1
  ) &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
      PID_TO_META["$pid"]="${opt}|${run_name}|${log_file}|${rank}|${rho}"
  PID_TO_START_EPOCH["$pid"]="$run_start"
  PID_TO_START_HUMAN["$pid"]="$run_start_human"
}

finish_one_job_if_any() {
  local pid
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      local exit_code
      set +e
      wait "$pid"
      exit_code=$?
      set -e

      local run_end
      run_end=$(date +%s)
      local run_end_human
      run_end_human=$(date +"%F %T")
      local duration_sec=$((run_end - PID_TO_START_EPOCH[$pid]))
      local duration_min=$(((duration_sec + 59) / 60))

      local gpu="${PID_TO_GPU[$pid]}"
      IFS='|' read -r opt run_name log_file rank rho <<< "${PID_TO_META[$pid]}"

      if [[ "$exit_code" -eq 0 ]]; then
        local status="ok"
        echo "[DONE] ${run_name} on GPU ${gpu} | duration=${duration_min}min(ceil) | end=${run_end_human}"
      else
        local status="fail(${exit_code})"
        echo "[FAIL] ${run_name} on GPU ${gpu} | duration=${duration_min}min(ceil) | end=${run_end_human} | exit=${exit_code}, check ${log_file}"
      fi

      echo -e "${opt}\tViT-B/16\t${rank}\t${rho}\t${PID_TO_START_HUMAN[$pid]}\t${run_end_human}\t${duration_min}\t${status}\t${log_file}" >> "$SUMMARY_FILE"

      FREE_GPUS+=("$gpu")
      unset 'PID_TO_GPU[$pid]'
      unset 'PID_TO_META[$pid]'
      unset 'PID_TO_START_EPOCH[$pid]'
      unset 'PID_TO_START_HUMAN[$pid]'
      return 0
    fi
  done
  return 1
}

task_idx=0
total_tasks=${#TASKS[@]}
echo "Total tasks:   ${total_tasks}"

while [[ "$task_idx" -lt "$total_tasks" ]] || [[ ${#PID_TO_GPU[@]} -gt 0 ]]; do
  while [[ "$task_idx" -lt "$total_tasks" ]] && [[ ${#FREE_GPUS[@]} -gt 0 ]]; do
    task="${TASKS[$task_idx]}"
    task_idx=$((task_idx + 1))
    IFS='|' read -r rank opt rho <<< "$task"

    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    launch_job "$gpu" "$rank" "$opt" "$rho"
  done

  if ! finish_one_job_if_any; then
    sleep 2
  fi
done

echo "All runs finished."
echo "Summary is at: ${SUMMARY_FILE}"

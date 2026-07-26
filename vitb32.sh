#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SEED=${SEED:-1}
GPU_POOL=${GPU_POOL:-""}
DATASET=${DATASET:-"imagenet"}
SHOTS=${SHOTS:-16}
POSITION=${POSITION:-"half-up"}
EVAL_DATASETS=${EVAL_DATASETS:-"imagenet-a,imagenet-r,imagenet-v2,imagenet-sketch,objectnet"}

RESULT_ROOT=${RESULT_ROOT:-"/home/dingzijin/results/vitl14_rank4_optimizers"}
SAVE_ROOT=${SAVE_ROOT:-"/home/dingzijin/models/vitl14_rank4_optimizers"}

mkdir -p logs "$RESULT_ROOT" "$SAVE_ROOT"

SUMMARY_FILE="logs/vitb32_rank2_optimizers_summary.tsv"
echo -e "opt\tbackbone\trank\tstart\tend\tduration_min\tstatus\tlog_file" > "$SUMMARY_FILE"

BACKBONE="ViT-B/32"
RANK=2
LR=1e-4

# Use labels for reporting and map to main.py options.
declare -a OPT_LABELS=(
  "sgd"
  "adam"
  "entropy-sgd"
  "sam"
  "adaptive_sam"
  "fisher_sam"
  "focal_sam"
)

to_main_opt_name() {
  case "$1" in
    entropy-sgd) echo "entropy_sgd" ;;
    adaptive_sam) echo "sam" ;;
    *) echo "$1" ;;
  esac
}

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

echo "============================================================"
echo "Backbone:      ${BACKBONE}"
echo "Rank:          ${RANK}"
echo "LR:            ${LR}"
echo "Dataset:       ${DATASET}"
echo "Shots:         ${SHOTS}"
echo "Optimizers:    ${OPT_LABELS[*]}"
echo "GPU pool:      ${GPU_POOL}"
echo "Parallel jobs: ${#GPU_LIST[@]}"
echo "Result root:   ${RESULT_ROOT}"
echo "Save root:     ${SAVE_ROOT}"
echo "Summary:       ${SUMMARY_FILE}"
echo "============================================================"

declare -a TASKS=()
for opt_label in "${OPT_LABELS[@]}"; do
  TASKS+=("${opt_label}")
done

declare -a FREE_GPUS=("${GPU_LIST[@]}")
declare -A PID_TO_GPU
declare -A PID_TO_META
declare -A PID_TO_START_EPOCH
declare -A PID_TO_START_HUMAN

launch_job() {
  local gpu="$1"
  local opt_label="$2"
  local opt
  opt=$(to_main_opt_name "$opt_label")

  local run_name="${opt_label}_vitl14_r${RANK}_lr${LR}"
  local log_file="logs/${run_name}.log"

  local run_start
  run_start=$(date +%s)
  local run_start_human
  run_start_human=$(date +"%F %T")

  local -a extra_args=()
  if [[ "$opt_label" == "adaptive_sam" ]]; then
    extra_args+=(--adaptive_sam true)
  fi

  echo "[START] ${run_name} on GPU ${gpu}"
  echo "[TIME]  ${run_start_human}"
  echo "[LOG]   ${log_file}"

  (
    set +e
    python main.py \
      --opt "$opt" \
      --dataset "$DATASET" \
      --shots "$SHOTS" \
      --backbone "$BACKBONE" \
      --r "$RANK" \
      --lr "$LR" \
      --gpu_ids "$gpu" \
      --position "$POSITION" \
      --eval_datasets "$EVAL_DATASETS" \
      --result_path "$RESULT_ROOT" \
      --save_path "$SAVE_ROOT" \
      --seed "$SEED" \
      "${extra_args[@]}" \
      > "$log_file" 2>&1
  ) &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_META["$pid"]="${opt_label}|${run_name}|${log_file}"
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
      IFS='|' read -r opt_label run_name log_file <<< "${PID_TO_META[$pid]}"

      local status
      if [[ "$exit_code" -eq 0 ]]; then
        status="ok"
        echo "[DONE] ${run_name} on GPU ${gpu} | duration=${duration_min}min(ceil) | end=${run_end_human}"
      else
        status="fail(${exit_code})"
        echo "[FAIL] ${run_name} on GPU ${gpu} | duration=${duration_min}min(ceil) | end=${run_end_human} | exit=${exit_code}, check ${log_file}"
      fi

      echo -e "${opt_label}\t${BACKBONE}\t${RANK}\t${PID_TO_START_HUMAN[$pid]}\t${run_end_human}\t${duration_min}\t${status}\t${log_file}" >> "$SUMMARY_FILE"

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
    opt_label="${TASKS[$task_idx]}"
    task_idx=$((task_idx + 1))

    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    launch_job "$gpu" "$opt_label"
  done

  if ! finish_one_job_if_any; then
    sleep 2
  fi
done

echo "All runs finished."
echo "Summary is at: ${SUMMARY_FILE}"

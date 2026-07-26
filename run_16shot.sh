#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SEED=${SEED:-1}
GPU_POOL=${GPU_POOL:-"2,3,4,5,6,7"}
EVAL_DATASETS=${EVAL_DATASETS:-"imagenet-a,imagenet-r,imagenet-v2,imagenet-sketch,objectnet"}
POSITION=${POSITION:-"half-up"}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT=${RESULT_ROOT:-"/home/dingzijin/results/rank_backbone_16shot_${TIMESTAMP}"}
SAVE_ROOT=${SAVE_ROOT:-"/home/dingzijin/models/rank_backbone_16shot_${TIMESTAMP}"}

mkdir -p logs "$RESULT_ROOT" "$SAVE_ROOT"

BACKBONES=("ViT-B/32")
RANKS=(2)
OPTS=("sgd" "adam" "sam" "adaptive_sam" "fisher_sam" "focal_sam" "entropy_sgd")

SUMMARY_FILE="logs/16shot_summary_${TIMESTAMP}.tsv"
echo -e "opt\tbackbone\trank\tstart\tend\tduration_sec\tstatus\tlog_file" > "$SUMMARY_FILE"

to_main_opt_name() {
  case "$1" in
    entropy-sgd|entropy_sgd) echo "entropy_sgd" ;;
    adaptive_sam) echo "sam" ;;
    *) echo "$1" ;;
  esac
}

echo "Result root: $RESULT_ROOT"
echo "Save root:   $SAVE_ROOT"
echo "GPU pool:    $GPU_POOL"
echo "Eval sets:   $EVAL_DATASETS"
echo "Position:    $POSITION"
echo "Summary:     $SUMMARY_FILE"

script_start=$(date +%s)

IFS=',' read -r -a GPU_LIST <<< "$GPU_POOL"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "ERROR: GPU_POOL is empty"
  exit 1
fi

declare -a TASKS=()
for opt_label in "${OPTS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for rank in "${RANKS[@]}"; do
      TASKS+=("${opt_label}|${backbone}|${rank}")
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
  local opt_label="$2"
  local backbone="$3"
  local rank="$4"
  local opt
  opt=$(to_main_opt_name "$opt_label")

  local safe_backbone
  safe_backbone=${backbone//\//_}
  local log_file="logs/16shot_${opt_label}_${safe_backbone}_r${rank}.log"

  local -a extra_args=()
  if [ "$opt_label" = "adaptive_sam" ]; then
    extra_args+=(--adaptive_sam true)
  fi

  local run_start
  run_start=$(date +%s)
  local run_start_human
  run_start_human=$(date +"%F %T")

  echo "============================================================"
  echo "Running: opt=$opt_label(main:$opt), backbone=$backbone, r=$rank"
  echo "GPU:     $gpu"
  echo "Start:   $run_start_human"
  echo "Log:     $log_file"

  (
    set +e
    python main.py \
      --opt "$opt" \
      --dataset imagenet \
      --shots 16 \
      --backbone "$backbone" \
      --r "$rank" \
      --gpu_ids "$gpu" \
      --position "$POSITION" \
      --eval_datasets "$EVAL_DATASETS" \
      --result_path "$RESULT_ROOT" \
      --save_path "$SAVE_ROOT" \
      --seed "$SEED" \
      --lr 1e-4 \
        "${extra_args[@]}" \
      > "$log_file" 2>&1
  ) &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_META["$pid"]="${opt_label}|${backbone}|${rank}|${log_file}"
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
      local duration=$((run_end - PID_TO_START_EPOCH[$pid]))

      local status
      if [ "$exit_code" -eq 0 ]; then
        status="ok"
      else
        status="fail($exit_code)"
      fi

      IFS='|' read -r opt_label backbone rank log_file <<< "${PID_TO_META[$pid]}"
      local gpu="${PID_TO_GPU[$pid]}"

      echo "Done:    $run_end_human | gpu=$gpu | opt=$opt_label, backbone=$backbone, r=$rank | duration=${duration}s | status=$status"
      echo -e "${opt_label}\t${backbone}\t${rank}\t${PID_TO_START_HUMAN[$pid]}\t${run_end_human}\t${duration}\t${status}\t${log_file}" >> "$SUMMARY_FILE"

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
echo "Total tasks: $total_tasks"
echo "Max parallel jobs: ${#GPU_LIST[@]}"

while [ "$task_idx" -lt "$total_tasks" ] || [ "${#PID_TO_GPU[@]}" -gt 0 ]; do
  while [ "$task_idx" -lt "$total_tasks" ] && [ "${#FREE_GPUS[@]}" -gt 0 ]; do
    task="${TASKS[$task_idx]}"
    task_idx=$((task_idx + 1))
    IFS='|' read -r opt_label backbone rank <<< "$task"

    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    launch_job "$gpu" "$opt_label" "$backbone" "$rank"
  done

  if ! finish_one_job_if_any; then
    sleep 2
  fi
done

script_end=$(date +%s)
total_duration=$((script_end - script_start))

echo "============================================================"
echo "All runs completed."
echo "Results are under: $RESULT_ROOT"
echo "LoRA weights are under: $SAVE_ROOT"
echo "Summary is at:      $SUMMARY_FILE"
echo "Total duration:     ${total_duration}s"

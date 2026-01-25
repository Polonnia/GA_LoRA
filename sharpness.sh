#!/bin/bash

# 创建日志文件夹，保持目录整洁
mkdir -p logs

# 定义 GPU ID 数组 (对应关系)
# Index:  0   1   2   3   4
GPUS=(    2   3   4   5   6   )

# 定义 优化器 数组 (对应关系)
OPTS=( "sgd" "sgd_momentum" "entropy_sgd" "adam" "sam" )

# 循环遍历数组并启动任务
# ${!GPUS[@]} 获取数组索引列表 (0, 1, 2, 3, 4)
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    OPT=${OPTS[$i]}
    
    echo "Starting task: Optimizer = $OPT, GPU = $GPU_ID, Shots = 4"
    
    # 核心 nohup 命令
    # --gpu_ids $GPU_ID: 这里传入单个数字，脚本解析列表时取第一个即为此ID
    nohup python eval_sharpness.py \
        --opt "$OPT" \
        --shots 4 \
        --gpu_ids "$GPU_ID" \
        > "logs/${OPT}_4shot_sharpness.log" 2>&1 &
done

echo "All 5 tasks submitted successfully!"
#!/bin/bash

# 创建日志目录
mkdir -p logs

# 直接运行所有命令
nohup python main.py --opt adam --dataset imagenet-v2 --eval_datasets imagenet,imagenet-a,imagenet-r,imagenet-sketch > logs/imagenet-v2.log 2>&1 &
echo "启动 imagenet-v2, PID: $!"

nohup python main.py --opt adam --dataset imagenet-sketch --eval_datasets imagenet,imagenet-a,imagenet-r,imagenet-v2 > logs/imagenet-sketch.log 2>&1 &
echo "启动 imagenet-sketch, PID: $!"
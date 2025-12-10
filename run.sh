#!/bin/bash
#nohup python main.py --opt adam --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/adam-in.log 2>&1 &
nohup python main.py --opt adam --gpu_ids 6 --n_iters 300 --dataset imagenet-v2 --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet > logs/adam-v2.log 2>&1 &
#nohup python main.py --opt adam --dataset imagenet-a --eval_datasets imagenet,imagenet-r,imagenet-sketch,imagenet-v2 > logs/adam-in.log 2>&1 &
#nohup python main.py --opt adam --dataset imagenet-r --eval_datasets imagenet,imagenet-a,imagenet-sketch,imagenet-v2 > logs/adam-r.log 2>&1 &
#nohup python main.py --opt adam --dataset imagenet-sketch --eval_datasets imagenet,imagenet-a,imagenet-r,imagenet-v2 > logs/adam-sketch.log 2>&1 &
# nohup python main.py --opt sgd --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/sgd-8shot.log 2>&1 &
# nohup python main.py --opt ga --dataset imagenet-v2 --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet > logs/ga-8shot.log 2>&1 &
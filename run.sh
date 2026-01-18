#!/bin/bash
nohup python main.py --opt adam --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/adam-16.log 2>&1 &
#nohup python main.py --opt adam --gpu_ids 6 --n_iters 300 --dataset imagenet-v2 --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet > logs/adam-v2.log 2>&1 &
#nohup python main.py --opt adam --dataset imagenet-a --eval_datasets imagenet,imagenet-r,imagenet-sketch,imagenet-v2 > logs/adam-in.log 2>&1 &
#nohup python main.py --opt adam --dataset imagenet-r --eval_datasets imagenet,imagenet-a,imagenet-sketch,imagenet-v2 > logs/adam-r.log 2>&1 &
#nohup python main.py --opt adam --dataset imagenet-sketch --eval_datasets imagenet,imagenet-a,imagenet-r,imagenet-v2 > logs/adam-sketch.log 2>&1 &
nohup python main.py --opt sgd --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/sgd-16.log 2>&1 &

# nohup python main.py --opt ga --dataset imagenet-v2 --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet > logs/ga-8shot.log 2>&1 &
# nohup python main.py --opt ga --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/ga-in.log 2>&1 &

#nohup python main.py --opt sgd --gpu_ids 5 --dataset imagenet-a --eval_datasets imagenet,imagenet-r,imagenet-sketch,imagenet-v2 > logs/sgd-in.log 2>&1 &
#nohup python main.py --opt sgd --gpu_ids 6 --dataset imagenet-r --eval_datasets imagenet,imagenet-a,imagenet-sketch,imagenet-v2 > logs/sgd-r.log 2>&1 &
#nohup python main.py --opt sgd --gpu_ids 7 --dataset imagenet-sketch --eval_datasets imagenet,imagenet-a,imagenet-r,imagenet-v2 > logs/sgd-sketch.log 2>&1 &

# nohup python main.py --opt zo_sgd --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/zo_sgd-8shot.log 2>&1 &

nohup python main.py --opt sam --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/sam-16.log 2>&1 &
nohup python main.py --opt adam --n_iters 5000 --dataset imagenet --eval_datasets imagenet-a,imagenet-r,imagenet-sketch,imagenet-v2 > logs/adam-all.log 2>&1 &
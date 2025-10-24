
import random
import argparse  
import numpy as np 
import torch

    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='/root/autodl-tmp', help='path to dataset root directory')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--shots', default=16, type=int, help='Number of shots per class. Use -1 for all training data.')
    parser.add_argument('--opt', default='ga', type=str, help='Optimization method. Use "adam" or "ga".')
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='text')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    parser.add_argument('--save_path', default='/root/models', help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    parser.add_argument('--result_path', default='/root/results', type=str, help='directory to save validation results (e.g., accuracies)')
    parser.add_argument('--eval_datasets', default='imagenet-a,imagenet-r', type=lambda x: x.split(','), help='comma-separated list of eval datasets, e.g. imagenet-v2,imagenet-a,imagenet-r')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='batch size for evaluation/feature preloading to avoid OOM')
    args = parser.parse_args()

    return args
    

        

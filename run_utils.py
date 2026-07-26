
import random
import argparse  
import numpy as np 
import torch


def parse_gpu_ids(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, int):
        return [value]

    text = str(value).strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]

    if not text:
        return []

    return [int(item.strip()) for item in text.split(',') if item.strip()]

    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--gpu_ids", default=[1,2,3,4,5,6,7], type=parse_gpu_ids,
                        help='Comma-separated GPU ids, e.g. "2,3" or "2,3,2,3"')
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='/data/dataset', help='path to dataset root directory')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--shots', default=16, type=int, help='Number of shots per class.')
    parser.add_argument('--opt', default='adam', type=str, help='Optimization method. Examples: adam, sgd, sam, fisher_sam, focal_sam, entropy_sgd, ga.')
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    # Memory optimization arguments
    parser.add_argument('--grad_accumulation_steps', default=1, type=int, help='gradient accumulation steps for large models')
    parser.add_argument('--use_amp', default=True, type=lambda x: x.lower() in ['true', '1', 'yes'], help='use automatic mixed precision (AMP)')
    parser.add_argument('--use_gradient_checkpointing', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help='enable gradient checkpointing to save memory')
    # LoRA arguments
    parser.add_argument('--position', type=str, default='half-up', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='vision')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    parser.add_argument('--save_path', default='/home/dingzijin/models', help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    parser.add_argument('--result_path', default='/home/dingzijin/results', type=str, help='directory to save validation results (e.g., accuracies)')
    parser.add_argument('--eval_datasets', default='imagenet-a,imagenet-r,imagenet-v2,imagenet-sketch,objectnet', type=lambda x: [i.strip() for i in x.split(',') if i.strip()], help='comma-separated list of eval datasets, e.g. imagenet-v2,imagenet-a,imagenet-r')
    parser.add_argument('--train_from_ga', default=False, action='store_true', help='whether to load LoRA weights trained from GA as initialization')
    
    # Sharpness
    parser.add_argument('--rho', default=0.0002, type=float, help='Perturbation radius')
    parser.add_argument('--adaptive_sam', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help='Enable adaptive SAM (ASAM) behavior')
    parser.add_argument('--fsam_keep_ratio', default=0.1, type=float, help='FSAM Fisher mask keep ratio in (0,1]')
    parser.add_argument('--fsam_mask_update_interval', default=100, type=int, help='FSAM mask update interval (iterations)')
    parser.add_argument('--fsam_fisher_beta', default=0.9, type=float, help='FSAM Fisher EMA coefficient in [0,1)')
    parser.add_argument('--sharpness_iters', default=20, type=int, help='Number of PGD iterations')
    parser.add_argument('--step_size_mult', default=1.0, type=float)
    parser.add_argument('--norm', default='l2', choices=['l2', 'linf'])
    parser.add_argument('--n_eval_samples', default=1000, type=int, help='Num samples to eval')
    args = parser.parse_args()

    return args
    

        

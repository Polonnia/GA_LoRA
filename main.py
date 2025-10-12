import os
import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora


def main():

    # Load config file
    args = get_arguments()
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    shots_list = [16]
    base_result_path = getattr(args, 'result_path', None)
    
    # # Zero-shot CLIP evaluation before training
    # print("="*60)
    # print("ZERO-SHOT CLIP EVALUATION")
    # print("="*60)
    
    # # Use the first shot configuration for zero-shot evaluation
    # args.shots = shots_list[0]
    # print(f"Preparing dataset for zero-shot evaluation (shots={args.shots}).")
    # dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    
    # # Create data loaders for zero-shot evaluation
    # if args.dataset == 'imagenet':
    #     test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=args.eval_batch_size, num_workers=8, shuffle=False, pin_memory=True)
    # else:
    #     test_loader = build_data_loader(data_source=dataset.test, batch_size=args.eval_batch_size, is_train=False, tfm=preprocess, shuffle=False, num_workers=8)
    
    # # Evaluate zero-shot CLIP
    # from lora import evaluate_lora
    # zs_test_acc = evaluate_lora(args, clip_model, test_loader, dataset)
    # print(f"Zero-shot CLIP Test Accuracy: {zs_test_acc:.2f}%")

    # result_path = os.path.join(base_result_path, 'shot0')
    # os.makedirs(result_path, exist_ok=True)
    # with open(os.path.join(result_path, 'val_results.json'), 'w') as f:
    #     import json
    #     json.dump({f'zs_test_acc_seed{args.seed}': zs_test_acc}, f)
    
    # # Evaluate on ImageNet variants if specified
    # if getattr(args, 'eval_datasets', None):
    #     eval_list = [d.strip() for d in args.eval_datasets if d.strip()]
    #     if len(eval_list) > 0 and args.dataset == 'imagenet':
    #         print("\n--- Zero-shot CLIP on ImageNet Variants ---")
    #         from lora import _evaluate_imagenet_variant
    #         zs_variant_results = {}
    #         for variant in eval_list:
    #             try:
    #                 v_acc = _evaluate_imagenet_variant(args, clip_model, variant, args.root_path)
    #                 print(f"Zero-shot CLIP {variant} accuracy: {v_acc:.2f}%")
    #                 zs_variant_results[f'zs_acc_{variant}_seed{args.seed}'] = float(v_acc)
    #             except Exception as e:
    #                 print(f"Warning: failed evaluating {variant}: {e}")
    # with open(os.path.join(result_path, 'val_results.json'), 'w') as f:
    #     json.dump(zs_variant_results, f)
    print("="*60)
    print("STARTING FEW-SHOT TRAINING")
    print("="*60)

    for s in shots_list:
        # Update shots and per-shot result directory
        args.shots = s
        if base_result_path is not None:
            args.result_path = os.path.join(base_result_path, f"shots{s}")
            os.makedirs(args.result_path, exist_ok=True)

        # Re-load CLIP model for each shot to avoid LoRA state contamination
        print(f"Re-loading CLIP model for shots={s}")
        clip_model, preprocess = clip.load(args.backbone)
        clip_model.eval()
        logit_scale = 100

        print(f"Preparing dataset (shots={s}).")
        dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
        
        if args.dataset == 'imagenet':
            val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=args.eval_batch_size, num_workers=8, shuffle=False, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=args.eval_batch_size, num_workers=8, shuffle=False, pin_memory=True)
        else:
            val_loader = build_data_loader(data_source=dataset.val, batch_size=args.eval_batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=args.eval_batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            
        train_loader = None
        if not args.eval_only:
            train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
            
            if args.dataset == 'imagenet':
                train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
            else:
                train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

        run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()
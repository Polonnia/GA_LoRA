import os
import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils import *
from sgd import run_lora_sgd
from adam import run_lora_adam
from ga import run_lora_ga
from sam import run_lora_sam
from entropy_SGD import run_lora_entropy_sgd
from loralib.utils import apply_lora, load_lora

def main():

    # Load config file
    args = get_arguments()
    print(args)
    
    set_random_seed(args.seed)
    
    logit_scale = 100

    shots_list = [1,4,8,16]
    base_result_path = getattr(args, 'result_path', None)
    
    # # Zero-shot CLIP evaluation before training
    # print("="*60)
    # print("ZERO-SHOT CLIP EVALUATION")
    # print("="*60)
    
    # # Use the first shot configuration for zero-shot evaluation
    # args.shots = shots_list[0]
    # print(f"Preparing dataset for zero-shot evaluation.")
    # dataset = build_dataset(args.dataset, args.root_path, 0, preprocess)
    
    # # Create data loaders for zero-shot evaluation
    # if args.dataset == 'imagenet':
    #     test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=args.eval_batch_size, num_workers=8, shuffle=False, pin_memory=True)
    # else:
    #     test_loader = build_data_loader(data_source=dataset.test, batch_size=args.eval_batch_size, is_train=False, tfm=preprocess, shuffle=False, num_workers=8)
    
    # result_path = os.path.join(base_result_path, 'shot0')
    # # Evaluate zero-shot CLIP
    # from utils import evaluate
    # evaluate(clip_model, "zs", test_loader, dataset, args.eval_datasets, result_path, args.seed)
    
    print("="*60)
    print("STARTING FEW-SHOT TRAINING")
    print("="*60)
    
    for s in shots_list:
        # Update shots and per-shot result directory
        args.shots = s
        if base_result_path is not None:
            if s == -1:
                args.result_path = os.path.join(base_result_path, "shots_all", args.dataset)
            else:
                args.result_path = os.path.join(base_result_path, f"shots{s}", args.dataset)
            os.makedirs(args.result_path, exist_ok=True)

        # Re-load CLIP model for each shot to avoid LoRA state contamination
        print(f"Loading CLIP model for shots={s}")
        target_device = torch.device(f"cuda:{args.gpu_ids[0]}") 
        clip_model, preprocess = clip.load(args.backbone, device=target_device)
        clip_model.eval()
        logit_scale = 100

        root_path = args.root_path
        print(f"Preparing dataset (shots={s}).")
        if(args.dataset == 'imagenet-v2' or args.dataset == 'imagenet-sketch'):
            root_path = '/home/dingzijin/datasets'
        dataset = build_dataset(args.dataset, root_path, args.shots, preprocess, args.batch_size)
        
        # if args.dataset == 'imagenet':
        #     val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=args.eval_batch_size, num_workers=8, shuffle=False, pin_memory=True)
        #     test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=args.eval_batch_size, num_workers=8, shuffle=False, pin_memory=True)
        # else:
        #     val_loader = build_data_loader(data_source=dataset.val, batch_size=args.eval_batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        #     test_loader = build_data_loader(data_source=dataset.test, batch_size=args.eval_batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
            
        # train_loader = None
        # if not args.eval_only:
        #     train_transform = transforms.Compose([
        #         transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        #     ])
            
        #     if args.dataset == 'imagenet':
        #         train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        #     else:
        #         train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_transform, is_train=True, shuffle=True, num_workers=8)

        if args.eval_only:
            if args.shots == 0:
                print("Evaluating zero-shot CLIP model.")
                if 'objectnet' in args.eval_datasets:
                    from datasets.objectnet import ObjectNet
                    
                    print("Evaluating on ObjectNet...")
                    objectnet_ds = ObjectNet(root=args.root_path, preprocess=preprocess)

                    acc1 = evaluate_objectnet(
                        model=clip_model,
                        data_loader=objectnet_ds.test_loader,
                        device=target_device,
                        objectnet_obj=objectnet_ds,
                        result_path=args.result_path,
                        opt="zs",
                        seed=args.seed
                    )
                    
                    print(f"Final ObjectNet Accuracy: {acc1}%")
                    continue
                evaluate(clip_model, "zs", dataset, args.eval_datasets, args.result_path, args.seed, args.root_path)
                continue
            else:
                print(f"Evaluating LoRA model for shots={s}.")
                list_lora_layers = apply_lora(args, clip_model)
                load_lora(args, list_lora_layers)
                target_device = torch.device(f"cuda:{args.gpu_ids[3]}") 
                clip_model = clip_model.to(target_device)
                clip_model.eval()
                if 'objectnet' in args.eval_datasets:
                    from datasets.objectnet import ObjectNet
                    
                    print("Evaluating on ObjectNet...")
                    
                    objectnet_ds = ObjectNet(root=args.root_path, preprocess=preprocess)
                    
                    acc1 = evaluate_objectnet(
                        model=clip_model,
                        data_loader=objectnet_ds.test_loader,
                        device=target_device,
                        objectnet_obj=objectnet_ds,
                        result_path=args.result_path,
                        opt=args.opt,
                        seed=args.seed
                    )
                    
                    print(f"Final ObjectNet Accuracy: {acc1}%")
                    continue
                evaluate(clip_model, args.opt, dataset, args.eval_datasets, args.result_path, args.seed, args.root_path)
                continue
        
        if args.opt == 'adam':
            print("Running LoRA with AdamW optimization")
            run_lora_adam(args, clip_model, logit_scale, dataset, device_id=6, train_from_ga=args.train_from_ga, finetune_full=False)
        elif args.opt == 'sgd':
            print("Running LoRA with SGD optimization")
            run_lora_sgd(args, clip_model, logit_scale, dataset, device_id=5, momentum=False)
        elif args.opt == 'sgd_momentum':
            print("Running LoRA with SGD with momentum optimization")
            run_lora_sgd(args, clip_model, logit_scale, dataset, device_id=4, momentum=True)
        elif args.opt == 'entropy_sgd':
            print("Running LoRA with Entropy-SGD optimization")
            run_lora_entropy_sgd(args, clip_model, logit_scale, dataset, device_id=1)
        elif args.opt == 'ga':
            print("Running LoRA with GA optimization")
            run_lora_ga(args, clip_model, dataset, gpu_ids=args.gpu_ids)
        elif args.opt == 'sam':
            print("Running LoRA with SAM optimization")
            run_lora_sam(args, clip_model, logit_scale, dataset, device_id=4)
        else:
            raise ValueError("Unknown optimization method specified.")
        
        print(f"Evaluating LoRA model for {args.opt} on {args.dataset}.")
        list_lora_layers = apply_lora(args, clip_model)
        load_lora(args, list_lora_layers)
        target_device = torch.device(f"cuda:{args.gpu_ids[0]}") 
        clip_model = clip_model.to(target_device)
        evaluate(clip_model, args.opt, dataset, args.eval_datasets, args.result_path, args.seed, args.root_path)

if __name__ == '__main__':
    main()
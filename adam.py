import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, save_lora, load_lora

def run_lora_adam(args, clip_model, logit_scale, dataset, device_id, train_from_ga=True, finetune_full=False):
    
    VALIDATION = True
  
    # Skip LoRA setup when doing full finetune
    if not finetune_full:
        list_lora_layers = apply_lora(args, clip_model)
        if train_from_ga:
            args.opt = 'ga'
            load_lora(args, list_lora_layers)
            args.opt = 'mix'
    else:
        args.opt = 'adam_full'
        print("Fine-tuning full vision encoder.")
        list_lora_layers = None

    torch.cuda.set_device(int(device_id))
    clip_model = clip_model.cuda()

    # Choose trainable parameters: default LoRA, or full vision encoder when requested
    if finetune_full:
        vision_module = getattr(clip_model, 'visual', None)
        if vision_module is None:
            # Fallback: unfreeze all parameters if a dedicated visual module isn't found
            for p in clip_model.parameters():
                p.requires_grad = True
            params_to_optimize = [p for p in clip_model.parameters() if p.requires_grad]
            print("Warning: `clip_model.visual` not found; unfreezing all parameters for full finetune.")
        else:
            # Freeze everything then unfreeze vision encoder parameters only
            for p in clip_model.parameters():
                p.requires_grad = False
            for p in vision_module.parameters():
                p.requires_grad = True
            params_to_optimize = vision_module.parameters()
        print("Full fine-tuning enabled: training vision encoder parameters.")
    else:
        mark_only_lora_as_trainable(clip_model)
        params_to_optimize = get_lora_parameters(clip_model)

    if args.shots != -1:
        total_iters = args.n_iters * args.shots
    else:
        total_iters = args.n_iters
    # Use AdamW for weight decay handling
    optimizer = torch.optim.AdamW(params_to_optimize, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    
    # For full finetune, add warmup scheduler to stabilize training
    if finetune_full:
        warmup_iters = min(1000, total_iters // 10)  # 10% warmup or 1000 iters, whichever is smaller
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters - warmup_iters, eta_min=1e-7)
        ])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-7)
    
    # Disable AMP for full finetune to avoid dtype mismatches (all params stay float32)
    use_amp = not finetune_full
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    count_iters = 0
    finish = False
    
    # Initialize training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_iterations = []  # Separate list for validation iterations
    learning_rates = []
    iterations = []
    
    template = "a photo of a {}."
    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
    
    # Prepare text features without AMP during full finetune
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
    else:
        texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(texts)
    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
    
    start_time = time.time()
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.

        for i, (images, target) in enumerate(dataset.train_loader):
            images, target = images.cuda(), target.cuda()
            
            # Compute embeddings with or without AMP
            if use_amp:
                if args.encoder == 'text' or args.encoder == 'both':
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        class_embeddings = clip_model.encode_text(texts)
                    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
                if args.encoder == 'vision' or args.encoder == 'both':
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
                else:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            image_features = clip_model.encode_image(images)
            else:
                # No AMP: compute all embeddings in float32
                if args.encoder == 'text' or args.encoder == 'both':
                    class_embeddings = clip_model.encode_text(texts)
                    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
                if args.encoder == 'vision' or args.encoder == 'both':
                    image_features = clip_model.encode_image(images)
                else:
                    with torch.no_grad():
                        image_features = clip_model.encode_image(images)
            
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping for full finetune to prevent NaN loss
                if finetune_full:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            
            # Record training metrics
            train_losses.append(loss_epoch)
            train_accuracies.append(acc_train)
            learning_rates.append(current_lr)
            iterations.append(count_iters)
            
            print('Iter: {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(
                count_iters, current_lr, acc_train, loss_epoch))

        
        # Eval - 在每次epoch结束后进行验证
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(clip_model, dataset.val_loader, dataset.classnames)
            val_accuracies.append(acc_val)
            val_iterations.append(count_iters)  # 记录验证时的迭代次数
            print("**** Iter: {}, Val accuracy: {:.2f}. ****\n".format(count_iters, acc_val))
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    # Plot training curves
    plot_training_curves(args, iterations, train_losses, train_accuracies, val_iterations, val_accuracies, learning_rates)
    
    if args.save_path != None and not finetune_full:
        save_lora(args, list_lora_layers)
    return
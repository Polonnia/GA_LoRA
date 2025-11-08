import os
import sys
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def run_lora_adam(args, clip_model, logit_scale, dataset, train_from_ga=True):
    
    VALIDATION = True
  
    list_lora_layers = apply_lora(args, clip_model)
    if train_from_ga:
        args.opt = 'ga'
        load_lora(args, list_lora_layers)

    torch.cuda.set_device(7)
    clip_model = clip_model.cuda()
    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-7)
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
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
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(texts)
    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
    
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.

        for i, (images, target) in enumerate(dataset.train_loader):
            images, target = images.cuda(), target.cuda()
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
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
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
            acc_val = evaluate_lora(clip_model, dataset.val_loader, dataset)
            val_accuracies.append(acc_val)
            val_iterations.append(count_iters)  # 记录验证时的迭代次数
            print("**** Iter: {}, Val accuracy: {:.2f}. ****\n".format(count_iters, acc_val))
    
    # Plot training curves
    plot_training_curves(args, iterations, train_losses, train_accuracies, val_iterations, val_accuracies, learning_rates)
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

def plot_training_curves(args, iterations, train_losses, train_accuracies, val_iterations, val_accuracies, learning_rates):
    """Plot training curves including loss, accuracy and learning rate"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    ax1.plot(iterations, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot training and validation accuracy
    ax2.plot(iterations, train_accuracies, 'g-', linewidth=2, label='Training Accuracy')
    if val_accuracies and len(val_iterations) == len(val_accuracies):
        ax2.plot(val_iterations, val_accuracies, 'r-', linewidth=2, label='Validation Accuracy')
    elif val_accuracies:
        # 如果维度不匹配，使用训练迭代次数作为x轴
        ax2.plot(iterations[:len(val_accuracies)], val_accuracies, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot learning rate
    ax3.plot(iterations, learning_rates, 'purple', linewidth=2, label='Learning Rate')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.legend()
    
    # Plot combined view
    ax4.plot(iterations, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(iterations, train_accuracies, 'g-', linewidth=2, label='Training Accuracy')
    if val_accuracies and len(val_iterations) == len(val_accuracies):
        ax4_twin.plot(val_iterations, val_accuracies, 'r-', linewidth=2, label='Validation Accuracy')
    elif val_accuracies:
        ax4_twin.plot(iterations[:len(val_accuracies)], val_accuracies, 'r-', linewidth=2, label='Validation Accuracy')
    
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('Accuracy (%)', color='g')
    ax4.set_title('Loss and Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    if args.result_path:
        plot_path = os.path.join(args.result_path, f'training_curves_shots{args.shots}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
    
    # Also save data as JSON for later analysis
    if args.result_path:
        data_path = os.path.join(args.result_path, f'training_data_shots{args.shots}.json')
        training_data = {
            'iterations': iterations,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_iterations': val_iterations,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates,
            'shots': args.shots,
            'dataset': args.dataset,
            'encoder': args.encoder
        }
        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"Training data saved to: {data_path}")
    
    plt.close()

def plot_simple_training_curve(args, iterations, train_losses, train_accuracies, val_iterations, val_accuracies):
    """Simplified version for quick plotting"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_losses, 'b-', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(iterations, train_accuracies, 'g-', linewidth=2, label='Train')
    if val_accuracies and len(val_iterations) == len(val_accuracies):
        plt.plot(val_iterations, val_accuracies, 'r-', linewidth=2, label='Val')
    elif val_accuracies:
        plt.plot(iterations[:len(val_accuracies)], val_accuracies, 'r-', linewidth=2, label='Val')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if args.result_path:
        plot_path = os.path.join(args.result_path, f'adamw_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Simple training curves saved to: {plot_path}")
    
    plt.close()
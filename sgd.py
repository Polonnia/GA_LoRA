import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora

def run_lora_sgd(args, clip_model, logit_scale, dataset, device_id, momentum = True):
    
    VALIDATION = True
  
    list_lora_layers = apply_lora(args, clip_model)

    torch.cuda.set_device(device_id)
    clip_model = clip_model.cuda()
    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    # ========= 核心区别：使用带动量的 SGD =========
    if momentum:
        print("Using SGD with momentum.")
        optimizer = torch.optim.SGD(
            get_lora_parameters(clip_model),
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-2,   # 和 AdamW 设置相同，方便对比
            nesterov=True        # 如不想用 Nesterov，可改为 False
        )
        args.opt = 'sgd_momentum'
    else:
        print("Using vanilla SGD without momentum.")
        optimizer = torch.optim.SGD(
            get_lora_parameters(clip_model),
            lr=0.005,
            momentum=False,
            weight_decay=1e-2,   # 和 AdamW 设置相同，方便对比
            nesterov=False        # 如不想用 Nesterov，可改为 False
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        total_iters,
        eta_min=1e-7
    )
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_iterations = []
    learning_rates = []
    iterations = []
    
    template = "a photo of a {}."
    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(texts)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.

        for i, (images, target) in enumerate(dataset.train_loader):
            images, target = images.cuda(), target.cuda()

            # 根据 encoder 设置是否每个 step 重新算 text feature
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
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
            
            train_losses.append(loss_epoch)
            train_accuracies.append(acc_train)
            learning_rates.append(current_lr)
            iterations.append(count_iters)
            
            print('Iter: {}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(
                count_iters, current_lr, acc_train, loss_epoch))

        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(clip_model, dataset.val_loader, dataset.classnames)
            val_accuracies.append(acc_val)
            val_iterations.append(count_iters)
            print("**** Iter: {}, Val accuracy: {:.2f}. ****\n".format(count_iters, acc_val))
    
    plot_training_curves(
        args,
        iterations,
        train_losses,
        train_accuracies,
        val_iterations,
        val_accuracies,
        learning_rates
    )
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return

import os
import sys
import json
import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def _build_clip_preprocess():
    # Minimal CLIP-like eval preprocess
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

@torch.no_grad()
def _evaluate_imagenet_variant(args, clip_model, variant_name, root_path):
    from datasets.imagenet_a import ImageNetA
    from datasets.imagenet_r import ImageNetR 

    preprocess = _build_clip_preprocess()

    if variant_name == 'imagenet-a':
        dataset_obj = ImageNetA(preprocess=preprocess, location=root_path)
    elif variant_name == 'imagenet-r':
        dataset_obj = ImageNetR(preprocess=preprocess, location=root_path)
    else:
        raise ValueError(f'Unknown variant {variant_name}')

    # zero-shot classifier
    template = ['a photo of a {}.']
    texts = [template[0].format(c.replace('_', ' ')) for c in dataset_obj.classnames]
    base = unwrap(clip_model)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        tokenized = clip.tokenize(texts).cuda()
        class_embeddings = base.encode_text(tokenized)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    tot = 0
    clip_model.eval()
    for batch in dataset_obj.test_loader:
        # ? 兼容 tuple 格式
        if isinstance(batch, (list, tuple)):
            images, labels = batch
        elif isinstance(batch, dict):
            images = batch['images']
            labels = batch['labels']
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.cuda()
        labels = labels.cuda()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = base.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.t()
        acc += cls_acc(logits, labels) * images.shape[0]
        tot += images.shape[0]
    
    return float(acc / max(tot, 1))


def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    base = unwrap(clip_model)
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = base.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = base.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
  
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        clip_model = torch.nn.DataParallel(clip_model)
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        # acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        # print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        
        # Handle eval_datasets for eval_only mode
        eval_list = []
        if getattr(args, 'eval_datasets', None):
            if isinstance(args.eval_datasets, str):
                eval_list = [d.strip() for d in args.eval_datasets.split(',') if d.strip()]
            else:
                eval_list = [d.strip() for d in args.eval_datasets if d.strip()]

        if len(eval_list) > 0 and args.dataset == 'imagenet':
            try:
                for v in eval_list:
                    print(f"Evaluating {v}...")
                    v_acc = _evaluate_imagenet_variant(args, clip_model, v, args.root_path)
                    print(f"**** {v} accuracy: {v_acc:.2f}. ****")
                    # save results to json
                    result_json_path = os.path.join(args.result_path, 'val_results.json')
                    if os.path.exists(result_json_path):
                        try:
                            with open(result_json_path, 'r') as f:
                                exist = json.load(f)
                        except Exception:
                            exist = {}
                    else:
                        exist = {}
                    exist.update({f'acc_{v}_seed{args.seed}': float(v_acc)})
                    with open(result_json_path, 'w') as f:
                        json.dump(exist, f, indent=2)
            except Exception as e:
                print(f"Warning: failed evaluating variants {eval_list}: {e}")
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    base = unwrap(clip_model)
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        # if args.encoder == 'vision': 
        #     text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            base = unwrap(clip_model)
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = base.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = base.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = base.encode_image(images)
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
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    # Optional: evaluate on ImageNet variants and save results
    results_to_save = {}
    results_to_save[f'acc_test_seed{args.seed}'] = float(acc_test)

    # Controlled by either explicit list or legacy flag
    eval_list = []
    if getattr(args, 'eval_datasets', None):
        eval_list = [d.strip() for d in args.eval_datasets if d.strip()]

    if len(eval_list) > 0 and args.dataset == 'imagenet':
        try:
            for v in eval_list:
                v_acc = _evaluate_imagenet_variant(args, clip_model, v, args.root_path)
                print(f"**** {v} accuracy: {v_acc:.2f}. ****")
                results_to_save[f'acc_{v}_seed{args.seed}'] = float(v_acc)
        except Exception as e:
            print(f"Warning: failed evaluating variants {eval_list}: {e}")

    if args.result_path is not None:
        try:
            os.makedirs(args.result_path, exist_ok=True)
            result_file = os.path.join(args.result_path, 'val_results.json')
            with open(result_file, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            print(f"Results saved to {result_file}")
        except Exception as e:
            print(f"Warning: failed saving results: {e}")
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return
            
    
            

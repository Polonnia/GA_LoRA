from tqdm import tqdm
import torch
import clip
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet import ImageNet

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            # Use single template (first entry) for zero-shot classifier
            text = template[0].format(classname)
            texts = clip.tokenize([text]).cuda()
            # Handle DataParallel wrapper
            model = clip_model.module if hasattr(clip_model, 'module') else clip_model
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings[0]
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights

def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

@torch.no_grad()
def evaluate_lora(clip_model, loader, classnames, cached_text_features=None, cached_image_batches=None, cached_tokens=None):
    clip_model.eval()
    
    # Ensure the model is on the correct device
    device = next(clip_model.parameters()).device
    
    # Prepare or reuse text features
    if cached_text_features is not None:
        # Ensure cached text features are on the correct device
        text_features = cached_text_features.to(device)
    else:
        if cached_tokens is not None:
            texts = cached_tokens.to(device)  # Ensure tokens are on the correct device
        else:
            template = "a photo of a {}."
            texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(texts).to(device)  # Move tokens to the correct device
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    # Ensure text_features is on the correct device
    if text_features.device != device:
        text_features = text_features.to(device)

    acc = 0.0
    tot_samples = 0
    with torch.no_grad():
        if cached_image_batches is not None:
            for image_features, target in cached_image_batches:
                # Ensure image_features and target are on the correct device
                image_features = image_features.to(device)
                target = target.to(device)

                # Ensure text_features is on the same device as image_features
                if text_features.device != image_features.device:
                    text_features = text_features.to(image_features.device)

                # Ensure dtypes match (image_features may be half under autocast)
                if image_features.dtype != text_features.dtype:
                    text_features = text_features.to(image_features.dtype)

                cosine_similarity = image_features @ text_features.t()
                acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
                tot_samples += len(cosine_similarity)
        else:
            for images, target in loader:
                images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Ensure dtypes match and both tensors are on the same device
                if image_features.dtype != text_features.dtype:
                    text_features = text_features.to(image_features.dtype)
                
                # Ensure both image_features and text_features are on the same device
                if image_features.device != text_features.device:
                    text_features = text_features.to(image_features.device)

                cosine_similarity = image_features @ text_features.t()
                acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
                tot_samples += len(cosine_similarity)
    
    acc /= tot_samples
    return acc



# _evaluate_imagenet_variant function
@torch.no_grad()
def evaluate_imagenet_variant(clip_model, variant_name, root_path):
    preprocess = _build_clip_preprocess()

    if variant_name == 'imagenet-sketch':
        dataset_obj = ImageNetSketch(preprocess=preprocess, root="/home/dingzijin/datasets")
    elif variant_name == 'imagenet-v2':
        dataset_obj = ImageNetV2(preprocess=preprocess, root="/home/dingzijin/datasets")
    elif variant_name == 'imagenet-a':
        dataset_obj = ImageNetA(preprocess=preprocess, root=root_path)
    elif variant_name == 'imagenet-r':
        dataset_obj = ImageNetR(preprocess=preprocess, root=root_path)
    elif variant_name == 'imagenet':
        dataset_obj = ImageNet(preprocess=preprocess, root=root_path)
    else:
        raise ValueError(f'Unknown variant {variant_name}')


    # Zero-shot classifier
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
        if image_features.dtype != text_features.dtype:
            text_features = text_features.to(image_features.dtype)
        logits = image_features @ text_features.t()
        acc += cls_acc(logits, labels) * images.shape[0]
        tot += images.shape[0]
    
    return float(acc / max(tot, 1))

def evaluate(clip_model, opt, dataset, eval_datasets, result_path, seed, root_path=None):
    import os
    import json
    result_json_path = os.path.join(result_path, f'val_results_{opt}.json')
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    try:
        with open(result_json_path, 'r') as f:
            exist = json.load(f)
    except Exception:
        exist = {}

    acc_test = evaluate_lora(clip_model, dataset.test_loader, dataset.classnames)
    print("**** ID accuracy: {:.2f}. ****\n".format(acc_test))
    exist.update({f'acc_test_seed{seed}': float(acc_test)})

    tot_acc = 0
    if len(eval_datasets) > 0:
        try:
            for v in eval_datasets:
                print(f"Evaluating {v}...")
                v_acc = evaluate_imagenet_variant(clip_model, v, root_path)
                tot_acc += v_acc
                print(f"**** {v} accuracy: {v_acc:.2f}. ****")
                
                exist.update({f'acc_{v}_seed{seed}': float(v_acc)})
        except Exception as e:
            print(f"Warning: failed evaluating variants {eval_datasets}: {e}")
    avg_acc = tot_acc / len(eval_datasets) if len(eval_datasets) > 0 else 0.0
    exist.update({f'avg_acc_variants_seed{seed}': float(avg_acc)})
    # Save results
    with open(result_json_path, 'w') as f:
        json.dump(exist, f, indent=2)
    return

def _build_clip_preprocess():
    # Minimal CLIP-like eval preprocess
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


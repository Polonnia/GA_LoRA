import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from tqdm import tqdm
import clip

def cls_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算分类准确率"""
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item()

@torch.no_grad()
def evaluate_lora(clip_model: nn.Module, loader, dataset, 
                 cached_text_features: Optional[torch.Tensor] = None,
                 cached_image_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                 cached_tokens: Optional[torch.Tensor] = None) -> float:
    """评估LoRA模型"""
    clip_model.eval()
    device = next(clip_model.parameters()).device
    
    # 准备文本特征
    if cached_text_features is not None:
        text_features = cached_text_features
    else:
        if cached_tokens is not None:
            texts = cached_tokens.to(device)
        else:
            # Handle datasets with or without template attribute
            if hasattr(dataset, 'template') and dataset.template is not None:
                template = dataset.template[0] if isinstance(dataset.template, (list, tuple)) else dataset.template
            else:
                # Default template for ImageNet and similar datasets
                template = "a photo of a {}."
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            texts = clip.tokenize(texts).to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    # 评估
    acc = 0.0
    tot_samples = 0
    
    with torch.no_grad():
        if cached_image_batches is not None:
            for image_features, target in cached_image_batches:
                image_features = image_features.to(device)
                target = target.to(device)
                
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
                
                if image_features.dtype != text_features.dtype:
                    text_features = text_features.to(image_features.dtype)
                
                if image_features.device != text_features.device:
                    text_features = text_features.to(image_features.device)
                
                cosine_similarity = image_features @ text_features.t()
                acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
                tot_samples += len(cosine_similarity)
    
    return acc / tot_samples if tot_samples > 0 else 0.0

def precompute_text_features(clip_model: nn.Module, dataset, batch_size: int = 32) -> torch.Tensor:
    """预计算文本特征"""
    device = next(clip_model.parameters()).device
    # Handle datasets with or without template attribute
    if hasattr(dataset, 'template') and dataset.template is not None:
        template = dataset.template[0] if isinstance(dataset.template, (list, tuple)) else dataset.template
    else:
        # Default template for ImageNet and similar datasets
        template = "a photo of a {}."
    texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]
    
    torch.cuda.empty_cache()
    
    text_features_list = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            batch_tokens = clip.tokenize(batch_texts).to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                batch_embeddings = clip_model.encode_text(batch_tokens)
                batch_features = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            text_features_list.append(batch_features.cpu())
            
            del batch_tokens, batch_embeddings, batch_features
            torch.cuda.empty_cache()
    
    text_features = torch.cat(text_features_list, dim=0)
    return text_features

def apply_genes_to_layers(genes: List[torch.Tensor], lora_layers: List[torch.nn.Module]):
    """应用基因到LoRA层"""
    idx = 0
    with torch.no_grad():
        for layer in lora_layers:
            enabled = []
            if hasattr(layer, "q_proj") and hasattr(layer.q_proj, "enable_lora"):
                enabled = list(layer.q_proj.enable_lora)
            for proj in enabled:
                if proj in ("q", "k", "v"):
                    mod = getattr(layer, f"{proj}_proj", None)
                elif proj in ("o", "out"):
                    mod = getattr(layer, "proj", None)
                else:
                    mod = None
                if mod is None or not (hasattr(mod, "w_lora_A") and hasattr(mod, "w_lora_B")):
                    continue

                if idx + 1 >= len(genes):
                    raise RuntimeError("Gene count mismatch")

                a_t = genes[idx].to(mod.w_lora_A.device, dtype=mod.w_lora_A.dtype)
                b_t = genes[idx + 1].to(mod.w_lora_B.device, dtype=mod.w_lora_B.dtype)
                mod.w_lora_A.copy_(a_t)
                mod.w_lora_B.copy_(b_t)
                idx += 2
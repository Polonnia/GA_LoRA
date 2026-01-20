import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import copy
import os

from datasets import build_dataset
from loralib.utils import apply_lora, load_lora, mark_only_lora_as_trainable
from loralib.layers import LoRALayer
from run_utils import *

# --- 核心组件 1: CLIP 分类包装器 ---
# 将 CLIP 包装成标准的 input -> logits 模型，固定文本分类器
class CLIPClassifierWrapper(nn.Module):
    def __init__(self, clip_model, text_features, logit_scale=100.0):
        super().__init__()
        self.clip_model = clip_model
        # 注册为 buffer，不参与梯度更新
        self.register_buffer('text_features', text_features)
        self.logit_scale = logit_scale

    def forward(self, images):
        # 提取图像特征 (这里会用到 LoRA 参数)
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 计算 Logits
        logits = self.logit_scale * image_features @ self.text_features.t()
        return logits

# --- 核心组件 2: 独立的 Sharpness 计算逻辑 ---
def calc_worst_case_sharpness(model, dataloader, rho, n_iters, norm, device):
    """
    计算 Worst-Case Sharpness，仅针对 requires_grad=True 的参数 (LoRA)。
    """
    model.eval()
    
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_train(True)
            
    loss_fn = nn.CrossEntropyLoss()
    
    # 1. 筛选 LoRA 参数
    # 保存原始参数的引用和数据，用于计算后恢复
    orig_params = {
        name: p.clone().detach() 
        for name, p in model.named_parameters() 
        if p.requires_grad
    }
    
    if len(orig_params) == 0:
        print("Warning: No trainable parameters found! Sharpness will be 0.")
        return 0.0

    avg_loss = 0.0
    avg_init_loss = 0.0
    n_batches = 0
    
    # 步长设置
    step_size = rho * 2.0 / n_iters 

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # 计算该 Batch 的初始 Loss
        model.zero_grad()
        with torch.no_grad():
            init_output = model(images)
            init_loss = loss_fn(init_output, labels).item()
        avg_init_loss += init_loss

        # --- APGD / PGD 攻击循环 ---
        
        # 初始化扰动 (随机)
        delta_dict = {}
        for name, p in orig_params.items():
            delta = torch.randn_like(p).to(device)
            if norm == 'l2':
                delta = delta / (delta.norm() + 1e-12) * rho
            elif norm == 'linf':
                delta = delta.sign() * rho
            delta_dict[name] = delta
            
        worst_loss_batch = init_loss

        for _ in range(n_iters):
            # A. 应用扰动
            for name, p in model.named_parameters():
                if name in delta_dict:
                    p.data = orig_params[name] + delta_dict[name]
            
            # B. 前向传播 & 计算梯度
            model.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            
            # C. 梯度上升 (更新扰动)
            with torch.no_grad():
                if norm == 'l2':
                    # 计算所有 LoRA 参数的梯度的总范数
                    grad_norm = torch.norm(torch.stack([
                        p.grad.norm() for name, p in model.named_parameters() if name in delta_dict and p.grad is not None
                    ]))
                    
                    for name, p in model.named_parameters():
                        if name in delta_dict and p.grad is not None:
                            # 归一化梯度并上升
                            g = p.grad / (grad_norm + 1e-12)
                            delta_dict[name] += step_size * g
                            
                elif norm == 'linf':
                    for name, p in model.named_parameters():
                        if name in delta_dict and p.grad is not None:
                            delta_dict[name] += step_size * p.grad.sign()

            # D. 投影 (Projection)
            with torch.no_grad():
                if norm == 'l2':
                    # 计算当前总扰动范数
                    delta_total_norm = torch.norm(torch.stack([
                        d.norm() for d in delta_dict.values()
                    ]))
                    if delta_total_norm > rho:
                        scale = rho / (delta_total_norm + 1e-12)
                        for name in delta_dict:
                            delta_dict[name] *= scale
                elif norm == 'linf':
                    for name in delta_dict:
                        delta_dict[name] = torch.clamp(delta_dict[name], -rho, rho)

            # E. 记录最坏 Loss (可选：每次 step 都检查一下)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in delta_dict:
                        p.data = orig_params[name] + delta_dict[name]
                curr_loss = loss_fn(model(images), labels).item()
                if curr_loss > worst_loss_batch:
                    worst_loss_batch = curr_loss

        # 恢复原始参数，准备下一个 Batch
        for name, p in model.named_parameters():
            if name in orig_params:
                p.data = orig_params[name]
        
        avg_loss += worst_loss_batch
        n_batches += 1

    return (avg_loss - avg_init_loss) / n_batches

def main():
    args = get_arguments()
    args.shots = 2
    set_random_seed(args.seed)
    gpu_id = 5
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading CLIP backbone: {args.backbone}")
    clip_model, preprocess = clip.load(args.backbone, device=device)

    # 1. apply LoRA
    print("Applying LoRA structure...")
    list_lora_layers = apply_lora(args, clip_model)
    clip_model.to(device)  
    mark_only_lora_as_trainable(clip_model)
    # 2. 加载 LoRA 权重
    load_lora(args, list_lora_layers)
    
    # 3. 准备数据
    print(f"Preparing dataset: {args.dataset}")
    # 这里为了简便，我们只取 test set 或 val set 的一部分
    dataset = build_dataset(args.dataset, args.root_path, 1, preprocess, args.batch_size)
    
    # 获取类别文本特征
    print("Building zero-shot classifier...")
    classnames = dataset.classnames # 根据 datasets/imagenet.py 等调整

    # 简单的 Prompt 模板
    templates = ["a photo of a {}."] 
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    
    # 4. 构建 Sharpness 计算模型
    # 此时 model 的 forward 输入图片，输出 Logits
    sharpness_model = CLIPClassifierWrapper(clip_model, zeroshot_weights.t()) # wrapper 期望 (N_classes, Dim)
    
    # 准备 DataLoader (只评测一部分数据以节省时间)
    if hasattr(dataset, 'test'):
        data_source = dataset.test
    else:
        data_source = dataset.val
        
    # 如果数据量太大，可以截断
    indices = list(range(min(len(data_source), args.n_eval_samples)))
    subset = torch.utils.data.Subset(data_source, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 5. 计算 Sharpness
    
    #for r in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
    for r in [0.005,0.001,0.0005,0.0001,0.00005,0.00001]:
        args.rho = r
        sharpness_val = calc_worst_case_sharpness(
            sharpness_model, 
            loader, 
            rho=args.rho, 
            n_iters=args.sharpness_iters, 
            norm=args.norm, 
            device=device
        )

        print(f"opt = {args.opt}, rho = {args.rho}: {sharpness_val:.6f}\n")

if __name__ == '__main__':
    main()
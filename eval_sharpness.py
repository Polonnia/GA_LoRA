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

# --- 辅助函数：保存结果 ---
def save_result(file_path, opt, shots, sharpness_type, rho, value):
    """
    将结果追加写入到文件中
    格式: [Optimizer] [Shots] [Type] Rho=... Value=...
    """
        
    file_name = "sharpness_results.txt"
    file_path = os.path.join(file_path, file_name)
    
    line = f"Opt: {opt}, Shots: {shots}, Type: {sharpness_type}, Rho: {rho}, Sharpness: {value:.6f}\n"
    
    try:
        with open(file_path, 'a') as f:
            f.write(line)
        print(f"Saved to {file_path}: {line.strip()}")
    except Exception as e:
        print(f"Error saving result: {e}")

# --- 核心组件 1: CLIP 分类包装器 ---
class CLIPClassifierWrapper(nn.Module):
    def __init__(self, clip_model, text_features, logit_scale=100.0):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer('text_features', text_features)
        self.logit_scale = logit_scale

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * image_features @ self.text_features.t()
        return logits

# --- 核心组件 2: Worst-Case Sharpness 计算 (原有逻辑) ---
def calc_worst_case_sharpness(model, dataloader, rho, n_iters, norm, device):
    """
    计算 Adaptive Worst-Case Sharpness (S_max)
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_train(True)
            
    loss_fn = nn.CrossEntropyLoss()
    
    orig_params = {
        name: p.clone().detach() 
        for name, p in model.named_parameters() 
        if p.requires_grad
    }
    
    if len(orig_params) == 0:
        return 0.0

    avg_sharpness = 0.0
    n_batches = 0
    step_size_ratio = 2.0 / n_iters 

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # 1. 计算原始 Loss
        model.zero_grad()
        with torch.no_grad():
            output_init = model(images)
            loss_init = loss_fn(output_init, labels)
        
        # 2. 初始化随机扰动
        delta_dict = {}
        for name, p in orig_params.items():
            rand_sign = torch.randint_like(p, high=2) * 2 - 1 
            if norm == 'linf':
                # Adaptive: rho * |w|
                delta = rand_sign * rho * (p.abs() + 1e-12)
            else:
                delta = torch.randn_like(p) * rho # 简化处理 L2
            delta_dict[name] = delta.to(device)

        # 3. PGD 攻击寻找最大 Loss
        worst_loss = loss_init.item() # 至少是初始 loss
        
        for _ in range(n_iters):
            # Apply
            for name, p in model.named_parameters():
                if name in delta_dict:
                    p.data = orig_params[name] + delta_dict[name]
            
            # Backward
            model.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            
            # Update & Project
            with torch.no_grad():
                current_loss_val = loss.item()
                if current_loss_val > worst_loss:
                    worst_loss = current_loss_val

                for name, p in model.named_parameters():
                    if name in delta_dict and p.grad is not None:
                        w_abs = orig_params[name].abs() + 1e-12
                        if norm == 'linf':
                            step = step_size_ratio * rho * w_abs * p.grad.sign()
                            delta_dict[name] += step
                            max_delta = rho * w_abs
                            delta_dict[name] = torch.max(torch.min(delta_dict[name], max_delta), -max_delta)

        # 恢复参数
        for name, p in model.named_parameters():
            if name in orig_params:
                p.data = orig_params[name]
        
        # 4. 计算 Sharpness = Max Loss - Initial Loss
        avg_sharpness += (worst_loss - loss_init.item())
        n_batches += 1

    return avg_sharpness / n_batches

# --- 核心组件 3: Adaptive Average-Case Sharpness 计算 (新增) ---
def calc_adaptive_average_sharpness(model, dataloader, rho, n_samples, device):
    """
    计算 Adaptive Average-Case Sharpness (S_avg)
    原理: E[L(w + noise)] - L(w)
    其中 noise ~ N(0, (rho * |w|)^2)
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_train(True)
    loss_fn = nn.CrossEntropyLoss()

    # 1. 保存原始参数
    orig_params = {
        name: p.clone().detach() 
        for name, p in model.named_parameters() 
        if p.requires_grad
    }
    
    if len(orig_params) == 0:
        return 0.0

    total_sharpness = 0.0
    n_batches = 0

    print(f"Calculating Adaptive Average Sharpness (Samples={n_samples}, Rho={rho})...")

    with torch.no_grad(): # 全程不需要梯度
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # A. 计算原始 Loss
            output_init = model(images)
            loss_init = loss_fn(output_init, labels).item()

            loss_perturbed_sum = 0.0

            # B. 蒙特卡洛采样 (Monte Carlo Sampling)
            for _ in range(n_samples):
                # 施加高斯噪声
                for name, p in model.named_parameters():
                    if name in orig_params:
                        # 核心: Adaptive Noise
                        # noise ~ N(0, 1) * rho * |w|
                        # 这里的 rho 充当了 sigma 的缩放系数
                        w_abs = orig_params[name].abs() + 1e-12
                        noise = torch.randn_like(p) * rho * w_abs
                        p.data = orig_params[name] + noise
                
                # 计算扰动后的 Loss
                output_perturbed = model(images)
                loss_perturbed = loss_fn(output_perturbed, labels).item()
                loss_perturbed_sum += loss_perturbed

            # C. 恢复参数 (为下一个 Batch 做准备)
            for name, p in model.named_parameters():
                if name in orig_params:
                    p.data = orig_params[name]

            # D. 计算当前 Batch 的 Sharpness
            # Sharpness = Average Perturbed Loss - Initial Loss
            avg_perturbed_loss = loss_perturbed_sum / n_samples
            total_sharpness += (avg_perturbed_loss - loss_init)
            n_batches += 1

    return total_sharpness / n_batches

def main():
    args = get_arguments()
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    clip_model, preprocess = clip.load(args.backbone, device=device)
    print(f"evaluating {args.opt} for shot = {args.shots}\n")
    # 1. apply LoRA
    list_lora_layers = apply_lora(args, clip_model)
    clip_model.to(device)  
    mark_only_lora_as_trainable(clip_model)
    # 2. 加载 LoRA 权重
    load_lora(args, list_lora_layers)
    
    # 3. 准备数据
    dataset = build_dataset(args.dataset, args.root_path, 1, preprocess, args.batch_size)
    
    print("Building zero-shot classifier...")
    classnames = dataset.classnames 
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
    sharpness_model = CLIPClassifierWrapper(clip_model, zeroshot_weights.t())
    
    if hasattr(dataset, 'train'):
        data_source = dataset.train
    else:
        data_source = dataset.val
        
    indices = list(range(min(len(data_source), args.n_eval_samples)))
    subset = torch.utils.data.Subset(data_source, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 5. 计算 Sharpness 对比
    print("\n" + "="*50)
    print(f"Starting Sharpness Evaluation for {args.opt} ({args.shots} shots)")
    print(f"Results will be appended to: {args.result_path}")
    print("="*50)

    rhos_worst = [0.002, 0.001, 0.0005, 0.0002] 
    rhos_avg = [0.2, 0.1, 0.05, 0.01]
    
    # --- Worst-Case Sharpness Loop ---
    for r in rhos_worst:
        print(f"\n--- Testing Worst-Case Rho = {r} ---")
        s_max = calc_worst_case_sharpness(
            sharpness_model, 
            loader, 
            rho=r, 
            n_iters=args.sharpness_iters, 
            norm='linf', 
            device=device
        )
        print(f"[Result] Worst-Case Sharpness (S_max): {s_max:.6f}")
        
        # 保存结果
        save_result(
            args.result_path, 
            opt=args.opt, 
            shots=args.shots, 
            sharpness_type="Worst-Case", 
            rho=r, 
            value=s_max
        )

    # --- Average-Case Sharpness Loop ---
    for r in rhos_avg:
        print(f"\n--- Testing Average-Case Rho = {r} ---")
        s_avg = calc_adaptive_average_sharpness(
            sharpness_model, 
            loader, 
            rho=r, 
            n_samples=20, 
            device=device
        )
        print(f"[Result] Average-Case Sharpness (S_avg): {s_avg:.6f}")
        
        # 保存结果
        save_result(
            args.result_path, 
            opt=args.opt, 
            shots=args.shots, 
            sharpness_type="Average-Case", 
            rho=r, 
            value=s_avg
        )

if __name__ == '__main__':
    main()
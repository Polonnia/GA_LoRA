import os
import sys
import json
import math
import random
import time
from typing import List, Optional, Tuple, Dict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip

from utils import *
from loralib.utils import apply_lora, save_lora, load_lora

# ------------------------------
# 全局资源容器 (用于多线程共享)
# ------------------------------
# 结构: { device_id (int): { 'model': model, 'lora_layers': [...], 'data': {...} } }
_GPU_RESOURCES = {}

# ------------------------------
# 全局参数
# ------------------------------
POPULATION_SIZE = 20
NUM_GENERATIONS = 100
MUTATION_RATE = 1.0
MUTATION_RATIO = 1.0
NUM_ELITES = 1
NUM_PARENTS = 4
STD_DEV = 0.1
SEED = 42

EARLY_STOP_ENABLED = True
EARLY_STOP_MARGIN = 0.5
EARLY_STOP_MIN_SAMPLES = 256
EARLY_STOP_TOLERANCE = 0.1

INITIAL_STD_DEV = 0.1
FINAL_STD_DEV = 0.001


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MutationScheduler:
    def __init__(
        self, 
        initial_std: float = INITIAL_STD_DEV,
        final_std: float = FINAL_STD_DEV,
        total_generations: int = NUM_GENERATIONS,
        schedule_type: str = "linear"
    ):
        self.initial_std = initial_std
        self.final_std = final_std
        self.total_generations = total_generations
        self.schedule_type = schedule_type
        
    def get_std_dev(self, generation: int) -> float:
        if generation >= self.total_generations:
            return self.final_std
            
        progress = generation / self.total_generations
        if self.schedule_type == "linear":
            current_std = self.initial_std - (self.initial_std - self.final_std) * progress
        elif self.schedule_type == "exponential":
            decay_factor = math.exp(-3 * progress)
            current_std = self.final_std + (self.initial_std - self.final_std) * decay_factor
        else:
            current_std = self.initial_std
            
        return max(self.final_std, current_std)
    
    def get_mutation_params(self, generation: int) -> dict:
        return {
            "mutation_std": self.get_std_dev(generation),
            "mutation_rate": MUTATION_RATE,
            "mutation_ratio": MUTATION_RATIO
        }


class Chromosome:
    def __init__(self, lora_layers: List[torch.nn.Module]=[], enabled_lora = ['q','v']):
        self.enabled_lora: List[str] = enabled_lora
        self.fitness: Optional[float] = None
        self.need_update: bool = True
        # 基因保存在 CPU 上以节省显存，评估时流式传输到 GPU
        self.genes: List[torch.Tensor] = []
        
        if len(lora_layers):
            first_layer = lora_layers[0]
            if hasattr(first_layer, "enable_lora"):
                self.enabled_lora = list(first_layer.enable_lora)
            for layer in lora_layers:
                # 注意：这里逻辑必须与 apply_genes_to_layers 完全一致
                targets = []
                # 依据 CLIP 结构查找 Q, K, V, O
                if hasattr(layer, "q_proj"): targets.append(layer.q_proj)
                if hasattr(layer, "k_proj"): targets.append(layer.k_proj)
                if hasattr(layer, "v_proj"): targets.append(layer.v_proj)
                if hasattr(layer, "proj"): targets.append(layer.proj) # output projection
                
                for mod in targets:
                    if mod is not None and hasattr(mod, "w_lora_A") and hasattr(mod, "w_lora_B"):
                        # 仅当该模块启用了 LoRA 时保存
                        a = mod.w_lora_A.detach().cpu().clone()
                        b = mod.w_lora_B.detach().cpu().clone()
                        self.genes.append(a)
                        self.genes.append(b)


def init_pop(pop_size: int, list_lora_layers: List[torch.nn.Module]) -> List[Chromosome]:
    population: List[Chromosome] = []
    base = Chromosome(list_lora_layers)
    population.append(base)
    print(f"Gene count per individual: {len(base.genes)}")
    for _ in range(pop_size - 1):
        c = Chromosome()
        # 显式复制 enabled_lora
        c.enabled_lora = base.enabled_lora
        c.genes = [g.clone() + torch.randn_like(g) * STD_DEV for g in base.genes]
        population.append(c)
    return population


def mutate_one(
    chromosome: Chromosome,
    mutation_rate: float,
    mutation_ratio: float,
    mutation_std: float,
) -> Chromosome:
    mutated = Chromosome()
    mutated.enabled_lora = chromosome.enabled_lora
    mutated.genes = [g.clone() for g in chromosome.genes]
    mutated.fitness = None
    mutated.need_update = True
    
    if random.random() < mutation_rate:
        for g in mutated.genes:
            # 在 CPU 上进行变异计算，避免占用 GPU 算力
            mask = (torch.rand_like(g) < mutation_ratio)
            if mask.any():
                noise = torch.randn_like(g) * mutation_std
                g[mask] += noise[mask]
    return mutated


def reproduce(
    population: List[Chromosome],
    mutation_scheduler: MutationScheduler,
    current_generation: int,
    num_elites: int = NUM_ELITES,
    num_parents: int = NUM_PARENTS,
) -> Tuple[List[Chromosome], float]:
    
    pop_size = len(population)
    ranked = sorted(
        population,
        key=lambda c: (c.fitness if c.fitness is not None else -float("inf")),
        reverse=True,
    )

    new_population: List[Chromosome] = []
    # Elitism
    for i in range(min(num_elites, pop_size)):
        ranked[i].need_update = False
        new_population.append(ranked[i])

    mating_pool = ranked[:max(2, min(num_parents, pop_size))]
    mutation_params = mutation_scheduler.get_mutation_params(current_generation)
    
    while len(new_population) < pop_size:
        p = random.choice(mating_pool)
        child = mutate_one(p, **mutation_params)
        new_population.append(child)
        
    return new_population, mutation_params["mutation_std"]


# ------------------------------
# 核心：并行评估逻辑 (ThreadPool)
# ------------------------------

def apply_genes_to_layers_fast(genes: List[torch.Tensor], lora_layers: List[torch.nn.Module], device: torch.device):
    """
    将基因从 CPU 快速应用到指定 GPU 上的 LoRA 层。
    使用 non_blocking=True 加速 H2D 传输。
    """
    idx = 0
    # 遍历顺序必须与 Chromosome.__init__ 中的提取顺序严格一致
    with torch.no_grad():
        for layer in lora_layers:
            targets = []
            if hasattr(layer, "q_proj"): targets.append(layer.q_proj)
            if hasattr(layer, "k_proj"): targets.append(layer.k_proj)
            if hasattr(layer, "v_proj"): targets.append(layer.v_proj)
            if hasattr(layer, "proj"): targets.append(layer.proj)

            for mod in targets:
                if mod is not None and hasattr(mod, "w_lora_A") and hasattr(mod, "w_lora_B"):
                    if idx + 1 >= len(genes):
                        break # Should match, but safety check
                    
                    # 确保 LoRA 处于 active 状态以接受赋值
                    # loralib 的实现通常不需要显式调用 lora_train(True) 就能赋值，
                    # 但为了确保合并逻辑正确，这里直接修改 w_lora_A/B 即可。
                    
                    # 关键优化：copy_(tensor, non_blocking=True)
                    # 只有当 source 在 pinned memory 时 non_blocking 效果最好，但即使不是也比直接 .to() 快
                    mod.w_lora_A.copy_(genes[idx], non_blocking=True)
                    mod.w_lora_B.copy_(genes[idx+1], non_blocking=True)
                    idx += 2


def cls_acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[:1].view(-1).float().sum(0, keepdim=True).cpu().numpy()[0]


def evaluate_worker(chrom_idx: int, chrom_genes: List[torch.Tensor], gpu_id: int):
    """
    线程工作函数：在指定 GPU 上评估一个个体
    """
    device_key = gpu_id
    if device_key not in _GPU_RESOURCES:
        return chrom_idx, 0.0

    res = _GPU_RESOURCES[device_key]
    model = res['model']
    lora_layers = res['lora_layers']
    data = res['data']
    device = torch.device(f"cuda:{gpu_id}")

    # 1. 应用基因 (H2D copy)
    apply_genes_to_layers_fast(chrom_genes, lora_layers, device)

    # 2. 准备数据
    text_features = data['text_features'] # [N_cls, Dim]
    image_data = data['image_data']      # List[(feats, target)] or DataLoader
    
    # 确保 text_features 在正确设备 (初始化时应该已做好，这里double check)
    # text_features = text_features.to(device, non_blocking=True) 

    acc_accum = 0.0
    total_samples = 0
    correct_samples = 0.0

    # 3. 推理循环
    if isinstance(image_data, list):
        with torch.no_grad():
            for img_feats, target in image_data:
                # 移动到 GPU (如果尚未在 GPU)
                img_feats = img_feats.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # 计算 Logits: Image @ Text.T
                # image: [B, Dim], text: [C, Dim] -> logits: [B, C]
                logits = img_feats @ text_features.t()
                
                bs = logits.size(0)
                batch_acc = cls_acc(logits, target) # 返回百分比值
                
                batch_correct = (batch_acc / 100.0) * bs
                correct_samples += batch_correct
                total_samples += bs
                
                # 简单的早停检查 (可选)
                # if total_samples > 500 and (correct_samples/total_samples) < 0.1: break
    
    # 如果没有预计算 Image Features，则运行完整模型 (较慢)
    else:
        # data['image_data'] is loader
        model.eval()
        with torch.no_grad():
            for images, target in image_data:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    img_feats = model.encode_image(images)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                
                logits = img_feats @ text_features.t()
                bs = logits.size(0)
                batch_acc = cls_acc(logits, target)
                correct_samples += (batch_acc / 100.0) * bs
                total_samples += bs

    fitness = (100.0 * correct_samples / max(total_samples, 1))
    return chrom_idx, fitness


@torch.no_grad()
def update_fitness_parallel(
    population: List[Chromosome],
    gpu_ids: List[int]
):
    """
    使用 ThreadPoolExecutor 进行多 GPU 并行评估
    """
    tasks = []
    # 找出需要评估的个体
    for i, chrom in enumerate(population):
        if chrom.need_update:
            tasks.append((i, chrom))
    
    if not tasks:
        return

    # 结果容器
    results = {}
    
    # 启动线程池
    # max_workers 可以设为 gpu 数量，因为计算是 GPU 密集型的，
    # 一个 GPU 同时跑多个 worker 并不一定快（除非 batch 很小显存不满）
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for i, (chrom_idx, chrom) in enumerate(tasks):
            # 简单的 Round-Robin 调度
            gpu_id = gpu_ids[i % len(gpu_ids)]
            # 提交任务
            f = executor.submit(evaluate_worker, chrom_idx, chrom.genes, gpu_id)
            futures.append(f)
        
        # 等待结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", leave=False):
            try:
                c_idx, fit = future.result()
                results[c_idx] = fit
            except Exception as e:
                print(f"Error in evaluation: {e}")
                import traceback
                traceback.print_exc()

    # 回填 Fitness
    for idx, fit in results.items():
        population[idx].fitness = fit
        population[idx].need_update = False


def init_gpu_resources(args, clip_model_base, dataset, gpu_ids: List[int]):
    """
    在主线程中将模型和数据加载到所有目标 GPU 上。
    """
    print(f"Initializing resources on GPUs: {gpu_ids} ...")
    
    # 1. 预计算所有数据的特征 (在 CPU 上做一次，或者在主 GPU 上做)
    #    这里选择在 CPU 上预计算 Image Features，然后分发，
    #    这样可以避免在每个 worker 里重复跑 DataLoader。
    
    # 1a. Text Features
    print("Precomputing text features...")
    # template = dataset.template[0]
    template = "a photo of a {}."
    texts = [template.format(c.replace("_", " ")) for c in dataset.classnames]
    device_0 = torch.device(f"cuda:{gpu_ids[0]}")
    clip_model_base = clip_model_base.to(device_0)
    
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device_0)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            text_features_base = clip_model_base.encode_text(text_tokens)
            text_features_base = text_features_base / text_features_base.norm(dim=-1, keepdim=True)
    
    # 1b. Image Features (Cached)
    # 如果显存足够，建议全部缓存到 GPU；如果不够，缓存到 CPU RAM。
    # 这里缓存到 CPU RAM (List of Tensors)，评估时 copy 到 GPU。
    print("Precomputing image features (this might take a while)...")
    cached_image_data = [] # List[(feats_cpu, target_cpu)]
    
    # 使用 DataLoader 跑一遍
    loader = dataset.train_loader
    for images, target in tqdm(loader, desc="Caching Images"):
        images = images.to(device_0)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            feats = clip_model_base.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        cached_image_data.append((feats.cpu(), target.cpu())) # 存回 CPU
    
    # 2. 将模型复制到每个 GPU
    
    print("Replicating models to workers...")
    
    # # 先把 base model 移回 CPU 以便复制，或者直接 deepcopy
    # clip_model_base = clip_model_base.cpu()
    
    for gid in gpu_ids:
        device = torch.device(f"cuda:{gid}")
        print(f" -> Setting up GPU {gid}")
        
        # 深拷贝模型 (确保每个 GPU 有独立的实例)
        model_replica = deepcopy(clip_model_base)
        # 应用 LoRA
        # 注意：args 里的参数决定了 LoRA 加在哪里
        lora_layers = apply_lora(args, model_replica)
        model_replica = model_replica.to(device)
        model_replica.eval()
        
        # 将 text_features 复制到该 GPU
        text_feats_gpu = text_features_base.to(device)
        
        _GPU_RESOURCES[gid] = {
            'model': model_replica,
            'lora_layers': lora_layers,
            'data': {
                'text_features': text_feats_gpu,
                'image_data': cached_image_data # 这是一个 CPU 列表，共享引用即可
            }
        }
        
        # 预热一下 CUDA
        torch.cuda.synchronize(gid)

    print("Resources initialized.")


def run_lora_ga(args, clip_model, dataset, gpu_ids=[0], num_proc_per_gpu=None):
    set_global_seed(SEED)
    
    # 1. 资源初始化 (多 GPU 准备)
    # 注意：传入的 clip_model 应该是原始未修改的模型
    init_gpu_resources(args, clip_model, dataset, gpu_ids)
    
    # 获取主 GPU 上的 lora_layers 用于初始化种群结构
    main_lora_layers = _GPU_RESOURCES[gpu_ids[0]]['lora_layers']
    
    # 2. 初始化种群
    print(f"Initializing population (Size: {POPULATION_SIZE})...")
    population = init_pop(POPULATION_SIZE, main_lora_layers)
    
    scheduler = MutationScheduler()
    
    # 记录日志
    result_dir = getattr(args, "result_path", ".")
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, "ga_log.json")
    history = []

    best_val_acc = 0.0

    # 3. 进化循环
    for gen in range(NUM_GENERATIONS):
        start_time = time.time()
        
        # 并行评估
        update_fitness_parallel(population, gpu_ids)
        
        # 统计
        fits = [c.fitness for c in population if c.fitness is not None]
        best_fitness = max(fits)
        avg_fitness = sum(fits) / len(fits)
        best_ind = max(population, key=lambda c: c.fitness)
        
        mutation_std = scheduler.get_std_dev(gen)
        
        # 可以在这里加入验证集评估 (Validation)
        # 为了速度，我们可以每 N 代跑一次，或者只跑最好的个体
        val_msg = ""
        if dataset.val_loader is not None and (gen % 5 == 0 or gen == NUM_GENERATIONS - 1):
            # 使用主 GPU 跑验证
            # 需要先把最优基因应用到主 GPU 模型
            res_main = _GPU_RESOURCES[gpu_ids[0]]
            apply_genes_to_layers_fast(best_ind.genes, res_main['lora_layers'], torch.device(f"cuda:{gpu_ids[0]}"))
            
            # 复用 cached_tokens 等如果 evaluate_lora 支持
            # 这里简化调用
            current_val_acc = evaluate_lora(
                res_main['model'], 
                dataset.val_loader, 
                dataset.classnames
            )
            val_msg = f" | Val Acc: {current_val_acc:.2f}%"
            
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                # 保存最佳
                if getattr(args, "save_path", None):
                    save_lora(args, res_main['lora_layers'])

        elapsed = time.time() - start_time
        print(f"[Gen {gen:03d}] Best: {best_fitness:.2f}% | Avg: {avg_fitness:.2f}% | Std: {mutation_std:.4f} | Time: {elapsed:.1f}s{val_msg}")
        
        history.append({
            "gen": gen,
            "best": best_fitness,
            "avg": avg_fitness,
            "val": best_val_acc if val_msg else None
        })
        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)
            
        # 繁殖下一代
        population, _ = reproduce(population, scheduler, gen)

    print("GA Finished.")
    return

import os
import sys
import json
import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
import clip

# 导入本地模块
from utils import *
from loralib.utils import apply_lora, save_lora
from loralib import layers as lora_layers

# 导入并行工作模块
from parallel_worker import WorkerPool, update_fitness_parallel_mp

# ------------------------------
# 全局参数（可按需修改）
# ------------------------------
POPULATION_SIZE = 4
NUM_GENERATIONS = 2
MUTATION_RATE = 1
MUTATION_RATIO = 1
NUM_ELITES = 1
NUM_PARENTS = 2
STD_DEV = 0.1  # 高斯噪声标准差
SEED = 42

# Mutation scheduler parameters
INITIAL_STD_DEV = 0.1  # 初始标准差
FINAL_STD_DEV = 0.001   # 最终标准差


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MutationScheduler:
    """
    变异调度器：随着代数增加线性降低变异强度
    """
    def __init__(
        self, 
        initial_std: float = INITIAL_STD_DEV,
        final_std: float = FINAL_STD_DEV,
        total_generations: int = NUM_GENERATIONS,
        schedule_type: str = "linear"  # "linear" or "exponential"
    ):
        self.initial_std = initial_std
        self.final_std = final_std
        self.total_generations = total_generations
        self.schedule_type = schedule_type
        
    def get_std_dev(self, generation: int) -> float:
        """根据当前代数返回变异标准差"""
        if generation >= self.total_generations:
            return self.final_std
            
        if self.schedule_type == "linear":
            # 线性衰减
            progress = generation / self.total_generations
            current_std = self.initial_std - (self.initial_std - self.final_std) * progress
        elif self.schedule_type == "exponential":
            # 指数衰减
            progress = generation / self.total_generations
            decay_factor = math.exp(-3 * progress)  # 可调整衰减速度
            current_std = self.final_std + (self.initial_std - self.final_std) * decay_factor
        else:
            current_std = self.initial_std
            
        return max(self.final_std, current_std)
    
    def get_mutation_params(self, generation: int) -> dict:
        """返回当前代数的变异参数"""
        return {
            "mutation_std": self.get_std_dev(generation),
            "mutation_rate": MUTATION_RATE,
            "mutation_ratio": MUTATION_RATIO
        }


class Chromosome:
    """
    一个个体：保存 LoRA A/B 因子的拷贝，并能回写到模型中。
    genes 顺序：按 enabled_lora（如 ['q','k','v','o']）遍历每层，依次 push [A, B]。
    """
    def __init__(self, lora_layers: List[torch.nn.Module]=[]):
        self.fitness: Optional[float] = None
        self.need_update: bool = True
        self.genes: List[torch.Tensor] = []
        
        if len(lora_layers):
            first_layer = lora_layers[0]
            if hasattr(first_layer, "enable_lora"):
                self.enabled_lora = list(first_layer.enable_lora)
            for layer in lora_layers:
                for proj in self.enabled_lora:
                    if proj in ("q", "k", "v"):
                        proj_layer = getattr(layer, f"{proj}_proj", None)
                    elif proj in ("o", "out"):
                        proj_layer = getattr(layer, "proj", None)
                    else:
                        proj_layer = None

                    if proj_layer is None:
                        continue
                    if hasattr(proj_layer, "w_lora_A") and hasattr(proj_layer, "w_lora_B"):
                        a = proj_layer.w_lora_A.detach().clone().cpu().requires_grad_(False)
                        b = proj_layer.w_lora_B.detach().clone().cpu().requires_grad_(False)
                        self.genes.append(a)
                        self.genes.append(b)


def init_pop(pop_size: int, list_lora_layers: List[torch.nn.Module]) -> List[Chromosome]:
    """高斯扰动初始化种群（以第一条基因为基础）。"""
    population: List[Chromosome] = []
    base = Chromosome(list_lora_layers)
    population.append(base)
    for _ in range(pop_size - 1):
        c = Chromosome(list_lora_layers)
        c.genes = [g.clone() + torch.randn_like(g) * STD_DEV for g in base.genes]
        population.append(c)
    return population


def crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """按基因粒度二择一整块继承（0.5 概率），保持 tensor 形状一致。"""
    child = Chromosome()
    child.enabled_lora = parent1.enabled_lora[:]
    child.genes = []
    for i, (g1, g2) in enumerate(zip(parent1.genes, parent2.genes)):
        if torch.rand(1).item() < 0.5:
            chosen = g1.detach().clone().cpu()
        else:
            chosen = g2.detach().clone().cpu()
        child.genes.append(chosen)
    return child


def mutate_one(
    chromosome: Chromosome,
    mutation_rate: float = MUTATION_RATE,
    mutation_ratio: float = MUTATION_RATIO,
    mutation_std: float = STD_DEV,
) -> Chromosome:
    """对单个个体突变（按元素独立掩码添加高斯噪声）。"""
    if torch.rand(1).item() < mutation_rate:
        for g in chromosome.genes:
            mask = (torch.rand_like(g) < mutation_ratio)
            if mask.any():
                noise = torch.randn_like(g) * mutation_std
                g[mask] += noise[mask]
    return chromosome


def reproduce(
    population: List[Chromosome],
    mutation_scheduler: MutationScheduler,
    current_generation: int,
    num_elites: int = NUM_ELITES,
    num_parents: int = NUM_PARENTS,
) -> List[Chromosome]:
    """
    Generate the next generation using tournament selection with adaptive mutation.
    """
    pop_size = len(population)
    assert pop_size > 0, "empty population"

    ranked = sorted(
        population,
        key=lambda c: (c.fitness if c.fitness is not None else -float("inf")),
        reverse=True,
    )

    new_population: List[Chromosome] = []
    keep = min(num_elites, pop_size)
    for i in range(keep):
        ranked[i].need_update = False
        new_population.append(ranked[i])

    num_parents = max(2, min(num_parents, pop_size))

    # 截断选择：直接取排序后适应度最高的 num_parents 个个体作为交配池
    mating_pool = ranked[:num_parents]

    # 获取当前代数的变异参数
    mutation_params = mutation_scheduler.get_mutation_params(current_generation)
    
    # 记录当前变异强度（用于日志）
    current_std = mutation_params["mutation_std"]

    while len(new_population) < pop_size:
        p = random.choice(mating_pool)
        child = mutate_one(p, **mutation_params)
        new_population.append(child)   
    return new_population, current_std


def apply_genes_to_layers(genes: List[torch.Tensor], lora_layers: List[torch.nn.Module]):
    """Apply a flat gene list to a list of lora_layers (in-place)."""
    idx = 0
    with torch.no_grad():
        for layer in lora_layers:
            # determine enabled projections order using layer's q_proj
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
                    raise RuntimeError("Gene count mismatch when applying genes to layers.")
                
                # If module supports lora_train, ensure it's unmerged so copying A/B affects forward
                try:
                    if hasattr(mod, 'lora_train'):
                        mod.lora_train(True)
                except Exception:
                    pass

                target_device = mod.w_lora_A.device
                target_dtype_A = mod.w_lora_A.dtype
                target_dtype_B = mod.w_lora_B.dtype
                
                a_t = genes[idx].to(device=target_device, dtype=target_dtype_A)
                b_t = genes[idx + 1].to(device=target_device, dtype=target_dtype_B)
                
                # 复制到LoRA层（copy_是in-place操作，确保设备匹配）
                mod.w_lora_A.copy_(a_t)
                mod.w_lora_B.copy_(b_t)

                # merge back so evaluation sees the updated combined weights
                try:
                    if hasattr(mod, 'lora_train'):
                        mod.lora_train(False)
                        # After merging, ensure the merged weight is on the correct device
                        if hasattr(mod, 'weight') and mod.weight.device != target_device:
                            mod.weight.data = mod.weight.data.to(target_device)
                except Exception:
                    pass
                idx += 2


def precompute_text_features(
    clip_model,
    dataset,
) -> Tuple[Optional[torch.Tensor]]:
    """
    Precompute text features for the dataset with memory optimization.
    """
    device = next(clip_model.parameters()).device
    template = dataset.template[0]
    texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]
    
    torch.cuda.empty_cache()
    
    batch_size = 32
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


def _precompute_text_and_images(
    args,
    clip_model,
    dataset,
    loader,
) -> Tuple[Optional[torch.Tensor], Optional[list]]:
    """
    预计算文本和图像特征缓存
    """
    if getattr(args, "encoder", None) != "text":
        return None, None

    template = dataset.template[0]
    texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]

    with torch.no_grad():
        tokens_cache = clip.tokenize(texts)
        tokens_cache = tokens_cache.cpu()

    image_features_cache = []
    progress = tqdm(loader, desc="Precompute val image features", leave=False)

    sub_batch = 32

    for images, target in progress:
        bs = images.shape[0]
        feats_list = []
        targets_cpu = target.cpu()

        for start in range(0, bs, sub_batch):
            end = min(start + sub_batch, bs)
            img_chunk = images[start:end].cuda(non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    feats_chunk = clip_model.encode_image(img_chunk)
                    feats_chunk = feats_chunk / feats_chunk.norm(dim=-1, keepdim=True)
                feats_list.append(feats_chunk.cpu())
            del img_chunk, feats_chunk
            torch.cuda.empty_cache()

        feats = torch.cat(feats_list, dim=0)
        image_features_cache.append((feats, targets_cpu))

    return tokens_cache, image_features_cache


def run_lora_ga(args, clip_model, dataset, gpu_ids=[0], num_proc_per_gpu=4):
    """
    入口：对应用了 LoRA 的 clip_model 进行 GA 搜索 LoRA 因子。
    """
    set_global_seed(SEED)
    
    # 设置主GPU
    main_gpu = gpu_ids[0]
    torch.cuda.set_device(main_gpu)
    clip_model = clip_model.cuda()

    # 应用 LoRA
    list_lora_layers = apply_lora(args, clip_model)

    # 初始化变异调度器
    mutation_scheduler = MutationScheduler(
        initial_std=INITIAL_STD_DEV,
        final_std=FINAL_STD_DEV,
        total_generations=NUM_GENERATIONS,
        schedule_type="linear"  # 可以选择 "linear" 或 "exponential"
    )
    print(f"Mutation scheduler: {INITIAL_STD_DEV} -> {FINAL_STD_DEV} over {NUM_GENERATIONS} generations")

    # 准备结果目录
    result_dir = getattr(args, "result_path", None) or os.getcwd()
    os.makedirs(result_dir, exist_ok=True)
    gen_log_path = os.path.join(result_dir, "ga_generations.json")
    generation_log = []

    # 预计算特征缓存
    if args.encoder == "text":
        tokens_cache, image_features_cache = _precompute_text_and_images(
            args, clip_model, dataset, dataset.train_loader
        )
        tokens_cache = tokens_cache.cuda()
        image_features_cache = [(feat.cuda(), target) for feat, target in image_features_cache]
        cached_text_features = None
    elif args.encoder == "vision":
        text_features = precompute_text_features(clip_model, dataset)
        cached_text_features = text_features
        tokens_cache = None
        image_features_cache = None
        


    # 准备工作进程池的评估参数
    eval_args = {
        'dataset': args.dataset,
        'root_path': args.root_path,
        'shots': args.shots,
        'batch_size': args.batch_size,
        'cached_text_features': cached_text_features,
        'cached_tokens': tokens_cache,
        'cached_image_features': image_features_cache,
    }

    # 初始化工作进程池
    worker_pool = WorkerPool(gpu_ids, num_proc_per_gpu, args, eval_args)
    worker_pool.initialize_workers()

    # 初始化种群
    pop_size = max(2 * ((POPULATION_SIZE + 1) // 2), NUM_ELITES + 2)
    population = init_pop(pop_size=pop_size, list_lora_layers=list_lora_layers)

    try:
        # 演化主循环
        for gen in range(NUM_GENERATIONS):
            num_to_evaluate = len([c for c in population if c.need_update])
            print(f"Generation {gen}: Evaluating {num_to_evaluate} individuals...")
            
            update_fitness_parallel_mp(population, worker_pool)

            # 统计结果
            best_ind = max(
                population,
                key=lambda c: c.fitness if c.fitness is not None else -float("inf"),
            )
            avg_fitness = sum(
                c.fitness if c.fitness is not None else 0.0 for c in population
            ) / len(population)

            # 验证集评估
            val_acc = 0.0
            if dataset.val_loader is not None:
                apply_genes_to_layers(best_ind.genes, list_lora_layers)
                val_acc = evaluate_lora(
                    clip_model,
                    dataset.val_loader,
                    dataset.classnames,
                    cached_tokens=tokens_cache,
                    cached_image_batches=image_features_cache,
                    cached_text_features=cached_text_features,
                )
                save_lora(args, list_lora_layers)
            
            # 获取当前变异强度
            current_mutation_std = mutation_scheduler.get_std_dev(gen)
            
            print(f"[GA] Gen {gen:03d} | Best fitness={best_ind.fitness:.4f}, val_acc={val_acc:.4f}, mutation_std={current_mutation_std:.6f}")

            generation_log.append({
                "generation": int(gen),
                "best_fitness": best_ind.fitness,
                "avg_fitness": avg_fitness,
                "val_acc": val_acc,
                "mutation_std": current_mutation_std
            })
            
            try:
                with open(gen_log_path, "w", encoding="utf-8") as f:
                    json.dump(generation_log, f, indent=2)
            except Exception as e:
                print(f"[GA] Warning: failed to write generation log to {gen_log_path}: {e}")
            
            # 产生新一代（传入变异调度器和当前代数）
            population, current_std = reproduce(
                population, 
                mutation_scheduler, 
                gen,
                num_elites=NUM_ELITES, 
                num_parents=NUM_PARENTS
            )

    finally:
        # 确保工作进程池被正确关闭
        worker_pool.shutdown()

    # 最终评估和保存
    best_ind = max(
        population,
        key=lambda c: c.fitness if c.fitness is not None else -float("inf"),
    )
    apply_genes_to_layers(best_ind.genes, list_lora_layers)
    
    evaluate(clip_model, "ga", dataset.test_loader, dataset, args.eval_datasets, args.result_path, args.seed, args.root_path)

    if getattr(args, "save_path", None) is not None:
        print(f"[GA] Saving final LoRA weights to {args.save_path} ...")
        save_lora(args, list_lora_layers)

    return
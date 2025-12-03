import os
import sys
import json
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
from utils import *  # 需包含：apply_lora, evaluate_lora, evaluate, save_lora 等
from loralib.utils import (
    apply_lora,
    save_lora,
    load_lora,
)
from loralib import layers as lora_layers
from tqdm import tqdm
import clip  # OpenAI CLIP tokenizer

# 进程内共享：每个worker只加载一次模型和数据到对应GPU
_SHARED_WORKER_MODELS = {}
_SHARED_WORKER_DATA = {}


def _init_gpu_worker(gpu_id: int):
    """Initializer to pin worker processes to a specific GPU."""
    try:
        torch.cuda.set_device(gpu_id)
    except Exception:
        # If CUDA is not available in this context, keep silent to avoid worker crashes.
        pass


# ------------------------------
# 全局参数
# ------------------------------
POPULATION_SIZE = 400
NUM_GENERATIONS = 500
MUTATION_RATE = 1
MUTATION_RATIO = 1
NUM_ELITES = 4
NUM_PARENTS = 20
STD_DEV = 0.1  # 高斯噪声标准差
SEED = 42

EARLY_STOP_ENABLED = True
EARLY_STOP_MARGIN = 0.5       # 允许比当前最好低多少仍继续评估
EARLY_STOP_MIN_SAMPLES = 256  # 至少评估多少样本后才考虑早停
EARLY_STOP_TOLERANCE = 0.1    # 上界判断容忍度（百分比）

# Mutation scheduler parameters
INITIAL_STD_DEV = 0.02  # 初始标准差
FINAL_STD_DEV = 2e-4   # 最终标准差


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
    def __init__(self, lora_layers: List[torch.nn.Module]=[], enabled_lora = ['q','v']):
        self.enabled_lora: List[str] = enabled_lora
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


def mutate_one(
    chromosome: Chromosome,
    mutation_rate: float = MUTATION_RATE,
    mutation_ratio: float = MUTATION_RATIO,
    mutation_std: float = STD_DEV,
) -> Chromosome:
    """对单个个体突变"""
    # 创建染色体的深拷贝
    mutated = Chromosome()
    mutated.genes = [g.clone() for g in chromosome.genes]
    mutated.fitness = None  # 新个体需要重新评估
    mutated.need_update = True
    
    if torch.rand(1).item() < mutation_rate:
        for g in mutated.genes:  # 修改副本，不影响原染色体
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


def evaluate_single_worker(args):
    """工作进程：评估单个染色体"""
    (chrom_idx, chrom_genes, device_idx, lora_args, eval_args) = args

    torch.cuda.set_device(device_idx)
    device = torch.device(f'cuda:{device_idx}')
    global _SHARED_WORKER_MODELS, _SHARED_WORKER_DATA
    clip_model = None
    list_lora_layers = None
    device_key = f"cuda:{device_idx}"

    # 1) 模型 + LoRA 层：每个 worker / GPU 只初始化一次
    if device_key in _SHARED_WORKER_MODELS:
        clip_model, list_lora_layers = _SHARED_WORKER_MODELS[device_key]
    else:
        # 在每个进程中独立创建CLIP模型（仅首次）
        clip_model, _ = clip.load(lora_args.backbone, device=device)
        clip_model = clip_model.to(device)
        clip_model.eval()

        # 确保模型所有参数都在正确的设备上
        for param in clip_model.parameters():
            if param.device != device:
                param.data = param.data.to(device)

        # 应用LoRA（仅初始化一次）
        list_lora_layers = apply_lora(lora_args, clip_model)
        _SHARED_WORKER_MODELS[device_key] = (clip_model, list_lora_layers)
    
    # 应用基因到LoRA层
    apply_genes_to_layers(chrom_genes, list_lora_layers)
    
    # 确保模型和所有参数都在正确的设备上
    clip_model = clip_model.to(device)
    for name, param in clip_model.named_parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    # 确保所有LoRA参数都在正确的设备上
    for layer in list_lora_layers:
        for proj in ['q_proj', 'k_proj', 'v_proj', 'proj']:
            mod = getattr(layer, proj, None)
            if mod is not None:
                if hasattr(mod, 'w_lora_A') and mod.w_lora_A.device != device:
                    mod.w_lora_A.data = mod.w_lora_A.data.to(device)
                if hasattr(mod, 'w_lora_B') and mod.w_lora_B.device != device:
                    mod.w_lora_B.data = mod.w_lora_B.data.to(device)
                # Also check base weight if it exists
                if hasattr(mod, 'weight') and mod.weight.device != device:
                    mod.weight.data = mod.weight.data.to(device)
    
    torch.cuda.synchronize(device)

    # 2) 全局数据缓存：每个 worker / GPU 只搬一次到目标设备
    shared_cache = _SHARED_WORKER_DATA.setdefault(device_key, {})

    cached_text_features = shared_cache.get("cached_text_features")
    if cached_text_features is None and eval_args.get('cached_text_features') is not None:
        cached_text_features = eval_args['cached_text_features'].to(device)
        shared_cache["cached_text_features"] = cached_text_features

    cached_tokens = shared_cache.get("cached_tokens")
    if cached_tokens is None and eval_args.get('cached_tokens') is not None:
        cached_tokens = eval_args['cached_tokens'].to(device)
        shared_cache["cached_tokens"] = cached_tokens

    cached_image_features = shared_cache.get("cached_image_features")
    if cached_image_features is None and eval_args.get('cached_image_features') is not None:
        cached_image_features = [
            (feats.to(device), target.to(device))
            for feats, target in eval_args['cached_image_features']
        ]
        shared_cache["cached_image_features"] = cached_image_features

    # 评估适应度（直接在此实现，不再调用 evaluate_lora）
    clip_model.eval()
    device = next(clip_model.parameters()).device
    
    # ------------------------------
    # 文本特征准备阶段计时
    # ------------------------------
    # 准备 / 复用 文本特征
    if cached_text_features is not None:
        text_features = cached_text_features.to(device)
    else:
        if cached_tokens is not None:
            texts = cached_tokens.to(device)
        else:
            template = "a photo of a {}."
            texts = [
                template.format(classname.replace("_", " "))
                for classname in eval_args["classnames"]
            ]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(texts).to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    if text_features.device != device:
        text_features = text_features.to(device)

    acc = 0.0
    tot_samples = 0
    correct_samples = 0.0
    early_cfg = eval_args.get("early_stop_cfg", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    early_target = early_cfg.get("target", None)
    early_dataset = early_cfg.get("dataset_size", None)
    early_min_samples = early_cfg.get("min_samples", 0)
    early_tolerance = early_cfg.get("tolerance", 0.0)
    early_triggered = False

    def maybe_should_stop(processed: int, correct: float) -> bool:
        if (
            not early_enabled
            or early_target is None
            or early_dataset is None
            or processed < early_min_samples
        ):
            return False
        remaining = max(early_dataset - processed, 0)
        optimistic_correct = correct + remaining
        optimistic_acc = 100.0 * optimistic_correct / max(early_dataset, 1)
        return (optimistic_acc + early_tolerance) < early_target

    with torch.no_grad():
        # 如果有预计算的图像特征，则直接用它们
        if cached_image_features is not None:
            for image_features, target in cached_image_features:
                image_features = image_features.to(device)
                target = target.to(device)

                if text_features.device != image_features.device:
                    text_features = text_features.to(image_features.device)
                if image_features.dtype != text_features.dtype:
                    text_features = text_features.to(image_features.dtype)

                # 仅包含相似度和精度计算的时间
                cosine_similarity = image_features @ text_features.t()

                batch_size = cosine_similarity.size(0)
                batch_acc = cls_acc(cosine_similarity, target)
                batch_correct = (batch_acc / 100.0) * batch_size
                acc += batch_acc * batch_size
                tot_samples += batch_size
                correct_samples += batch_correct

                if maybe_should_stop(tot_samples, correct_samples):
                    early_triggered = True
                    break
        else:
            # 否则按 batch 通过 clip_model 提取图像特征
            train_loader_cfg = eval_args.get("train_loader_cfg")
            if train_loader_cfg is None:
                raise RuntimeError("train_loader configuration missing for image-feature evaluation.")

            local_loader = shared_cache.get("train_loader")
            if local_loader is None:
                cfg = dict(train_loader_cfg)
                dataset = cfg.pop("dataset")
                cfg.setdefault("num_workers", 0)
                cfg.setdefault("persistent_workers", False)
                local_loader = torch.utils.data.DataLoader(dataset, **cfg)
                shared_cache["train_loader"] = local_loader

            for images, target in local_loader:
                # DataLoader + CPU 预处理 + H2D 拷贝时间
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # 图像编码计时
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)


                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                if image_features.dtype != text_features.dtype:
                    text_features = text_features.to(image_features.dtype)
                if image_features.device != text_features.device:
                    text_features = text_features.to(image_features.device)

                # 相似度 & 精度计算计时
                cosine_similarity = image_features @ text_features.t()

                batch_size = cosine_similarity.size(0)
                batch_acc = cls_acc(cosine_similarity, target)
                batch_correct = (batch_acc / 100.0) * batch_size
                acc += batch_acc * batch_size
                tot_samples += batch_size
                correct_samples += batch_correct

                if maybe_should_stop(tot_samples, correct_samples):
                    early_triggered = True
                    break

    if early_triggered and tot_samples > 0:
        print(
            f"[GA][Chromosome {chrom_idx}] early stop after {tot_samples} samples; "
            f"best_possible≈{100.0 * min(correct_samples + max((early_dataset or tot_samples) - tot_samples, 0), early_dataset or tot_samples) / max(early_dataset or tot_samples, 1):.2f}% "
            f"target≈{early_target:.2f}%"
        )

    fitness = (100.0 * correct_samples / max(tot_samples, 1)) if tot_samples else 0.0

    return chrom_idx, float(fitness)


@torch.no_grad()
def update_fitness_parallel_mp(
    population: List[Chromosome],
    lora_args,
    gpu_ids: List[int],
    train_loader,
    classnames,
    cached_text_features=None,
    cached_tokens=None,
    cached_image_features=None,
    num_proc_per_gpu: int = 1,
    executor: Optional[ProcessPoolExecutor] = None,
    executor_map: Optional[Dict[int, ProcessPoolExecutor]] = None,
):
    """使用多进程实现真正的并行评估"""
    
    # 准备需要评估的染色体
    chromosomes_to_evaluate = []
    for idx, chrom in enumerate(population):
        chromosomes_to_evaluate.append((idx, chrom))
    
    if not chromosomes_to_evaluate:
        return
    
    # 准备评估参数
    loader_cfg = None
    if train_loader is not None:
        loader_cfg = {
            "dataset": train_loader.dataset,
            "batch_size": getattr(train_loader, "batch_size", 1),
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": getattr(train_loader, "pin_memory", False),
            "drop_last": getattr(train_loader, "drop_last", False),
            "collate_fn": getattr(train_loader, "collate_fn", None),
            "persistent_workers": False,
        }

    eval_args = {
        'train_loader_cfg': loader_cfg,
        'classnames': classnames,
        'cached_text_features': cached_text_features,
        'cached_tokens': cached_tokens,
        'cached_image_features': cached_image_features,
    }
    
    # 估算训练集大小（用于早停上界）
    dataset_size = None
    try:
        dataset_size = len(train_loader.dataset)
    except Exception:
        dataset_size = None
    if cached_image_features is not None:
        try:
            dataset_size = sum(
                t.shape[0] if hasattr(t, "shape") else len(t)  # type: ignore
                for _, t in cached_image_features
            )
        except Exception:
            pass

    best_known_fitness = None
    existing_fitness = [
        c.fitness for c in population if c.fitness is not None and math.isfinite(c.fitness)
    ]
    if existing_fitness:
        best_known_fitness = max(existing_fitness)

    early_stop_enabled = (
        EARLY_STOP_ENABLED
        and best_known_fitness is not None
        and dataset_size is not None
    )
    eval_args["early_stop_cfg"] = {
        "enabled": bool(early_stop_enabled),
        "target": (best_known_fitness - EARLY_STOP_MARGIN) if best_known_fitness is not None else None,
        "dataset_size": dataset_size,
        "min_samples": EARLY_STOP_MIN_SAMPLES,
        "tolerance": EARLY_STOP_TOLERANCE,
    }
    
    # 分配任务到GPU
    tasks = []
    total_workers = len(gpu_ids) * num_proc_per_gpu
    for i, (chrom_idx, chrom) in enumerate(chromosomes_to_evaluate):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((
            chrom_idx, 
            chrom.genes, 
            gpu_id,
            lora_args,
            eval_args
        ))
    
    # 使用多进程并行评估
    print(f"Starting parallel evaluation of {len(tasks)} chromosomes on {len(gpu_ids)} GPUs...")

    results = [None] * len(population)

    created_executor = False
    futures = []

    if executor_map is not None:
        for task in tasks:
            gpu_id = task[2]
            pool = executor_map.get(gpu_id)
            if pool is None:
                raise RuntimeError(f"No executor found for GPU {gpu_id}")
            futures.append(pool.submit(evaluate_single_worker, task))
    else:
        pool = executor
        if pool is None:
            pool = ProcessPoolExecutor(max_workers=total_workers)
            created_executor = True

        for task in tasks:
            futures.append(pool.submit(evaluate_single_worker, task))

    for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating chromosomes"):
        chrom_idx, fitness = future.result()
        results[chrom_idx] = fitness

    if created_executor:
        pool.shutdown()
    
    # 更新种群适应度
    for idx, fitness in enumerate(results):
        if fitness is not None:
            population[idx].fitness = fitness


def apply_genes_to_layers(genes: List[torch.Tensor], lora_layers: List[torch.nn.Module]):
    """Apply a flat gene list to a list of lora_layers (in-place)."""
    idx = 0
    with torch.no_grad():
        for layer in lora_layers:
            # determine enabled projections order using layer's q_proj
            layer.cuda()
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
    template = "a photo of a {}."
    texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]
    
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
        
    classnames = dataset.classnames
    train_loader = dataset.train_loader

    # 初始化种群
    pop_size = max(2 * ((POPULATION_SIZE + 1) // 2), NUM_ELITES + 2)
    population = init_pop(pop_size=pop_size, list_lora_layers=list_lora_layers)

    mp.set_start_method('spawn', force=True)
    spawn_ctx = mp.get_context("spawn")
    executor_map = {
        gpu_id: ProcessPoolExecutor(
            max_workers=num_proc_per_gpu,
            mp_context=spawn_ctx,
            initializer=_init_gpu_worker,
            initargs=(gpu_id,),
        )
        for gpu_id in gpu_ids
    }

    # 演化主循环
    try:
        for gen in range(NUM_GENERATIONS):
            num_to_evaluate = len([c for c in population if c.need_update])
            print(f"Generation {gen}: Evaluating {num_to_evaluate} individuals on {len(gpu_ids)} GPUs...")

            update_fitness_parallel_mp(
                population,
                args,           # 传递LoRA参数
                gpu_ids,
                train_loader,
                classnames,
                cached_text_features=cached_text_features,
                cached_tokens=tokens_cache,
                cached_image_features=image_features_cache,
                num_proc_per_gpu=num_proc_per_gpu,
                executor_map=executor_map
            )

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
                classnames,
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
        for pool in executor_map.values():
            pool.shutdown()

    # 最终评估和保存
    best_ind = max(
        population,
        key=lambda c: c.fitness if c.fitness is not None else -float("inf"),
    )
    apply_genes_to_layers(best_ind.genes, list_lora_layers)
    
    evaluate(clip_model, "ga", dataset, args.eval_datasets, args.result_path, args.seed, args.root_path)

    if getattr(args, "save_path", None) is not None:
        print(f"[GA] Saving final LoRA weights to {args.save_path} ...")
        save_lora(args, list_lora_layers)

    return

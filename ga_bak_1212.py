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
# 结构: { device_id (int): { 'model': model, 'lora_layers': [...], 'cache': {...} } }
_GPU_RESOURCES = {}

# ------------------------------
# 全局参数
# ------------------------------
POPULATION_SIZE = 100
NUM_GENERATIONS = 2000
NUM_ELITES = 2
NUM_PARENTS = 20
STD_DEV = 0.1
SEED = 42

EARLY_STOP_ENABLED = True
EARLY_STOP_MARGIN = 0.5
EARLY_STOP_MIN_SAMPLES = 256
EARLY_STOP_TOLERANCE = 0.1

INITIAL_STD_DEV = 0.01
FINAL_STD_DEV = 2e-4
INITIAL_MUT_RATIO = 1.0
FINAL_MUT_RATIO = 0.1
INITIAL_MUT_RATE = 0.1
FINAL_MUT_RATE = 0.1
workers_per_gpu = 1 #aojun: 大于1会报错，原因未知

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
        schedule_type: str = "linear",
        initial_ratio: float = INITIAL_MUT_RATIO,
        final_ratio: float = FINAL_MUT_RATIO,
        initial_rate: float = INITIAL_MUT_RATE,
        final_rate: float = FINAL_MUT_RATE,
    ):
        self.initial_std = initial_std
        self.final_std = final_std
        self.total_generations = total_generations
        self.schedule_type = schedule_type

        # 新增：ratio / rate 也做 schedule
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.initial_rate = initial_rate
        self.final_rate = final_rate

    def _schedule(self, start: float, end: float, generation: int) -> float:
        """通用的插值函数，支持 linear / exponential"""
        if generation >= self.total_generations:
            return end

        progress = generation / self.total_generations
        if self.schedule_type == "linear":
            return start + (end - start) * progress
        elif self.schedule_type == "exponential":
            decay = math.exp(-3 * progress)
            return end + (start - end) * decay
        else:
            return start

    def get_std_dev(self, generation: int) -> float:
        current_std = self._schedule(self.initial_std, self.final_std, generation)
        return max(self.final_std, current_std)
    
    def get_mutation_params(self, generation: int) -> dict:
        """同时返回 std / ratio / rate 三个量"""
        return {
            "mutation_std":   self.get_std_dev(generation),
            "mutation_ratio": self._schedule(self.initial_ratio, self.final_ratio, generation),
            "mutation_rate":  self._schedule(self.initial_rate, self.final_rate, generation),
        }



class Chromosome:
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
                        # 存在 CPU 以节省显存
                        a = proj_layer.w_lora_A.detach().cpu().clone()
                        b = proj_layer.w_lora_B.detach().cpu().clone()
                        self.genes.append(a)
                        self.genes.append(b)


def init_pop(pop_size: int, list_lora_layers: List[torch.nn.Module]) -> List[Chromosome]:
    population: List[Chromosome] = []
    base = Chromosome(list_lora_layers)
    population.append(base)
    # print(f"Gene count per individual: {len(base.genes)}")
    for _ in range(pop_size - 1):
        c = Chromosome()
        c.enabled_lora = list(base.enabled_lora)
        c.genes = [g.clone() + torch.randn_like(g) * STD_DEV for g in base.genes]
        population.append(c)
    return population

def crossover_blockwise(parent1: Chromosome, parent2: Chromosome, group_size: int = 2) -> Chromosome:
    """
    结构感知交叉：按LoRA的(A,B)成对作为一个“基因块”做交叉。
    - 假设 genes 列表是 [A1, B1, A2, B2, ...]
    - group_size=2 表示每2个tensor为一组一起从某个父代拷贝。
    """
    assert len(parent1.genes) == len(parent2.genes), "Parents must have same gene length"

    child = Chromosome()
    child.enabled_lora = list(parent1.enabled_lora)
    child.genes = []
    child.fitness = None
    child.need_update = True

    num_genes = len(parent1.genes)
    idx = 0
    while idx < num_genes:
        # 在两个父代中选择一个
        src = parent1 if random.random() < 0.5 else parent2
        # 以 group_size 为步长拷贝整块基因
        for j in range(group_size):
            g_idx = idx + j
            if g_idx >= num_genes:
                break
            g_src = src.genes[g_idx]
            child.genes.append(g_src.clone())
        idx += group_size

    return child

def crossover_blockwise_biased(
    parent1: Chromosome,
    parent2: Chromosome,
    group_size: int = 2,
    # fitness 差值的“温度/尺度”：越大越接近 0.5；越小越偏向强者
    fitness_scale: float = 10.0,
    # 保留多样性：概率下/上限（避免永远只抄一个父代）
    p_min: float = 0.1,
    p_max: float = 1.0,
) -> Chromosome:
    """
    Fitness-biased block crossover
    - genes: [A1,B1,A2,B2,...]
    - group_size=2 => (A,B) 一起拷贝
    - p_take_p1 = clamp(0.5 + (f1-f2)/(2*fitness_scale), [p_min,p_max])
    """
    assert len(parent1.genes) == len(parent2.genes), "Parents must have same gene length"

    f1 = parent1.fitness if parent1.fitness is not None else -1e9
    f2 = parent2.fitness if parent2.fitness is not None else -1e9

    scale = max(float(fitness_scale), 1e-8)
    # 线性映射：diff=0 => 0.5；diff=+scale => 0.75；diff=-scale => 0.25
    p_take_p1 = 0.5 + (f1 - f2) / (2.0 * scale)

    # clamp
    p_take_p1 = max(float(p_min), min(float(p_max), p_take_p1))

    child = Chromosome()
    child.enabled_lora = list(parent1.enabled_lora)
    child.genes = []
    child.fitness = None
    child.need_update = True

    num_genes = len(parent1.genes)
    idx = 0
    while idx < num_genes:
        src = parent1 if random.random() < p_take_p1 else parent2
        for j in range(group_size):
            g_idx = idx + j
            if g_idx >= num_genes:
                break
            child.genes.append(src.genes[g_idx].clone())
        idx += group_size

    return child


def crossover_arithmetic(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    child = Chromosome()
    child.enabled_lora = list(parent1.enabled_lora)
    child.genes = []
    
    alpha = 0.5
    
    for g1, g2 in zip(parent1.genes, parent2.genes):
        # 简单的线性插值
        new_gene = alpha * g1 + (1 - alpha) * g2
        child.genes.append(new_gene)
        
    return child

def mutate_one(
    chromosome: Chromosome,
    mutation_rate: float,
    mutation_ratio: float,
    mutation_std: float,
) -> Chromosome:
    mutated = Chromosome()
    mutated.enabled_lora = list(chromosome.enabled_lora)
    mutated.genes = [g.clone() for g in chromosome.genes]
    mutated.fitness = None
    mutated.need_update = True
    
    #if random.random() < mutation_rate:
    for g in mutated.genes:
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
    keep = min(num_elites, pop_size)
    for i in range(keep):
        ranked[i].need_update = False
        new_population.append(ranked[i])

    # 截断选择用于构建交配池
    num_parents = max(2, min(num_parents, pop_size))
    mating_pool = ranked[:num_parents]

    mutation_params = mutation_scheduler.get_mutation_params(current_generation)
    
    while len(new_population) < pop_size:
        # 这里的随机选择模拟了从优良父代中随机交配变异
        # 选两个父母（简单随机，可以改成 tournament selection）
        if random.random() < 0.3:
            parent1, parent2 = random.sample(mating_pool, 2)
            child = crossover_arithmetic(parent1, parent2)
        else:
            parent = random.choice(mating_pool)
            child = mutate_one(parent, **mutation_params)
        new_population.append(child)
        
    return new_population, mutation_params["mutation_std"]


# ------------------------------
# 核心逻辑修复
# ------------------------------

def apply_genes_to_layers_fast(genes: List[torch.Tensor], lora_layers: List[torch.nn.Module], enabled_lora: List[str]):
    """
    将基因应用到 GPU 上的 LoRA 层。
    增加 lora_train 切换逻辑以确保 merge 权重正确更新（如果实现依赖它）。
    """
    idx = 0
    with torch.no_grad():
        for layer in lora_layers:
            # 使用传入的 enabled_lora 列表来确定顺序，与 Chromosome.__init__ 一致
            for proj in enabled_lora:
                if proj in ("q", "k", "v"):
                    mod = getattr(layer, f"{proj}_proj", None)
                elif proj in ("o", "out"):
                    mod = getattr(layer, "proj", None)
                else:
                    mod = None

                if mod is None or not (hasattr(mod, "w_lora_A") and hasattr(mod, "w_lora_B")):
                    continue
                
                if idx + 1 >= len(genes):
                    break

                # 切换到训练模式以便能够修改 A/B
                if hasattr(mod, 'lora_train'):
                    mod.lora_train(True)
                
                # non_blocking=True 加速 H2D 复制
                mod.w_lora_A.copy_(genes[idx], non_blocking=True)
                mod.w_lora_B.copy_(genes[idx+1], non_blocking=True)
                
                # 切换回 eval 模式 (这通常会触发 merge)
                if hasattr(mod, 'lora_train'):
                    mod.lora_train(False)
                    # 确保 merge 后的 weight 在正确设备
                    if hasattr(mod, 'weight') and mod.weight.device != mod.w_lora_A.device:
                        mod.weight.data = mod.weight.data.to(mod.w_lora_A.device)

                idx += 2


def cls_acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[:1].view(-1).float().sum(0, keepdim=True).cpu().numpy()[0]

#key
def preload_dataset_to_gpu(dataset_loader, device):
    """
    将整个数据集预加载到GPU显存（如果显存足够）
    否则使用分块缓存策略
    """
    all_images = []
    all_targets = []
    
    print(f"Preloading dataset to GPU {device}...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataset_loader, desc=f"Loading to {device}")):
            # 转换到设备并保持
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 使用半精度节省显存
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                all_images.append(images.half() if images.dtype == torch.float32 else images)
                all_targets.append(targets)
            
            # 显存检查
            if torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory > 0.95:
                print(f"Warning: GPU {device} memory usage >95%")
    
    return all_images, all_targets

def evaluate_worker(chrom_idx: int, chrom_genes: List[torch.Tensor], chrom_enabled_lora: List[str], gpu_id: int, encoder_type: str, early_stop_cfg: dict):
    """
    线程工作函数：修复了核心评估逻辑。
    根据 encoder_type 决定哪部分特征是静态缓存，哪部分需要实时计算。
    """
    device_key = gpu_id
    if device_key not in _GPU_RESOURCES:
        return chrom_idx, 0.0

    res = _GPU_RESOURCES[device_key]
    model = res['model']
    lora_layers = res['lora_layers']
    cache = res['cache']
    device = torch.device(f"cuda:{gpu_id}")

    # 1. 应用基因 (修改模型权重)
    apply_genes_to_layers_fast(chrom_genes, lora_layers, chrom_enabled_lora)

    acc = 0.0
    tot_samples = 0
    correct_samples = 0.0
    
    # 解析早停参数
    early_enabled = early_stop_cfg.get("enabled", False)
    early_target = early_stop_cfg.get("target", None)
    early_dataset = early_stop_cfg.get("dataset_size", None)
    early_min = early_stop_cfg.get("min_samples", 0)
    early_tol = early_stop_cfg.get("tolerance", 0.0)

    # -----------------------------------------------------
    # 场景 A: 优化 Vision Encoder (Text 是静态的)
    # -----------------------------------------------------
    if encoder_type == 'vision':
        # Text features 已经在 init_gpu_resources 计算好并放在 GPU 上
        text_features = cache['text_features'] # [N_cls, Dim]
        data_loader = cache['train_loader']    # 实际的数据加载器
        
        #key
        preloaded_images = cache['preloaded_images']
        preloaded_targets = cache['preloaded_targets']
        
        # 确保 text_features 类型匹配
        dtype = next(model.parameters()).dtype
        if text_features.dtype != dtype:
            text_features = text_features.to(dtype)

        # with torch.no_grad():
        #     for images, target in data_loader:
        #         images = images.to(device, non_blocking=True)
        #         target = target.to(device, non_blocking=True)
        #key
        with torch.no_grad():
            for batch_idx, (images, target) in enumerate(zip(preloaded_images, preloaded_targets)):
                # 注意：images和targets已在GPU，无需传输！

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 计算相似度
                cosine_similarity = image_features @ text_features.t()
                
                batch_size = cosine_similarity.size(0)
                # 这里的 batch_acc 实际是 "这一批预测正确的样本数"
                batch_correct = cls_acc(cosine_similarity, target)

                tot_samples += batch_size
                correct_samples += batch_correct

                # 早停检查（使用正确样本数）
                if early_enabled and early_target is not None and tot_samples >= early_min:
                    # 剩余最多还能看到多少样本
                    remaining = max(early_dataset - tot_samples, 0)
                    # 最乐观情况：后面 remaining 个样本全都预测正确
                    optimistic_acc = 100.0 * (correct_samples + remaining) / max(early_dataset, 1)
                    if (optimistic_acc + early_tol) < early_target:
                        break

    # -----------------------------------------------------
    # 场景 B: 优化 Text Encoder (Image 是静态的)
    # -----------------------------------------------------
    elif encoder_type == 'text':
        # Image features 已经在 init_gpu_resources 计算好 (Cached)
        cached_image_batches = cache['image_features_batches'] # List of (feats, target)
        tokens = cache['text_tokens'] # [N_cls, 77]

        # 实时计算 Text Features (因为 Text Encoder 被突变了)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 遍历缓存的图像特征进行计算
        with torch.no_grad():
            for image_features, target in cached_image_batches:
                image_features = image_features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                if image_features.dtype != text_features.dtype:
                    image_features = image_features.to(text_features.dtype)

                cosine_similarity = image_features @ text_features.t()
                
                batch_size = cosine_similarity.size(0)
                # 这里 batch_correct 也是这一批的正确样本数
                batch_correct = cls_acc(cosine_similarity, target)
                
                tot_samples += batch_size
                correct_samples += batch_correct

                if early_enabled and early_target is not None and tot_samples >= early_min:
                    remaining = max(early_dataset - tot_samples, 0)
                    optimistic_acc = 100.0 * (correct_samples + remaining) / max(early_dataset, 1)
                    if (optimistic_acc + early_tol) < early_target:
                        break


    fitness = (100.0 * correct_samples / max(tot_samples, 1)) if tot_samples > 0 else 0.0
    return chrom_idx, float(fitness)


@torch.no_grad()
def update_fitness_parallel(
    population: List[Chromosome],
    gpu_ids: List[int],
    args,
    early_stop_cfg: dict
):
    tasks = []
    for i, chrom in enumerate(population):
        if chrom.need_update:
            tasks.append((i, chrom))
    
    if not tasks:
        return

    encoder_type = getattr(args, "encoder", "vision") # 默认为 vision

    with ThreadPoolExecutor(max_workers=len(gpu_ids)*workers_per_gpu) as executor:
        futures = []
        for i, (chrom_idx, chrom) in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            # 将基因传给 worker
            f = executor.submit(
                evaluate_worker, 
                chrom_idx, 
                chrom.genes, 
                chrom.enabled_lora,
                gpu_id, 
                encoder_type,
                early_stop_cfg
            )
            futures.append(f)
        
        results = {}
        for future in as_completed(futures):
            try:
                c_idx, fit = future.result()
                results[c_idx] = fit
            except Exception as e:
                print(f"Error in evaluation: {e}")
                import traceback
                traceback.print_exc()

    for idx, fit in results.items():
        population[idx].fitness = fit
        population[idx].need_update = False


def init_gpu_resources(args, clip_model_base, dataset, gpu_ids: List[int]):
    """
    根据 args.encoder 类型，智能缓存静态特征。
    """
    print(f"Initializing resources on GPUs: {gpu_ids} ...")
    encoder_type = getattr(args, "encoder", "vision")
    
    # ------------------------------
    # 准备共享数据 (在 CPU 或 主GPU)
    # ------------------------------
    shared_cache_cpu = {}
    device_0 = torch.device(f"cuda:{gpu_ids[0]}")
    clip_model_base = clip_model_base.to(device_0)
    clip_model_base.eval()

    # 1. 准备文本 Token
    template = "a photo of a {}."
    texts = [template.format(c.replace("_", " ")) for c in dataset.classnames]
    text_tokens = clip.tokenize(texts) # [N_cls, 77] (CPU)

    if encoder_type == 'vision':
        # 如果训练 Vision，Text Features 是不变的，计算一次缓存到 GPU
        print("Precomputing STATIC text features for Vision optimization...")
        with torch.no_grad():
            toks_dev = text_tokens.to(device_0)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                text_feats = clip_model_base.encode_text(toks_dev)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            shared_cache_cpu['text_features'] = text_feats.cpu() # 暂时放 CPU，后面分发
        
        # Image 数据需要 Loader，因为每次都要过模型
        # 这里我们为每个 GPU 创建一个独立的 loader 引用（Dataset共享，Loader轻量）
        # 或者直接复用 dataset 的 loader 配置
        
    elif encoder_type == 'text':
        # 如果训练 Text，Image Features 是不变的，计算一次缓存
        print("Precomputing STATIC image features for Text optimization...")
        shared_cache_cpu['text_tokens'] = text_tokens # Token 需要传给模型

        image_feats_cache = []
        loader = dataset.train_loader
        # 预计算所有图像特征
        for images, target in tqdm(loader, desc="Caching Images"):
            images = images.to(device_0)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                feats = clip_model_base.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            # 存回 CPU 列表，避免显存溢出，评估时流式 copy
            image_feats_cache.append((feats.cpu(), target.cpu()))
        shared_cache_cpu['image_features_batches'] = image_feats_cache

    # ------------------------------
    # 分发到 Worker GPUs
    # ------------------------------
    print("Replicating models and cache to workers...")
    
    # 获取 Loader 配置
    train_loader = dataset.train_loader
    loader_args = {
        "dataset": train_loader.dataset,
        "batch_size": train_loader.batch_size,
        "shuffle": False, # 评估不需要 shuffle，但也无所谓
        "num_workers": 0, # 多线程下 loader worker 设为 0 更安全
        "drop_last": False,
        "pin_memory": True,
    }

    for gid in gpu_ids:
        device = torch.device(f"cuda:{gid}")
        
        # 1. 模型副本
        model_replica = deepcopy(clip_model_base)
        model_replica = model_replica.to(device)
        lora_layers = apply_lora(args, model_replica) # 应用 LoRA 结构
        model_replica = model_replica.to(device)
        model_replica.eval()
        
        # 2. 缓存副本
        local_cache = {}
        
        if encoder_type == 'vision':
            # Text Feats -> GPU
            local_cache['text_features'] = shared_cache_cpu['text_features'].to(device)
            # Loader -> New Instance
            local_cache['train_loader'] = torch.utils.data.DataLoader(**loader_args)
            #key
            local_cache['preloaded_images'], local_cache['preloaded_targets'] = preload_dataset_to_gpu(
                train_loader, device
            )
            
        elif encoder_type == 'text':
            # Text Tokens -> GPU
            local_cache['text_tokens'] = shared_cache_cpu['text_tokens'].to(device)
            # Image Batches -> Keep CPU list (shared ref), moved to GPU in loop
            local_cache['image_features_batches'] = shared_cache_cpu['image_features_batches']

        _GPU_RESOURCES[gid] = {
            'model': model_replica,
            'lora_layers': lora_layers,
            'cache': local_cache
        }
        
        torch.cuda.synchronize(gid)

    print("Resources initialized.")

def save_mating_pool_lora(
    args,
    res_main: dict,
    mating_pool: List[Chromosome],
    prefix: str = "mp_final",
):
    """
    保存 mating pool 中每个个体对应的 LoRA 权重。
    通过临时修改 args.filename 来生成不同文件名。
    """
    if not getattr(args, "save_path", None):
        print("[GA] args.save_path is None, skip saving mating pool.")
        return
    
    # 额外保存一个索引，便于对照 fitness
    old_save_path = args.save_path
    save_root = os.path.join(old_save_path, "mating_pool")
    os.makedirs(save_root, exist_ok=True)
    args.save_path = save_root
    old_filename = getattr(args, "filename", "default")

    for i, ind in enumerate(mating_pool):
        fit = ind.fitness if ind.fitness is not None else -1.0

        # 将该个体基因应用到主 GPU 模型的 LoRA 层
        apply_genes_to_layers_fast(ind.genes, res_main["lora_layers"], ind.enabled_lora)

        # 生成唯一文件名：包含序号与 fitness（可选）
        safe_fit = f"{fit:.4f}".replace(".", "p")
        args.filename = f"{old_filename}_{prefix}_{i:03d}_fit{safe_fit}"

        save_lora(args, res_main["lora_layers"])  # 使用现有 save_lora 逻辑落盘

    # 恢复 filename，避免影响后续流程
    args.save_path = old_save_path
    args.filename = old_filename

def run_lora_ga(args, clip_model, dataset, gpu_ids=[0], num_proc_per_gpu=None):
    set_global_seed(SEED)
    
    # 1. 资源初始化
    init_gpu_resources(args, clip_model, dataset, gpu_ids)
    
    # 从主 GPU 获取层结构模板
    main_lora_layers = _GPU_RESOURCES[gpu_ids[0]]['lora_layers']
    
    # 2. 初始化种群
    print(f"Initializing population (Size: {POPULATION_SIZE})...")
    population = init_pop(POPULATION_SIZE, main_lora_layers)
    
    scheduler = MutationScheduler()
    
    result_dir = getattr(args, "result_path", ".")
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, "ga_generations.json")
    generation_log = []
    generation_log.append({
        "type": "meta",
        "pop_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "num_elites": NUM_ELITES,
        "num_parents": NUM_PARENTS,

        "mutation_scheduler": {
            "initial_std": INITIAL_STD_DEV,
            "final_std": FINAL_STD_DEV,
            "initial_ratio": INITIAL_MUT_RATIO,
            "final_ratio": FINAL_MUT_RATIO,
            "initial_rate": INITIAL_MUT_RATE,
            "final_rate": FINAL_MUT_RATE,
        },
    })
    print(f"Starting GA for {NUM_GENERATIONS} generations...")

    best_val_acc = 0.0
    
    # 准备早停配置
    dataset_size = len(dataset.train_loader.dataset)
    early_stop_cfg = {
        "enabled": EARLY_STOP_ENABLED,
        "target": None, # 动态更新
        "dataset_size": dataset_size,
        "min_samples": EARLY_STOP_MIN_SAMPLES,
        "tolerance": EARLY_STOP_TOLERANCE,
    }
    
    # 3. 进化循环
    for gen in range(NUM_GENERATIONS):
        start_time = time.time()
        
        # 更新早停目标 (基于当前最好)
        best_known = max([c.fitness for c in population if c.fitness is not None], default=0.0)
        if best_known > 0:
            early_stop_cfg["target"] = best_known - EARLY_STOP_MARGIN

        # 并行评估
        num_eval = len([c for c in population if c.need_update])
        # print(f"Generation {gen}: Evaluating {num_eval} individuals...")
        
        update_fitness_parallel(population, gpu_ids, args, early_stop_cfg)
        
        # 统计
        valid_population = [c for c in population if c.fitness is not None]
        if valid_population:  # 确保列表不为空
            fits = [c.fitness for c in valid_population]
            best_fitness = max(fits)
            avg_fitness = sum(fits) / len(fits)
            best_ind = max(valid_population, key=lambda c: c.fitness)
        else:
            best_fitness = avg_fitness = None
            best_ind = None
        
        mutation_std = scheduler.get_std_dev(gen)
        
        # 验证集评估 (逻辑与 Code 1 相同，应用最佳基因后评估)
        val_acc = 0.0
        if dataset.val_loader is not None:
             # 在主 GPU 上进行验证
            res_main = _GPU_RESOURCES[gpu_ids[0]]
            apply_genes_to_layers_fast(best_ind.genes, res_main['lora_layers'], best_ind.enabled_lora)
            
            try:
                # 可能需要更改
                val_acc = evaluate_lora(
                    res_main['model'],
                    dataset.val_loader,
                    dataset.classnames
                )
            except Exception:
                pass # 如果 evaluate_lora 接口不匹配暂且跳过

            if getattr(args, "save_path", None):
                 save_lora(args, res_main['lora_layers'])

        elapsed = time.time() - start_time
        print(f"[GA] Gen {gen:03d} | Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Val={val_acc:.4f}, Std={mutation_std:.6f}, Time={elapsed:.1f}s")
        
        generation_log.append({
            "generation": int(gen),
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "val_acc": val_acc,
            "mutation_std": mutation_std
        })
        with open(log_file, "w") as f:
            json.dump(generation_log, f, indent=2)
            
        # 繁殖下一代
        population, _ = reproduce(population, scheduler, gen)

    # 最终保存
    valid_population = [c for c in population if c.fitness is not None]
    if valid_population:  # 确保列表不为空
        ranked = sorted(valid_population, key=lambda c: c.fitness, reverse=True)
        best_ind = ranked[0]
        final_mating_pool = ranked[:min(NUM_PARENTS, len(ranked))]
    else:
        ranked = []
        best_ind = None
        final_mating_pool = []
        
    res_main = _GPU_RESOURCES[gpu_ids[0]]
    
    apply_genes_to_layers_fast(best_ind.genes, res_main['lora_layers'], best_ind.enabled_lora)
    if getattr(args, "save_path", None):
        print(f"[GA] Saving final LoRA weights to {args.save_path} ...")
        save_lora(args, res_main['lora_layers'])
        
    if final_mating_pool and getattr(args, "save_path", None):
        print(f"[GA] Saving ALL mating pool individuals ({len(final_mating_pool)}) to {args.save_path} ...")
        save_mating_pool_lora(args, res_main, final_mating_pool, prefix="mp_final")

    plot_ga_progress_from_log(generation_log, args)
    print("GA Finished.")
    return
import os
import sys
import json
import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *  # 需包含：apply_lora, evaluate_lora, evaluate, save_lora 等
from loralib.utils import (
    apply_lora,
    save_lora,
    load_lora,
)
from loralib import layers as lora_layers
from tqdm import tqdm
import clip  # OpenAI CLIP tokenizer


# ------------------------------
# 全局参数（可按需修改）
# ------------------------------
POPULATION_SIZE = 10
NUM_GENERATIONS = 200
MUTATION_RATE = 0.30
MUTATION_RATIO = 0.20
NUM_ELITES = 4
NUM_PARENTS = 2
STD_DEV = 0.1  # 高斯噪声标准差
SEED = 42


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Chromosome:
    """
    一个个体：保存 LoRA A/B 因子的拷贝（CPU、无梯度），并能回写到模型中。
    genes 顺序：按 enabled_lora（如 ['q','k','v','o']）遍历每层，依次 push [A, B]。
    """
    def __init__(self, lora_layers: List[torch.nn.Module]=[]):
        self.fitness: Optional[float] = None

        first_layer = lora_layers[0]
        if hasattr(first_layer, "enable_lora"):
            self.enabled_lora = list(first_layer.enable_lora)
            print(f"[GA] Enabled LoRA projections: {self.enabled_lora}")

        self.genes: List[torch.Tensor] = []
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

    # def apply_chromosome_to_model(self):
    #     """将 genes 拷回原模型的 LoRA 权重中（按 enabled_lora 顺序）。"""
    #     idx = 0
    #     with torch.no_grad():
    #         for layer in self.lora_layers:
    #             for proj in self.enabled_lora:
    #                 if proj in ("q", "k", "v"):
    #                     mod = getattr(layer, f"{proj}_proj", None)
    #                 elif proj in ("o", "out"):
    #                     mod = getattr(layer, "proj", None)
    #                 else:
    #                     mod = None
    #                 if mod is None or not (hasattr(mod, "w_lora_A") and hasattr(mod, "w_lora_B")):
    #                     continue

    #                 if idx + 1 >= len(self.genes):
    #                     raise RuntimeError(
    #                         "Gene count mismatch when applying chromosome to model."
    #                     )

    #                 a_t = self.genes[idx].to(mod.w_lora_A.device, dtype=mod.w_lora_A.dtype)
    #                 b_t = self.genes[idx + 1].to(mod.w_lora_B.device, dtype=mod.w_lora_B.dtype)
    #                 mod.w_lora_A.copy_(a_t)
    #                 mod.w_lora_B.copy_(b_t)
    #                 idx += 2


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
        # if i == 0:
        #     try:
        #         print(f"[GA] Crossover first-gene chosen_mean={chosen.mean().item():.6f}")
        #     except Exception:
        #         pass
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
    num_elites: int = NUM_ELITES,
    num_parents: int = NUM_PARENTS,
) -> List[Chromosome]:
    """
    生成下一代：
      - 精英保留：深拷贝前 num_elites 个体
      - 轮盘赌挑父母池（含全 0 兜底）
      - 交叉 + 变异 生成其余子代
    """
    pop_size = len(population)
    assert pop_size > 0, "empty population"

    # 排序（高 fitness 在前；None 视为 -inf）
    ranked = sorted(
        population,
        key=lambda c: (c.fitness if c.fitness is not None else -float("inf")),
        reverse=True,
    )

    # 精英拷贝
    new_population: List[Chromosome] = []
    keep = min(num_elites, pop_size)
    for i in range(keep):
        new_population.append(ranked[i])

    # 轮盘赌概率
    fits = torch.tensor(
        [max(0.0, c.fitness if c.fitness is not None else 0.0) for c in ranked],
        dtype=torch.float32,
    )
    total = float(fits.sum())
    if total <= 0:
        probs = torch.ones(len(ranked), dtype=torch.float32) / len(ranked)
    else:
        probs = fits / total

    # 父母池（有放回）
    num_parents = max(2, min(num_parents, pop_size))
    indices = torch.multinomial(probs, num_parents, replacement=True)
    mating_pool = [ranked[i] for i in indices.tolist()]

    # 生成子代直至满员（保证偶数填充）
    while len(new_population) < pop_size:
        p1, p2 = random.choice(mating_pool), random.choice(mating_pool)
        child1 = mutate_one(crossover(p1, p2))
        new_population.append(child1)
        if len(new_population) < pop_size:
            child2 = mutate_one(crossover(p2, p1))
            new_population.append(child2)

    return new_population


@torch.no_grad()
def update_fitness(
    population: List[Chromosome],
    clip_model: torch.nn.Module,
    list_lora_layers: List[torch.nn.Module],
    train_loader,
    dataset,
    cached_text_features=None,
    cached_tokens=None,
    cached_image_features=None,
):
    """逐个个体回写权重并用 evaluate_lora 计算适应度。"""
    for chromosome in population[NUM_ELITES:]:
        apply_genes_to_layers(chromosome.genes, list_lora_layers)
        chromosome.fitness = evaluate_lora(
            clip_model,
            train_loader,
            dataset,
            cached_text_features=cached_text_features,
            cached_tokens=cached_tokens,
            cached_image_batches=cached_image_features,
        )
        print(f"[GA] Chromosome fitness: {chromosome.fitness:.4f}")


def collect_lora_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Collect LoRA-enabled MHA layers from model into a list, same structure as apply_lora returns."""
    loras = []
    for m in model.modules():
        # heuristic: LoRA MHA wrapper contains attributes like q_proj
        if hasattr(m, "q_proj"):
            # ensure this module actually has w_lora_A somewhere
            qp = getattr(m, "q_proj", None)
            if qp is not None and hasattr(qp, "w_lora_A"):
                loras.append(m)
    return loras


def apply_genes_to_layers(genes: List[torch.Tensor], lora_layers: List[torch.nn.Module]):
    """Apply a flat gene list to a list of lora_layers (in-place)."""
    idx = 0
    with torch.no_grad():
        for layer in lora_layers:
            layer.cuda()
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
                        # unmerge (make LoRA params active)
                        mod.lora_train(True)
                except Exception:
                    pass

                a_t = genes[idx].to(mod.w_lora_A.device, dtype=mod.w_lora_A.dtype)
                b_t = genes[idx + 1].to(mod.w_lora_B.device, dtype=mod.w_lora_B.dtype)
                mod.w_lora_A.copy_(a_t)
                mod.w_lora_B.copy_(b_t)

                # merge back so evaluation sees the updated combined weights
                try:
                    if hasattr(mod, 'lora_train'):
                        mod.lora_train(False)
                except Exception:
                    pass
                idx += 2


# def update_fitness_parallel(
#     population: List[Chromosome],
#     device_models: List[torch.nn.Module],
#     device_lora_layers: List[List[torch.nn.Module]],
#     train_loader,
#     dataset,
#     cached_tokens=None,
#     cached_image_features=None,
# ):
#     """Evaluate population in parallel across multiple device_models.

#     Each device_model is a deepcopy of the base model moved to a specific device. We distribute
#     chromosomes round-robin to devices and evaluate them concurrently with threads.
#     """
#     num_devices = len(device_models)
#     if num_devices == 0:
#         raise RuntimeError("No device models provided for parallel evaluation.")

#     # partition indices round-robin
#     assignments = [[] for _ in range(num_devices)]
#     for i, chrom in enumerate(population):
#         assignments[i % num_devices].append((i, chrom))

#     def eval_chunk(device_idx, items):
#         # ensure this worker thread uses the correct CUDA device
#         try:
#             if torch.cuda.is_available():
#                 torch.cuda.set_device(device_idx)
#         except Exception:
#             pass

#         model = device_models[device_idx]
#         lora_layers = device_lora_layers[device_idx]
#         dev = next(model.parameters()).device if any(p is not None for p in model.parameters()) else torch.device("cpu")

#         # prepare per-device cached inputs
#         tokens = None
#         if cached_tokens is not None:
#             try:
#                 tokens = cached_tokens.to(dev)
#             except Exception:
#                 tokens = cached_tokens

#         image_feats = None
#         if cached_image_features is not None:
#             image_feats = []
#             for feats, target in cached_image_features:
#                 try:
#                     image_feats.append((feats.to(dev), target))
#                 except Exception:
#                     image_feats.append((feats, target))

#         results = []
#         for idx, chrom in items:
#             # apply genes into this device's model layers
#             apply_genes_to_layers(chrom.genes, lora_layers)
#             # run evaluation on this device model
#             fit = evaluate_lora(
#                 model,
#                 train_loader,
#                 dataset,
#                 cached_tokens=tokens,
#                 cached_image_batches=image_feats,
#             )
#             results.append((idx, float(fit)))
#         return results

#     # run threads
#     with ThreadPoolExecutor(max_workers=num_devices) as exe:
#         futures = [exe.submit(eval_chunk, i, assignments[i]) for i in range(num_devices)]
#         for fut in as_completed(futures):
#             for idx, fit in fut.result():
#                 population[idx].fitness = fit

def precompute_text_features(
    clip_model,
    dataset,
) -> Tuple[Optional[torch.Tensor]]:
    """
    Precompute text features for the dataset with memory optimization.
    """
    device = next(clip_model.parameters()).device
    print(f"clip_model is on device: {device}")
    template = dataset.template[0]
    texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    
    # 分批处理文本以避免内存溢出
    batch_size = 32  # 减小批次大小
    text_features_list = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            # Tokenize当前批次
            batch_tokens = clip.tokenize(batch_texts).to(device)
            
            # 使用更节省内存的autocast
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                batch_embeddings = clip_model.encode_text(batch_tokens)
                batch_features = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            # 立即移动到CPU并清理
            text_features_list.append(batch_features.cpu())
            
            # 清理当前批次的变量
            del batch_tokens, batch_embeddings, batch_features
            torch.cuda.empty_cache()
    
    # 在CPU上合并所有特征
    text_features = torch.cat(text_features_list, dim=0)
    
    # 最终如果需要返回GPU tensor，可以选择性放回GPU
    # text_features = text_features.to(device)
    
    return text_features

def _precompute_text_and_images(
    args,
    clip_model,
    dataset,
    loader,
) -> Tuple[Optional[torch.Tensor], Optional[list]]:
    """
    仅在 args.encoder == 'text' 时：
      - 预 Tokenize 文本（template -> texts -> tokens）
      - 预计算验证集图像特征（归一化）
    返回: (tokens_cache, image_features_cache)
    """
    if getattr(args, "encoder", None) != "text":
        return None, None

    template = dataset.template[0]
    texts = [template.format(classname.replace("_", " ")) for classname in dataset.classnames]

    # Tokenize on CPU and keep tokens on CPU to avoid holding large GPU tensors
    with torch.no_grad():
        tokens_cache = clip.tokenize(texts)
        # 确保tokens在CPU上，避免设备不匹配
        tokens_cache = tokens_cache.cpu()

    # Precompute image features in small sub-batches to avoid OOM.
    # We will move images to GPU in chunks of size args.eval_batch_size (or loader.batch_size)
    image_features_cache = []
    progress = tqdm(loader, desc="Precompute val image features", leave=False)

    # Determine sub-batch size to use on GPU. Prefer args.eval_batch_size if provided.
    sub_batch = getattr(args, "eval_batch_size", None) or getattr(loader, "batch_size", None) or 16
    if sub_batch <= 0:
        sub_batch = 16

    for images, target in progress:
        # images: tensor of shape [B, C, H, W]
        bs = images.shape[0]
        feats_list = []
        targets_cpu = target.cpu()

        # process in smaller chunks on GPU
        for start in range(0, bs, sub_batch):
            end = min(start + sub_batch, bs)
            img_chunk = images[start:end].cuda(non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    feats_chunk = clip_model.encode_image(img_chunk)
                    feats_chunk = feats_chunk / feats_chunk.norm(dim=-1, keepdim=True)
                # move to CPU immediately
                feats_list.append(feats_chunk.cpu())
            # free GPU memory between chunks
            del img_chunk, feats_chunk
            torch.cuda.empty_cache()

        feats = torch.cat(feats_list, dim=0)
        image_features_cache.append((feats, targets_cpu))

    # Ensure tokens_cache is on CPU (so consumers can .to(device) as needed)
    return tokens_cache, image_features_cache


def run_lora_ga(args, clip_model, dataset, train_loader, val_loader=None, test_loader=None, gpu_id=1):
    """
    入口：对应用了 LoRA 的 clip_model 进行 GA 搜索 LoRA 因子。
    """
   
    set_global_seed(SEED)
    torch.cuda.set_device(gpu_id)

    clip_model = clip_model.cuda()
    # 应用 LoRA 并上 GPU
    list_lora_layers = apply_lora(args, clip_model)
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        clip_model = clip_model.eval()
        acc_test = evaluate_lora(clip_model, test_loader, dataset)
        print(f"[GA] Final Test Acc for shots = {args.shots} is {acc_test:.4f}")
        return

    # prepare results directory and per-generation log
    result_dir = getattr(args, "result_path", None) or os.getcwd()
    os.makedirs(result_dir, exist_ok=True)
    gen_log_path = os.path.join(result_dir, "ga_generations.json")
    generation_log = []

    if args.encoder == "text":
        # 预计算文本/图像特征缓存（仅 text-encoder 模式）
        tokens_cache, image_features_cache = _precompute_text_and_images(
            args, clip_model, dataset, train_loader
        )
        
        tokens_cache = tokens_cache.cuda()
        image_features_cache = [(feat.cuda(), target) for feat, target in image_features_cache]
    elif args.encoder == "vision":
        text_features = precompute_text_features(clip_model, dataset)

    # 初始化种群
    # 确保种群大小为偶数（便于两两交叉），同时不小于精英数
    pop_size = max(2 * ((POPULATION_SIZE + 1) // 2), NUM_ELITES + 2)
    population = init_pop(pop_size=pop_size, list_lora_layers=list_lora_layers)
    # base = population[0]
    # apply_genes_to_layers(base.genes, list_lora_layers)
    # train_acc = evaluate_lora(clip_model, train_loader, dataset, cached_tokens=tokens_cache, cached_image_batches=image_features_cache)
    # val_acc = evaluate_lora(clip_model, val_loader, dataset, cached_tokens=tokens_cache, cached_image_batches=image_features_cache)
    # print(f"base train_acc: {train_acc:.4f}, base val_acc: {val_acc:.4f}\n")
    # 演化主循环
    for gen in range(NUM_GENERATIONS):
        # evaluate population (parallel across GPUs if available)
        update_fitness(
                population,
                clip_model,
                list_lora_layers,
                train_loader,
                dataset,
                cached_text_features=text_features,
                cached_tokens=None,
                cached_image_features=None,
            )
        best_ind = max(
            population,
            key=lambda c: c.fitness if c.fitness is not None else -float("inf"),
        )
        avg_fitness = sum(
            c.fitness if c.fitness is not None else 0.0 for c in population
        ) / len(population)

        # 验证集评估
        if val_loader is not None:
            apply_genes_to_layers(best_ind.genes, list_lora_layers)
            val_acc = evaluate_lora(
                clip_model,
                val_loader,
                dataset,
                cached_tokens=None,
                cached_image_batches=None,
                cached_text_features=text_features,
            )
            save_lora(args, list_lora_layers)
        print(
            f"[GA] Gen {gen:03d} | Best fitness={best_ind.fitness:.4f}, val_acc={val_acc:.4f}"
        )

        generation_log.append({
            "generation": int(gen),
            "best_fitness": best_ind.fitness,
            "avg_fitness": avg_fitness,
            "val_acc": val_acc,
        })
        try:
            with open(gen_log_path, "w", encoding="utf-8") as f:
                json.dump(generation_log, f, indent=2)
        except Exception as e:
            print(f"[GA] Warning: failed to write generation log to {gen_log_path}: {e}")
        # 产生新一代
        population = reproduce(population, num_elites=NUM_ELITES, num_parents=NUM_PARENTS)

    best_ind = max(
            population,
            key=lambda c: c.fitness if c.fitness is not None else -float("inf"),
        )
    apply_genes_to_layers(best_ind.genes, list_lora_layers)
    
    if test_loader is not None:
        print("[GA] Evaluating final best individual on test set...")
        # 最终测试：用最佳个体回写权重
        if test_loader is not None:
            acc_test = evaluate(
                clip_model,
                "ga",
                test_loader,
                dataset,
                args.eval_datasets,
                args.result_path,
                args.seed,
                args.root_path,
            )
            print(f"[GA] Final Test Acc = {acc_test:.4f}")

    if getattr(args, "save_path", None) is not None:
        # save the current LoRA weights (conventional single save)
        print(f"[GA] Saving final LoRA weights to {args.save_path} ...")
        save_lora(args, list_lora_layers)

        # additionally save the top NUM_ELITES chromosomes (all elites)
        try:
            num_elites_to_save = min(NUM_ELITES, len(population))
            if num_elites_to_save > 0:
                # sort population by fitness descending
                elites = sorted(
                    population,
                    key=lambda c: (c.fitness if c.fitness is not None else -float("inf")),
                    reverse=True,
                )[:num_elites_to_save]

                # preserve original filename if present
                orig_filename = getattr(args, 'filename', None)

                for rank, elite in enumerate(elites):
                    try:
                        apply_genes_to_layers(elite.genes, list_lora_layers)

                        # set a filename suffix for this elite save
                        suffix = f"elite{rank:02d}_acc{elite.fitness:.4f}"
                        if orig_filename:
                            args.filename = f"{orig_filename}_{suffix}"
                        else:
                            args.filename = f"lora_{suffix}"

                        # save this elite's LoRA weights
                        save_lora(args, list_lora_layers)
                    except Exception as e:
                        print(f"[GA] Warning: failed to save elite {rank}: {e}")

                # restore original filename
                if orig_filename is not None:
                    args.filename = orig_filename
                else:
                    try:
                        delattr(args, 'filename')
                    except Exception:
                        pass
        except Exception as e:
            print(f"[GA] Warning: failed to save elites: {e}")
    return

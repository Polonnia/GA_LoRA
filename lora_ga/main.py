import os
import sys
import json
import time
from typing import Optional
import torch
import random
import multiprocessing as mp

from .evolution import EvolutionEngine
from .parallel import ParallelEvaluator
from .utils.noise_table import SharedNoiseTable
from .utils.evaluation import evaluate_lora, apply_genes_to_layers, precompute_text_features
from .config import *
from loralib.utils import apply_lora, save_lora, load_lora

def set_global_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_lora_ga_parallel(args, clip_model, dataset, gpu_id=0):
    """
    优化的LoRA GA主函数
    """
    
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    args.gpu_ids = [1,4,5]
    set_global_seed(args.seed)
    torch.cuda.set_device(gpu_id)

    clip_model = clip_model.cuda()
    
    # 应用LoRA
    list_lora_layers = apply_lora(args, clip_model)
    
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        # 从参数中获取GPU ID列表
        if isinstance(args.gpu_ids, str):
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        else:
            gpu_ids = args.gpu_ids
    else:
        # 默认使用所有可用GPU
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = []
    
    print(f"[MAIN] 将使用GPU: {gpu_ids}")

    # 初始化优化组件
    noise_table = SharedNoiseTable(size=NOISE_TABLE_SIZE, seed=NOISE_SEED)
    evolution_engine = EvolutionEngine(
        noise_table=noise_table,
        use_adaptive_mutation=True
    )
    
    # 预计算文本特征，避免在每个worker中重复计算
    if args.encoder == "vision":
        text_features = precompute_text_features(clip_model, dataset)
    else:
        text_features = None
    
    # 初始化并行评估器，直接使用dataset.train_loader
    parallel_evaluator = ParallelEvaluator(
        base_model=clip_model,
        list_lora_layers=list_lora_layers,
        noise_table=noise_table,
        gpu_ids=[1,4,5],
        processes_per_gpu=8,
        dataset=dataset,  # dataset包含train_loader, test_loader, val_loader
        debug=True
    )
    
    stats = parallel_evaluator.get_stats()
    print(f"[MAIN] 并行评估器初始化完成: {stats}")

    # 初始化种群
    pop_size = max(2 * ((POPULATION_SIZE + 1) // 2), NUM_ELITES + 2)
    population = evolution_engine.initialize_population(list_lora_layers, pop_size)
    
    # 结果记录
    result_dir = getattr(args, "result_path", None) or os.getcwd()
    os.makedirs(result_dir, exist_ok=True)
    gen_log_path = os.path.join(result_dir, "ga_optimized_generations.json")
    generation_log = []

    # 进化主循环
    for gen in range(NUM_GENERATIONS):
        start_time = time.time()
        
        # 并行评估种群（dataset已在初始化时传入，这里不需要再传）
        parallel_evaluator.evaluate_population_parallel(
            population, cached_text_features=text_features
        )
        
        # 获取统计信息
        stats = evolution_engine.get_population_stats(population)
        best_ind = evolution_engine.get_best_individual(population)
        
        # 验证集评估
        if dataset.val_loader is not None:
            apply_genes_to_layers(best_ind.genes, list_lora_layers)
            val_acc = evaluate_lora(
                clip_model, dataset.val_loader, dataset, cached_text_features=text_features
            )
            save_lora(args, list_lora_layers)
        else:
            val_acc = 0.0
        
        # 记录日志
        generation_time = time.time() - start_time
        generation_log.append({
            "generation": gen,
            "best_fitness": stats['best_fitness'],
            "avg_fitness": stats['avg_fitness'],
            "std_fitness": stats['std_fitness'],
            "val_acc": val_acc,
            "generation_time": generation_time
        })
        
        print(
            f"[Optimized GA] Gen {gen:03d} | "
            f"Best: {stats['best_fitness']:.4f}, Avg: {stats['avg_fitness']:.4f} | "
            f"Val: {val_acc:.4f} | Time: {generation_time:.1f}s"
        )
        
        # 保存日志
        try:
            with open(gen_log_path, "w", encoding="utf-8") as f:
                json.dump(generation_log, f, indent=2)
        except Exception as e:
            print(f"[GA] Warning: failed to write generation log: {e}")
        
        # 产生新一代
        if gen < NUM_GENERATIONS - 1:  # 最后一代不需要繁殖
            population = evolution_engine.reproduce(population, current_generation=gen)

    # 最终评估和保存
    best_ind = evolution_engine.get_best_individual(population)
    apply_genes_to_layers(best_ind.genes, list_lora_layers)

    # 保存最佳模型
    if getattr(args, "save_path", None) is not None:
        save_lora(args, list_lora_layers)

    # 清理资源
    parallel_evaluator.cleanup()
    
    return best_ind.fitness

def _save_elites(self, args, population, list_lora_layers):
    """保存精英个体"""
    try:
        num_elites_to_save = min(NUM_ELITES, len(population))
        if num_elites_to_save > 0:
            elites = sorted(
                population,
                key=lambda c: (c.fitness if c.fitness is not None else -float("inf")),
                reverse=True,
            )[:num_elites_to_save]

            orig_filename = getattr(args, 'filename', None)

            for rank, elite in enumerate(elites):
                try:
                    apply_genes_to_layers(elite.genes, list_lora_layers)
                    suffix = f"elite{rank:02d}_acc{elite.fitness:.4f}"
                    
                    if orig_filename:
                        args.filename = f"{orig_filename}_{suffix}"
                    else:
                        args.filename = f"lora_{suffix}"

                    save_lora(args, list_lora_layers)
                except Exception as e:
                    print(f"[GA] Warning: failed to save elite {rank}: {e}")

            # 恢复原始文件名
            if orig_filename is not None:
                args.filename = orig_filename
            else:
                try:
                    delattr(args, 'filename')
                except Exception:
                    pass
    except Exception as e:
        print(f"[GA] Warning: failed to save elites: {e}")
        
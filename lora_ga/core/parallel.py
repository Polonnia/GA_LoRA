import torch
import torch.nn as nn
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import math

from .chromosome import Chromosome
from utils.evaluation import evaluate_lora

class ParallelEvaluator:
    """并行评估器，基于参考代码的ConcurrentWorkers思想"""
    def __init__(self, base_model: nn.Module, list_lora_layers: List[nn.Module], 
                 num_workers: Optional[int] = None):
        self.base_model = base_model
        self.list_lora_layers = list_lora_layers
        self.num_workers = num_workers or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        
        # 创建worker模型
        self.worker_models = []
        self.worker_lora_layers = []
        self._initialize_workers()
    
    def _initialize_workers(self):
        """初始化worker模型"""
        for i in range(self.num_workers):
            # 深拷贝模型到不同设备
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{i % torch.cuda.device_count()}')
            else:
                device = torch.device('cpu')
                
            model_copy = deepcopy(self.base_model).to(device)
            lora_layers_copy = self._copy_lora_layers(self.list_lora_layers, device)
            self.worker_models.append(model_copy)
            self.worker_lora_layers.append(lora_layers_copy)
    
    def _copy_lora_layers(self, lora_layers: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """复制LoRA层到指定设备"""
        copied_layers = []
        for layer in lora_layers:
            layer_copy = deepcopy(layer).to(device)
            copied_layers.append(layer_copy)
        return copied_layers
    
    def evaluate_population_parallel(self, population: List[Chromosome], train_loader, dataset, 
                                   cached_text_features: Optional[torch.Tensor] = None) -> None:
        """并行评估种群"""
        num_tasks = len(population)
        tasks_per_worker = math.ceil(num_tasks / self.num_workers)
        
        def evaluate_chunk(worker_idx: int, chunk_indices: List[int]):
            """单个worker评估一批个体"""
            model = self.worker_models[worker_idx]
            lora_layers = self.worker_lora_layers[worker_idx]
            device = next(model.parameters()).device
            
            # 准备缓存特征
            local_text_features = None
            if cached_text_features is not None:
                local_text_features = cached_text_features.to(device)
            
            results = []
            for idx in chunk_indices:
                chrom = population[idx]
                if chrom.need_update:
                    # 应用基因到模型
                    self._apply_genes_to_layers(chrom.genes, lora_layers)
                    # 评估适应度
                    fitness = evaluate_lora(
                        model, train_loader, dataset, 
                        cached_text_features=local_text_features
                    )
                    results.append((idx, float(fitness)))
                else:
                    results.append((idx, chrom.fitness))
            return results
        
        # 分配任务
        chunks = []
        for i in range(self.num_workers):
            start_idx = i * tasks_per_worker
            end_idx = min((i + 1) * tasks_per_worker, num_tasks)
            if start_idx < num_tasks:
                chunks.append((i, list(range(start_idx, end_idx))))
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(evaluate_chunk, worker_idx, indices) 
                      for worker_idx, indices in chunks]
            
            # 收集结果
            for future in as_completed(futures):
                for idx, fitness in future.result():
                    population[idx].fitness = fitness
                    population[idx].need_update = False
    
    def _apply_genes_to_layers(self, genes: List[torch.Tensor], lora_layers: List[nn.Module]):
        """应用基因到LoRA层"""
        idx = 0
        with torch.no_grad():
            for layer in lora_layers:
                # 确定启用的投影顺序
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

                    # 应用LoRA权重
                    a_t = genes[idx].to(mod.w_lora_A.device, dtype=mod.w_lora_A.dtype)
                    b_t = genes[idx + 1].to(mod.w_lora_B.device, dtype=mod.w_lora_B.dtype)
                    mod.w_lora_A.copy_(a_t)
                    mod.w_lora_B.copy_(b_t)
                    idx += 2
    
    def cleanup(self):
        """清理资源"""
        for model in self.worker_models:
            del model
        for layers in self.worker_lora_layers:
            del layers
        torch.cuda.empty_cache()
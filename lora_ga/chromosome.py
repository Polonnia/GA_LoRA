import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Any
from copy import deepcopy
import numpy as np

class Chromosome:
    """
    基于种子链的染色体类 - 简化版本
    只负责存储种子链和适应度，不存储权重矩阵
    """
    def __init__(self, lora_layers: List[torch.nn.Module] = None, seeds: Tuple = None):
        self.fitness: Optional[float] = None
        self.need_update: bool = True
        self.seeds = seeds  # 种子链: (base_seed, (mutation1_idx, mutation1_power), ...)
        self.enabled_lora = []
        self.num_params = 0  # 总参数数量（用于权重重建）
        
        # 只需要计算参数数量，不需要存储基因
        if lora_layers is not None and len(lora_layers) > 0:
            first_layer = lora_layers[0]
            if hasattr(first_layer, "enable_lora"):
                self.enabled_lora = list(first_layer.enable_lora)
            
            # 只计算参数数量，不初始化基因
            self._calculate_num_params(lora_layers)
    
    def _calculate_num_params(self, lora_layers):
        """只计算总参数数量，不存储基因"""
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
                    # 只累加参数数量，不存储实际的权重张量
                    self.num_params += proj_layer.w_lora_A.numel() + proj_layer.w_lora_B.numel()
    
    def clone(self):
        """深拷贝染色体 - 简化版本"""
        new_chrom = Chromosome()
        new_chrom.fitness = self.fitness
        new_chrom.need_update = self.need_update
        new_chrom.enabled_lora = self.enabled_lora[:]
        new_chrom.seeds = deepcopy(self.seeds)
        new_chrom.num_params = self.num_params
        return new_chrom
    
    def get_seed_chain_length(self):
        """获取种子链长度"""
        if self.seeds is None:
            return 0
        return len(self.seeds)
    
    def __repr__(self):
        seeds_repr = f"seeds(len={self.get_seed_chain_length()})" if self.seeds else "No seeds"
        fitness_repr = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Chromosome(fitness={fitness_repr}, {seeds_repr}, params={self.num_params})"
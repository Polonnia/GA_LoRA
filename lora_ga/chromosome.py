import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Any
from copy import deepcopy
import numpy as np
class Chromosome:
    """
    基于种子链的染色体类
    """
    def __init__(self, lora_layers: List[torch.nn.Module] = None, seeds: Tuple = None):
        self.fitness: Optional[float] = None
        self.need_update: bool = True
        self.genes: List[torch.Tensor] = []
        self.seeds = seeds  # 种子链: (base_seed, (mutation1_idx, mutation1_power), ...)
        self.enabled_lora = []
        self.num_params = 0  # 总参数数量
        
        if lora_layers is not None and len(lora_layers) > 0:
            first_layer = lora_layers[0]
            if hasattr(first_layer, "enable_lora"):
                self.enabled_lora = list(first_layer.enable_lora)
            
            # 初始化基因并计算总参数数量
            self._initialize_genes(lora_layers)
    
    def _initialize_genes(self, lora_layers):
        """初始化基因从LoRA层并计算总参数数量"""
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
                    self.num_params += a.numel() + b.numel()
    
    def compute_weights_from_seeds(self, noise_table, cache=None):
        """基于种子链计算权重"""
        if self.seeds is None or len(self.seeds) == 0:
            raise ValueError("No seeds available for weight computation")
        
        # 检查缓存
        if cache and self.seeds in cache:
            return cache[self.seeds]
        
        # 从基础种子开始
        base_seed = self.seeds[0]
        if isinstance(base_seed, tuple):
            # 基础种子本身也是一个突变链
            theta = self._compute_from_seed_chain(noise_table, base_seed)
        else:
            # 单个基础种子
            theta = noise_table.get(base_seed, self.num_params).copy()
        
        # 应用后续突变
        for mutation in self.seeds[1:]:
            if isinstance(mutation, tuple) and len(mutation) == 2:
                idx, power = mutation
                mutation_noise = noise_table.get(idx, self.num_params)
                theta = theta + power * mutation_noise
            else:
                # 处理旧的种子格式
                idx = mutation
                mutation_noise = noise_table.get(idx, self.num_params)
                theta = theta + 0.005 * mutation_noise  # 默认突变强度
        
        return theta
    
    def _compute_from_seed_chain(self, noise_table, seed_chain):
        """递归计算种子链的权重"""
        if isinstance(seed_chain, tuple) and len(seed_chain) > 1:
            # 递归处理嵌套的种子链
            base_theta = self._compute_from_seed_chain(noise_table, seed_chain[0])
            for mutation in seed_chain[1:]:
                if isinstance(mutation, tuple) and len(mutation) == 2:
                    idx, power = mutation
                    mutation_noise = noise_table.get(idx, self.num_params)
                    base_theta = base_theta + power * mutation_noise
            return base_theta
        else:
            # 基础种子
            return noise_table.get(seed_chain, self.num_params).copy()
    
    def apply_weights_to_genes(self, weights_flat: np.ndarray):
        """将扁平化的权重应用到基因中"""
        if len(weights_flat) != self.num_params:
            raise ValueError(f"Weight dimension mismatch: expected {self.num_params}, got {len(weights_flat)}")
        
        offset = 0
        for i, gene in enumerate(self.genes):
            gene_size = gene.numel()
            weight_slice = weights_flat[offset:offset + gene_size]
            gene_weights = torch.from_numpy(weight_slice).reshape(gene.shape).to(gene.dtype)
            self.genes[i] = gene_weights
            offset += gene_size
        
        self.need_update = True
    
    def clone(self):
        """深拷贝染色体"""
        new_chrom = Chromosome()
        new_chrom.fitness = self.fitness
        new_chrom.need_update = self.need_update
        new_chrom.genes = [g.clone() for g in self.genes]
        new_chrom.enabled_lora = self.enabled_lora[:]
        new_chrom.seeds = deepcopy(self.seeds)
        new_chrom.num_params = self.num_params
        return new_chrom
    
    @property
    def num_genes(self):
        return len(self.genes)
    
    @property
    def gene_shapes(self):
        return [g.shape for g in self.genes]
    
    def get_seed_chain_length(self):
        """获取种子链长度"""
        if self.seeds is None:
            return 0
        return len(self.seeds)
    
    def __repr__(self):
        seeds_repr = f"seeds(len={self.get_seed_chain_length()})" if self.seeds else "No seeds"
        fitness_repr = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Chromosome(fitness={fitness_repr}, {seeds_repr}, genes={self.num_genes})"
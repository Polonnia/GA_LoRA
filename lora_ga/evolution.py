import random
import math
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np

from .chromosome import Chromosome
from .utils.noise_table import SharedNoiseTable
from .utils.schedulers import MutationScheduler, AdaptiveMutationScheduler
from .config import *
from copy import deepcopy

class EvolutionEngine:
    """基于种子链的进化算法引擎"""
    
    def __init__(self, 
                 noise_table: SharedNoiseTable,
                 mutation_scheduler: Optional[MutationScheduler] = None,
                 use_adaptive_mutation: bool = False):
        
        self.noise_table = noise_table
        self.weight_cache = {}  # 权重缓存
        
        if use_adaptive_mutation:
            self.mutation_scheduler = AdaptiveMutationScheduler(
                initial_power=STD_DEV,
                min_power=0.001,
                max_power=0.5
            )
        else:
            self.mutation_scheduler = mutation_scheduler or MutationScheduler(
                initial_power=STD_DEV,
                schedule_type=MUTATION_SCHEDULE_TYPE,
                final_power=FINAL_MUTATION_POWER,
                schedule_steps=NUM_GENERATIONS * POPULATION_SIZE
            )
        
        self.use_adaptive_mutation = use_adaptive_mutation
        

    def initialize_population(self, list_lora_layers: List[torch.nn.Module], 
                            pop_size: int = POPULATION_SIZE) -> List[Chromosome]:
        """初始化基于种子链的种群"""
        population = []
        
        # 创建基础个体（使用随机种子）
        base_chrom = Chromosome(list_lora_layers)
        if base_chrom.num_params > 0:
            base_seed = self.noise_table.sample_index(base_chrom.num_params)
            base_chrom.seeds = (base_seed,)
            
            # 计算并设置基础权重
            base_weights = self._compute_weights_from_seeds(base_chrom.seeds, base_chrom.num_params)
            self.weight_cache[base_chrom.seeds] = base_weights
            
        population.append(base_chrom)
        
        # 创建初始变异个体
        for _ in range(pop_size - 1):
            chrom = Chromosome(list_lora_layers)
            if base_chrom.num_params > 0:
                # 对基础个体进行突变
                mutation_idx = self.noise_table.sample_index(base_chrom.num_params)
                chrom.seeds = base_chrom.seeds + ((mutation_idx, STD_DEV),)
                
                # 计算突变后的权重
                mutated_weights = self._compute_weights_from_seeds(chrom.seeds, chrom.num_params)
                self.weight_cache[chrom.seeds] = mutated_weights
                
            population.append(chrom)
        
        return population
    
    def mutate(self, chromosome: Chromosome, mutation_power: float, 
               mutation_rate: float = MUTATION_RATE) -> Chromosome:
        """基于种子链的突变操作"""
        if random.random() >= mutation_rate:
            return chromosome.clone()
        
        # 创建突变个体
        mutated = chromosome.clone()
        mutation_idx = self.noise_table.sample_index(chromosome.num_params)
        
        # 扩展种子链
        if mutated.seeds:
            mutated.seeds = mutated.seeds + ((mutation_idx, mutation_power),)
        else:
            mutated.seeds = (mutation_idx, mutation_power)
            
        mutated.need_update = True
        mutated.fitness = None
        return mutated
    
    
    def reproduce(self, population: List[Chromosome], 
                           num_elites: int = NUM_ELITES,
                           num_parents: int = NUM_PARENTS,
                           current_generation: int = 0) -> List[Chromosome]:
        """基于种子链的繁殖策略（类似参考代码）"""
        pop_size = len(population)
        
        # 排序种群
        ranked = sorted(
            population,
            key=lambda c: (c.fitness if c.fitness is not None else -float("inf")),
            reverse=True,
        )
        
        # 获取当前突变强度
        if self.use_adaptive_mutation:
            best_fitness = ranked[0].fitness if ranked[0].fitness is not None else 0.0
            mutation_power = self.mutation_scheduler.update(best_fitness)
        else:
            mutation_power = self.mutation_scheduler.get_power(
                current_generation, current_generation * pop_size
            )
        
        # 精英保留
        new_population = []
        keep = min(num_elites, pop_size)
        for i in range(keep):
            ranked[i].need_update = False
            new_population.append(ranked[i])
        
        # 选择父母池（精英个体）
        mating_pool = ranked[:num_parents]
        
        # 产生后代：只通过突变，没有交叉
        while len(new_population) < pop_size:
            parent = random.choice(mating_pool)
            child = self.mutate(parent, mutation_power)
            new_population.append(child)
        
        return new_population
    
    def tournament_selection(self, population: List[Chromosome], 
                           tournament_size: int = 3) -> Chromosome:
        """锦标赛选择"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda c: c.fitness if c.fitness is not None else -float("inf"))
    
    def get_best_individual(self, population: List[Chromosome]) -> Chromosome:
        """获取最佳个体"""
        return max(
            population,
            key=lambda c: c.fitness if c.fitness is not None else -float("inf"),
        )
    
    def get_population_stats(self, population: List[Chromosome]) -> dict:
        """获取种群统计信息"""
        fitnesses = [c.fitness for c in population if c.fitness is not None]
        seed_lengths = [c.get_seed_chain_length() for c in population if c.seeds is not None]
        
        if not fitnesses:
            return {
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'std_fitness': 0.0,
                'min_fitness': 0.0,
                'avg_seed_length': 0.0,
                'max_seed_length': 0
            }
        
        return {
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'std_fitness': torch.std(torch.tensor(fitnesses)).item() if len(fitnesses) > 1 else 0.0,
            'min_fitness': min(fitnesses),
            'avg_seed_length': sum(seed_lengths) / len(seed_lengths) if seed_lengths else 0.0,
            'max_seed_length': max(seed_lengths) if seed_lengths else 0
        }
    
    def clear_cache(self):
        """清理权重缓存"""
        self.weight_cache.clear()
        
    
    def _compute_weights_from_seeds(self, seeds: Tuple, num_params: int) -> np.ndarray:
        """基于种子链计算权重"""
        if seeds is None or len(seeds) == 0:
            raise ValueError("No seeds available for weight computation")
        
        # 检查缓存
        if seeds in self.weight_cache:
            return self.weight_cache[seeds]
        
        # 从基础种子开始
        base_seed = seeds[0]
        if isinstance(base_seed, tuple):
            # 基础种子本身也是一个突变链
            theta = self._compute_from_seed_chain(base_seed, num_params)
        else:
            # 单个基础种子
            theta = self.noise_table.get(base_seed, num_params).copy()
        
        # 应用后续突变
        for mutation in seeds[1:]:
            if isinstance(mutation, tuple) and len(mutation) == 2:
                idx, power = mutation
                mutation_noise = self.noise_table.get(idx, num_params)
                theta = theta + power * mutation_noise
            else:
                # 处理旧的种子格式
                idx = mutation
                mutation_noise = self.noise_table.get(idx, num_params)
                theta = theta + 0.005 * mutation_noise
        
        # 缓存结果
        self.weight_cache[seeds] = theta
        return theta

    def _compute_from_seed_chain(self, seed_chain, num_params: int) -> np.ndarray:
        """递归计算种子链的权重"""
        if isinstance(seed_chain, tuple) and len(seed_chain) > 1:
            # 递归处理嵌套的种子链
            base_theta = self._compute_from_seed_chain(seed_chain[0], num_params)
            for mutation in seed_chain[1:]:
                if isinstance(mutation, tuple) and len(mutation) == 2:
                    idx, power = mutation
                    mutation_noise = self.noise_table.get(idx, num_params)
                    base_theta = base_theta + power * mutation_noise
            return base_theta
        else:
            # 基础种子
            return self.noise_table.get(seed_chain, num_params).copy()
        
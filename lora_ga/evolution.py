import random
import math
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np

from .chromosome import Chromosome
from .utils.noise_table import SharedNoiseTable
from .utils.schedulers import MutationScheduler, AdaptiveMutationScheduler
from .config import *

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
            base_weights = base_chrom.compute_weights_from_seeds(self.noise_table)
            base_chrom.apply_weights_to_genes(base_weights)
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
                mutated_weights = chrom.compute_weights_from_seeds(self.noise_table, self.weight_cache)
                chrom.apply_weights_to_genes(mutated_weights)
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
        
        # 计算突变后的权重
        try:
            mutated_weights = mutated.compute_weights_from_seeds(self.noise_table, self.weight_cache)
            mutated.apply_weights_to_genes(mutated_weights)
            self.weight_cache[mutated.seeds] = mutated_weights
        except Exception as e:
            print(f"Warning: Mutation failed: {e}")
            # 回退到传统突变
            mutated = self._fallback_mutate(chromosome, mutation_power)
            
        mutated.need_update = True
        mutated.fitness = None
        return mutated
    
    def _fallback_mutate(self, chromosome: Chromosome, mutation_power: float) -> Chromosome:
        """传统的突变方法（备用）"""
        mutated = chromosome.clone()
        for i, gene in enumerate(mutated.genes):
            if random.random() < MUTATION_RATIO:
                noise = torch.randn_like(gene) * mutation_power
                mutated.genes[i] += noise
        
        # 生成新的种子链
        mutation_idx = self.noise_table.sample_index(chromosome.num_params)
        if chromosome.seeds:
            mutated.seeds = chromosome.seeds + ((mutation_idx, mutation_power),)
        else:
            mutated.seeds = (mutation_idx, mutation_power)
            
        mutated.need_update = True
        return mutated
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """基于种子链的交叉操作 - 选择父母之一的种子链"""
        child = Chromosome()
        child.enabled_lora = parent1.enabled_lora[:]
        
        # 随机选择父母的种子链
        if random.random() < 0.5:
            child.seeds = deepcopy(parent1.seeds)
        else:
            child.seeds = deepcopy(parent2.seeds)
        
        # 重新计算权重
        if child.seeds and child.num_params > 0:
            try:
                child_weights = child.compute_weights_from_seeds(self.noise_table, self.weight_cache)
                # 需要先初始化基因
                child.genes = [torch.zeros_like(g) for g in parent1.genes]
                child.apply_weights_to_genes(child_weights)
                self.weight_cache[child.seeds] = child_weights
            except Exception as e:
                print(f"Warning: Crossover weight computation failed: {e}")
                # 回退到传统交叉
                child = self._fallback_crossover(parent1, parent2)
        else:
            child = self._fallback_crossover(parent1, parent2)
            
        child.need_update = True
        return child
    
    def _fallback_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """传统的交叉方法（备用）"""
        child = Chromosome()
        child.enabled_lora = parent1.enabled_lora[:]
        child.genes = []
        
        # 块状交叉
        num_layers = len(parent1.genes) // (2 * len(parent1.enabled_lora))
        
        for layer_idx in range(num_layers):
            if random.random() < 0.5:
                source_parent = parent1
            else:
                source_parent = parent2
            
            start_idx = layer_idx * 2 * len(parent1.enabled_lora)
            end_idx = start_idx + 2 * len(parent1.enabled_lora)
            
            for i in range(start_idx, end_idx):
                child.genes.append(source_parent.genes[i].clone())
        
        # 随机选择父母的种子链
        if random.random() < 0.5 and parent1.seeds:
            child.seeds = deepcopy(parent1.seeds)
        elif parent2.seeds:
            child.seeds = deepcopy(parent2.seeds)
            
        child.need_update = True
        return child
    
    def reproduce_seed_based(self, population: List[Chromosome], 
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
    
    def reproduce_hybrid(self, population: List[Chromosome], 
                        num_elites: int = NUM_ELITES,
                        num_parents: int = NUM_PARENTS,
                        current_generation: int = 0) -> List[Chromosome]:
        """混合繁殖策略"""
        pop_size = len(population)
        ranked = sorted(
            population,
            key=lambda c: (c.fitness if c.fitness is not None else -float("inf")),
            reverse=True,
        )
        
        # 获取突变强度
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
        
        # 父母池
        mating_pool = ranked[:num_parents]
        
        # 自适应繁殖策略
        exploration_ratio = max(0.2, 1.0 - current_generation / NUM_GENERATIONS)
        
        while len(new_population) < pop_size:
            if random.random() < exploration_ratio:
                # 探索：基于种子链的突变
                parent = random.choice(mating_pool)
                child = self.mutate(parent, mutation_power)
            else:
                # 利用：交叉
                p = random.sample(mating_pool, 1)
                # 对交叉结果进行轻微突变
                child = self.mutate(p, mutation_power * 0.1, mutation_rate=0.5)
            
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
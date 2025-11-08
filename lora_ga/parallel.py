import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import math
import time
import psutil
import os
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import numpy as np

from .chromosome import Chromosome
from .utils.evaluation import evaluate_lora
from .utils.noise_table import SharedNoiseTable


class GPUWorkerProcess:
    """
    GPU Worker using process-level parallelism for parallel evaluation.
    Each process has its own independent model and LoRA layers copy.
    """
    
    def __init__(self, worker_id: int, gpu_id: int, base_model: nn.Module, 
                 lora_layers: List[nn.Module], noise_table: SharedNoiseTable,
                 debug: bool = True):
        """
        Initialize GPU Worker Process.
        
        Args:
            worker_id: Unique identifier for this worker
            gpu_id: GPU device ID to use
            base_model: Base model architecture to copy
            lora_layers: Original LoRA layers to copy
            noise_table: Shared noise table for seed-based weight reconstruction
            debug: Enable debug logging
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.noise_table = noise_table
        self.debug = debug
        
        # Set device for this worker
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Create independent model and LoRA copies for this process
        self.model = deepcopy(base_model).to(self.device)
        self.lora_layers = self._copy_lora_layers(lora_layers, self.device)
        
        # Process-local weight cache for seed-based reconstruction
        self.weight_cache = {}
        
        self._log_debug(f"GPUWorkerProcess {worker_id} initialized on {self.device}")
    
    def _copy_lora_layers(self, lora_layers: List[nn.Module], device: torch.device) -> List[nn.Module]:
        """Create a deep copy of LoRA layers for process isolation."""
        copied_layers = []
        for layer in lora_layers:
            layer_copy = deepcopy(layer).to(device)
            copied_layers.append(layer_copy)
        return copied_layers
    
    def _log_debug(self, message: str):
        """Process-safe debug logging."""
        if self.debug:
            timestamp = time.strftime("%H:%M:%S")
            process_name = mp.current_process().name
            print(f"[DEBUG GPUWorkerProcess{self.worker_id} {timestamp} {process_name}] {message}")
    
    def evaluate_single(self, chromosome: Chromosome, dataset, 
                   cached_text_features: Optional[torch.Tensor] = None,
                   data_loader_kwargs: Optional[Dict] = None) -> float:
        """
        Evaluate a single chromosome using seed-based weight reconstruction.
        """
        self._log_debug(f"Evaluating chromosome with {len(chromosome.seeds) if chromosome.seeds else 0} seeds")
        eval_start_time = time.time()
        
        try:
            # 🎯 SEED-BASED: Reconstruct weights from seeds and apply to LoRA layers
            apply_start = time.time()
            self._apply_seeds_to_layers(chromosome)
            apply_time = time.time() - apply_start
            
            # 🎯 修复：创建数据加载器
            from torch.utils.data import DataLoader
            local_data_loader_kwargs = data_loader_kwargs or {}
            
            # 确保每个进程有不同的随机种子
            import random
            random_seed = random.randint(0, 1000000)
            
            # 创建独立的数据加载器
            train_loader = DataLoader(
                dataset,
                batch_size=local_data_loader_kwargs.get('batch_size', 32),
                shuffle=local_data_loader_kwargs.get('shuffle', True),
                num_workers=local_data_loader_kwargs.get('num_workers', 0),  # 在子进程中设为0
                pin_memory=local_data_loader_kwargs.get('pin_memory', True)
            )
            
            # 设置不同的随机状态
            if hasattr(train_loader.sampler, 'generator'):
                train_loader.sampler.generator = torch.Generator().manual_seed(random_seed)
            
            self._log_debug(f"Created independent data loader with seed {random_seed}")
            
            # Prepare features if provided
            local_text_features = None
            if cached_text_features is not None:
                local_text_features = cached_text_features.to(self.device)
            
            # Evaluate fitness using process-specific model
            eval_start = time.time()
            
            # 🎯 修复：正确调用 evaluate_lora，传递数据加载器
            fitness = evaluate_lora(
                self.model, 
                train_loader,  # 传递数据加载器，不是数据集
                dataset,       # 传递数据集用于模板和类别名称
                cached_text_features=local_text_features
            )
            
            eval_time = time.time() - eval_start
            
            total_time = time.time() - eval_start_time
            self._log_debug(f"Chromosome evaluation completed - "
                        f"Fitness: {fitness:.4f}, "
                        f"SeedReconstruct: {apply_time:.2f}s, "
                        f"Eval: {eval_time:.2f}s, "
                        f"Total: {total_time:.2f}s")
            
            return float(fitness)
        
        except Exception as e:
            self._log_debug(f"Chromosome evaluation failed: {e}")
            import traceback
            self._log_debug(f"Traceback: {traceback.format_exc()}")
            return -1.0
                
    def _apply_seeds_to_layers(self, chromosome: Chromosome):
        """Apply chromosome seeds to LoRA layers by reconstructing weights from seeds."""
        if not chromosome.seeds:
            raise ValueError("Chromosome has no seeds for weight reconstruction")
        
        # 🎯 SEED-BASED: Reconstruct weights from seed chain
        weights_flat = self._compute_weights_from_seeds(chromosome)
        
        # Apply reconstructed weights to LoRA layers
        self._apply_flat_weights_to_layers(weights_flat)
    
    def _compute_weights_from_seeds(self, chromosome: Chromosome) -> np.ndarray:
        """Compute weights from chromosome's seed chain using shared noise table."""
        # Check local cache first
        cache_key = chromosome.seeds
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key].copy()
        
        if chromosome.seeds is None or len(chromosome.seeds) == 0:
            raise ValueError("No seeds available for weight computation")
        
        # Start with base seed
        base_seed = chromosome.seeds[0]
        if isinstance(base_seed, tuple):
            # Base seed is itself a mutation chain
            theta = self._compute_from_seed_chain(base_seed, chromosome.num_params)
        else:
            # Single base seed
            theta = self.noise_table.get(base_seed, chromosome.num_params).copy()
        
        # Apply subsequent mutations
        for mutation in chromosome.seeds[1:]:
            if isinstance(mutation, tuple) and len(mutation) == 2:
                idx, power = mutation
                mutation_noise = self.noise_table.get(idx, chromosome.num_params)
                theta = theta + power * mutation_noise
            else:
                # Handle old seed format
                idx = mutation
                mutation_noise = self.noise_table.get(idx, chromosome.num_params)
                theta = theta + 0.005 * mutation_noise
        
        # Cache the result
        self.weight_cache[cache_key] = theta.copy()
        
        return theta
    
    def _compute_from_seed_chain(self, seed_chain, num_params: int) -> np.ndarray:
        """Recursively compute weights from nested seed chain."""
        if isinstance(seed_chain, tuple) and len(seed_chain) > 1:
            # Recursively process nested seed chain
            base_theta = self._compute_from_seed_chain(seed_chain[0], num_params)
            for mutation in seed_chain[1:]:
                if isinstance(mutation, tuple) and len(mutation) == 2:
                    idx, power = mutation
                    mutation_noise = self.noise_table.get(idx, num_params)
                    base_theta = base_theta + power * mutation_noise
            return base_theta
        else:
            # Base seed
            return self.noise_table.get(seed_chain, num_params).copy()
    
    def _apply_flat_weights_to_layers(self, weights_flat: np.ndarray):
        """Apply flat weights array to LoRA layers."""
        gene_index = 0
        applied_pairs = 0
        
        with torch.no_grad():
            for layer in self.lora_layers:
                # Get enabled LoRA projections for this layer
                enabled_projections = []
                if hasattr(layer, "q_proj") and hasattr(layer.q_proj, "enable_lora"):
                    enabled_projections = list(layer.q_proj.enable_lora)
                
                for projection in enabled_projections:
                    # Get the specific projection module
                    if projection in ("q", "k", "v"):
                        projection_module = getattr(layer, f"{projection}_proj", None)
                    elif projection in ("o", "out"):
                        projection_module = getattr(layer, "proj", None)
                    else:
                        projection_module = None
                    
                    # Skip if projection doesn't exist or doesn't have LoRA weights
                    if (projection_module is None or 
                        not hasattr(projection_module, "w_lora_A") or 
                        not hasattr(projection_module, "w_lora_B")):
                        continue

                    # Check weight availability
                    a_size = projection_module.w_lora_A.numel()
                    b_size = projection_module.w_lora_B.numel()
                    
                    if gene_index + a_size + b_size > len(weights_flat):
                        raise RuntimeError(
                            f"Weight dimension mismatch: need {a_size + b_size} weights, "
                            f"but only {len(weights_flat) - gene_index} available"
                        )

                    # Extract and reshape weights
                    a_flat = weights_flat[gene_index:gene_index + a_size]
                    b_flat = weights_flat[gene_index + a_size:gene_index + a_size + b_size]
                    
                    # Convert to tensor and move to device
                    weight_a = torch.from_numpy(a_flat).reshape(projection_module.w_lora_A.shape).to(
                        self.device, dtype=projection_module.w_lora_A.dtype
                    )
                    weight_b = torch.from_numpy(b_flat).reshape(projection_module.w_lora_B.shape).to(
                        self.device, dtype=projection_module.w_lora_B.dtype
                    )
                    
                    # Apply weights
                    projection_module.w_lora_A.copy_(weight_a)
                    projection_module.w_lora_B.copy_(weight_b)
                    
                    gene_index += a_size + b_size
                    applied_pairs += 1

    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for this worker."""
        return {
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "cache_size": len(self.weight_cache),
            "device": str(self.device)
        }
    
    def cleanup(self):
        """Clean up GPU worker process resources."""
        self._log_debug("Cleaning up GPU worker process resources...")
        
        # Clear components
        del self.model
        del self.lora_layers
        
        # Clear weight cache
        self.weight_cache.clear()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._log_debug("GPU worker process cleanup completed")


# Worker function for process pool
def worker_initializer(gpu_id, base_model_state, lora_layers_state, noise_table, worker_id, debug):
    """Initialize worker process with model and LoRA layers."""
    try:
        # 🎯 安全地设置CUDA设备
        if torch.cuda.is_available():
            # 先初始化CUDA（如果还没初始化）
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            # 设置设备
            torch.cuda.set_device(gpu_id)
        
        # Reconstruct model and LoRA layers from state
        base_model = deepcopy(base_model_state)
        if torch.cuda.is_available():
            base_model = base_model.cuda(gpu_id)
            
        lora_layers = []
        for layer_state in lora_layers_state:
            layer_copy = deepcopy(layer_state)
            if torch.cuda.is_available():
                layer_copy = layer_copy.cuda(gpu_id)
            lora_layers.append(layer_copy)
        
        # Create worker instance
        worker = GPUWorkerProcess(
            worker_id=worker_id,
            gpu_id=gpu_id,
            base_model=base_model,
            lora_layers=lora_layers,
            noise_table=noise_table,
            debug=debug
        )
        
        # Store worker in process-local storage
        mp.current_process()._worker = worker
        
        if debug:
            print(f"[WORKER] Worker {worker_id} initialized on GPU{gpu_id}")
            
        return worker
        
    except Exception as e:
        print(f"[WORKER] Error initializing worker {worker_id} on GPU{gpu_id}: {e}")
        import traceback
        print(f"[WORKER] Traceback: {traceback.format_exc()}")
        raise

def evaluate_chromosome_worker(args):
    """Worker function for evaluating a single chromosome."""
    chromosome_data, dataset, data_loader_kwargs, cached_text_features = args
    worker = getattr(mp.current_process(), '_worker', None)
    
    if worker is None:
        raise RuntimeError("Worker not initialized in process")
    
    return worker.evaluate_single(
        chromosome_data, dataset, cached_text_features, data_loader_kwargs
    )


class ParallelEvaluator:
    """
    Process-based parallel evaluator with true parallelism:
    - Level 1: GPU-level parallelism (multiple GPUs)
    - Level 2: Process-level parallelism (multiple processes per GPU)
    - Seed-based: Transfer only seeds, reconstruct weights on each device
    """
    
    def __init__(self, base_model: nn.Module, list_lora_layers: List[nn.Module], 
                 noise_table: SharedNoiseTable,
                 gpu_ids: List[int] = None, 
                 processes_per_gpu: int = 2,
                 debug: bool = True):
        """
        Initialize process-based parallel evaluator.
        
        Args:
            base_model: Base model architecture
            list_lora_layers: LoRA layers to parallelize
            noise_table: Shared noise table for seed-based weight reconstruction
            gpu_ids: List of GPU device IDs to use
            processes_per_gpu: Number of processes per GPU for true parallel evaluation
            debug: Enable debug logging
        """
        self.base_model = base_model
        self.list_lora_layers = list_lora_layers
        self.noise_table = noise_table
        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        self.processes_per_gpu = processes_per_gpu
        self.debug = debug
        self.evaluation_count = 0
        
        self._validate_gpu_ids()
        self.process_pools = {}
        self._initialize_process_pools()
        
        self._log_debug(f"Process-based parallel evaluator initialized - "
                      f"GPUs: {self.gpu_ids}, "
                      f"Processes per GPU: {processes_per_gpu}, "
                      f"Total parallelism: {self.get_total_parallelism()}")
    
    def _validate_gpu_ids(self):
        """Validate that specified GPU IDs are available."""
        if not torch.cuda.is_available():
            self.gpu_ids = []
            self._log_debug("CUDA not available, will use CPU mode")
            return
        
        available_gpus = list(range(torch.cuda.device_count()))
        invalid_gpus = [gpu_id for gpu_id in self.gpu_ids if gpu_id not in available_gpus]
        
        if invalid_gpus:
            raise ValueError(f"Specified GPU IDs {invalid_gpus} are not available. "
                           f"Available GPUs: {available_gpus}")
        
        self._log_debug(f"Validated GPU IDs: {self.gpu_ids}")
    
    def _initialize_process_pools(self):
        """Initialize process pools for each GPU."""
        self._log_debug(f"Initializing process pools for {len(self.gpu_ids)} GPUs...")
        start_time = time.time()
        
        for gpu_id in self.gpu_ids:
            try:
                # Create process pool for this GPU
                pool = ProcessPoolExecutor(
                    max_workers=self.processes_per_gpu,
                    initializer=worker_initializer,
                    initargs=(
                        gpu_id,
                        self.base_model,
                        self.list_lora_layers,
                        self.noise_table,
                        gpu_id,  # worker_id
                        self.debug
                    )
                )
                self.process_pools[gpu_id] = pool
                self._log_debug(f"Process pool for GPU{gpu_id} initialized with {self.processes_per_gpu} processes")
                
            except Exception as e:
                self._log_debug(f"Failed to initialize process pool for GPU{gpu_id}: {e}")
                raise
        
        total_time = time.time() - start_time
        self._log_debug(f"All {len(self.process_pools)} process pools initialized in {total_time:.2f}s")
    
    def _log_debug(self, message: str):
        """Debug logging for the parallel evaluator."""
        if self.debug:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[DEBUG ParallelEvaluator {timestamp}] {message}")
    
    def evaluate_population_parallel(self, population: List[Chromosome], dataset, 
                                   data_loader_kwargs: Optional[Dict] = None,
                                   cached_text_features: Optional[torch.Tensor] = None) -> None:
        """
        Evaluate population using process-based parallelism with seed-based weight transfer.
        
        Args:
            population: List of chromosomes to evaluate (only seeds are transferred)
            dataset: Dataset for evaluation (not data loader)
            data_loader_kwargs: Arguments to create data loaders in worker processes
            cached_text_features: Pre-computed text features for efficiency
        """
        self.evaluation_count += 1
        self._log_debug(f"Starting process-based parallel evaluation #{self.evaluation_count} - "
                      f"Population size: {len(population)}")
        
        total_start_time = time.time()
        
        # Count chromosomes that need evaluation
        need_update_count = sum(1 for chrom in population if chrom.need_update)
        self._log_debug(f"Chromosomes needing update: {need_update_count}/{len(population)}")
        
        # Validate that all chromosomes have seeds
        chromosomes_without_seeds = [i for i, chrom in enumerate(population) if not chrom.seeds]
        if chromosomes_without_seeds:
            self._log_debug(f"Warning: {len(chromosomes_without_seeds)} chromosomes without seeds")
        
        # Fallback to sequential evaluation if no process pools available
        if not self.process_pools:
            self._log_debug("No process pools available, using sequential evaluation")
            self._evaluate_sequential(population, dataset, data_loader_kwargs, cached_text_features)
            return
        
        # Distribute chromosomes across GPUs (round-robin)
        tasks = self._distribute_tasks(population, dataset, data_loader_kwargs, cached_text_features)
        
        self._log_debug(f"Distributed {len(tasks)} tasks across {len(self.process_pools)} GPUs")
        
        # Submit all tasks to process pools
        self._log_debug("Starting process-based parallel evaluation...")
        parallel_start = time.time()
        
        futures = {}
        completed_count = 0
        failed_count = 0
        
        # Submit tasks to appropriate GPU process pools
        for task_idx, (gpu_id, task) in enumerate(tasks):
            pool = self.process_pools[gpu_id]
            future = pool.submit(evaluate_chromosome_worker, task)
            futures[future] = task_idx
        
        # Collect results as they complete
        for future in as_completed(futures):
            task_idx = futures[future]
            try:
                fitness = future.result()
                population[task_idx].fitness = fitness
                population[task_idx].need_update = False
                completed_count += 1
                
                if completed_count % 10 == 0:  # Progress reporting
                    progress = (completed_count / len(population)) * 100
                    self._log_debug(f"Progress: {completed_count}/{len(population)} ({progress:.1f}%)")
                    
            except Exception as e:
                failed_count += 1
                self._log_debug(f"Task {task_idx} evaluation failed: {e}")
                population[task_idx].fitness = -1.0
                population[task_idx].need_update = True
        
        parallel_time = time.time() - parallel_start
        total_time = time.time() - total_start_time
        
        # Calculate evaluation statistics
        self._log_evaluation_stats(population, completed_count, failed_count, parallel_time, total_time)
    
    def _distribute_tasks(self, population: List[Chromosome], dataset, 
                         data_loader_kwargs: Optional[Dict] = None,
                         cached_text_features: Optional[torch.Tensor] = None) -> List[Tuple[int, Tuple]]:
        """
        Distribute chromosomes across GPUs in round-robin fashion.
        
        Returns:
            List of (gpu_id, task_data) tuples
        """
        tasks = []
        gpu_ids = list(self.process_pools.keys())
        
        for idx, chrom in enumerate(population):
            if not chrom.need_update:
                continue
                
            # Round-robin distribution across GPUs
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            
            task_data = (
                chrom,  # chromosome_data
                dataset,
                data_loader_kwargs or {},  # 传递数据加载器参数
                cached_text_features
            )
            
            tasks.append((gpu_id, task_data))
        
        return tasks
    
    def _log_evaluation_stats(self, population: List[Chromosome], completed_count: int, 
                            failed_count: int, parallel_time: float, total_time: float):
        """Log evaluation statistics and results."""
        valid_fitness = [c.fitness for c in population if c.fitness is not None and c.fitness >= 0]
        
        if valid_fitness:
            avg_fitness = sum(valid_fitness) / len(valid_fitness)
            max_fitness = max(valid_fitness)
            min_fitness = min(valid_fitness)
            success_rate = (len(valid_fitness) / len(population)) * 100
            
            # 🎯 添加多样性统计
            unique_fitness = len(set(round(f, 4) for f in valid_fitness))
            diversity_rate = (unique_fitness / len(valid_fitness)) * 100
        else:
            avg_fitness = max_fitness = min_fitness = -1
            success_rate = 0
            diversity_rate = 0
        
        self._log_debug(f"Process-based parallel evaluation completed - "
                      f"Success: {completed_count}/{len(population)}, "
                      f"Failed: {failed_count}, "
                      f"Parallel time: {parallel_time:.2f}s, "
                      f"Total time: {total_time:.2f}s, "
                      f"Fitness range: [{min_fitness:.4f}, {max_fitness:.4f}], "
                      f"Average: {avg_fitness:.4f}, "
                      f"Success rate: {success_rate:.1f}%, "
                      f"Diversity: {diversity_rate:.1f}%")
    
    def _evaluate_sequential(self, population: List[Chromosome], dataset, 
                           data_loader_kwargs: Optional[Dict] = None,
                           cached_text_features: Optional[torch.Tensor] = None):
        """Fallback sequential evaluation when no process pools are available."""
        self._log_debug("Using sequential evaluation fallback")
        
        # Create a temporary worker on CPU or first available GPU
        if torch.cuda.is_available() and self.gpu_ids:
            gpu_id = self.gpu_ids[0]
        else:
            gpu_id = 0
            
        worker = GPUWorkerProcess(
            worker_id=0,
            gpu_id=gpu_id,
            base_model=self.base_model,
            lora_layers=self.list_lora_layers,
            noise_table=self.noise_table,
            debug=self.debug
        )
        
        for i, chrom in enumerate(population):
            if chrom.need_update:
                try:
                    fitness = worker.evaluate_single(
                        chrom, dataset, cached_text_features, data_loader_kwargs
                    )
                    chrom.fitness = fitness
                    chrom.need_update = False
                    self._log_debug(f"Sequential evaluation chromosome {i}: fitness {fitness:.4f}")
                except Exception as e:
                    self._log_debug(f"Sequential evaluation chromosome {i} failed: {e}")
                    chrom.fitness = -1.0
        
        worker.cleanup()
    
    def get_total_parallelism(self) -> int:
        """
        Calculate total parallelism degree.
        
        Returns:
            Total number of parallel evaluation slots (GPUs × processes per GPU)
        """
        return len(self.process_pools) * self.processes_per_gpu
    
    def get_stats(self) -> Dict:
        """
        Get evaluator statistics.
        
        Returns:
            Dictionary containing evaluator configuration and stats
        """
        return {
            "evaluation_count": self.evaluation_count,
            "total_gpus": len(self.process_pools),
            "processes_per_gpu": self.processes_per_gpu,
            "total_parallelism": self.get_total_parallelism(),
            "gpu_ids": self.gpu_ids
        }
    
    def cleanup(self):
        """Clean up all resources."""
        self._log_debug("Cleaning up process-based parallel evaluator...")
        start_time = time.time()
        
        # Shutdown all process pools
        for gpu_id, pool in self.process_pools.items():
            try:
                pool.shutdown(wait=True)
                self._log_debug(f"Process pool for GPU{gpu_id} shutdown")
            except Exception as e:
                self._log_debug(f"Process pool for GPU{gpu_id} shutdown failed: {e}")
        
        self.process_pools.clear()
        
        # Clear GPU caches
        if torch.cuda.is_available():
            for gpu_id in self.gpu_ids:
                try:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    self._log_debug(f"GPU{gpu_id} cache cleared")
                except Exception as e:
                    self._log_debug(f"GPU{gpu_id} cache clearance failed: {e}")
        
        cleanup_time = time.time() - start_time
        self._log_debug(f"Process-based cleanup completed in {cleanup_time:.2f}s")
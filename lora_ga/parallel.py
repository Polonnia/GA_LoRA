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
                 dataset=None, classnames=None, template=None, debug: bool = True):
        """
        Initialize GPU Worker Process.
        
        Args:
            worker_id: Unique identifier for this worker
            gpu_id: GPU device ID to use
            base_model: Base model architecture to copy
            lora_layers: Original LoRA layers to copy
            noise_table: Shared noise table for seed-based weight reconstruction
            dataset: Dataset object with train_loader (optional, used to create train_loader in worker)
            classnames: List of class names for evaluation
            template: Template string for text prompts
            debug: Enable debug logging
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.noise_table = noise_table
        self.debug = debug
        self.classnames = classnames
        self.template = template
        
        # Set device for this worker
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Create independent model and LoRA copies for this process
        self.model = deepcopy(base_model).to(self.device)
        self.lora_layers = self._copy_lora_layers(lora_layers, self.device)
        
        # Process-local weight cache for seed-based reconstruction
        self.weight_cache = {}
        
        # 🚀 性能优化：在worker初始化时创建并缓存train_loader
        self.train_loader = None
        if dataset is not None:
            if hasattr(dataset, 'train_loader') and dataset.train_loader is not None:
                # 直接使用dataset的train_loader（如果可访问）
                self.train_loader = dataset.train_loader
                self._log_debug(f"Using dataset.train_loader directly")
            elif hasattr(dataset, 'train_dataset') and dataset.train_dataset is not None:
                # 从train_dataset创建新的DataLoader
                from torch.utils.data import DataLoader
                self.train_loader = DataLoader(
                    dataset.train_dataset,
                    batch_size=getattr(dataset.train_loader, 'batch_size', 32) if hasattr(dataset, 'train_loader') else 32,
                    shuffle=False,  # 评估时不需要shuffle
                    num_workers=0,  # 必须在子进程中为0
                    pin_memory=True
                )
                self._log_debug(f"Created train_loader from train_dataset")
        
        if self.train_loader is not None:
            self._log_debug(f"Train loader cached in worker {worker_id}")
        else:
            self._log_debug(f"Warning: Train loader not available in worker {worker_id}")
        
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
    
    
    def evaluate_single(self, chromosome: Chromosome,
                   cached_text_features: Optional[torch.Tensor] = None) -> float:
        """
        Evaluate a single chromosome using seed-based weight reconstruction.
        
        🚀 性能优化：使用worker初始化时缓存的train_loader，无需传递
        
        Args:
            chromosome: Chromosome to evaluate
            cached_text_features: Pre-computed text features
        """
        # 🐛 调试：打印chromosome的seeds信息
        seeds_info = "None"
        if chromosome.seeds:
            if len(chromosome.seeds) > 0:
                base_seed = chromosome.seeds[0]
                seeds_info = f"base={base_seed}, mutations={len(chromosome.seeds)-1}"
                if len(chromosome.seeds) > 1:
                    # 显示前几个mutation
                    mutations_preview = chromosome.seeds[1:min(4, len(chromosome.seeds))]
                    seeds_info += f", preview={mutations_preview}"
        
        self._log_debug(f"Evaluating chromosome - seeds: {seeds_info}, num_params: {chromosome.num_params}")
        eval_start_time = time.time()
        
        try:
            # 🎯 SEED-BASED: Reconstruct weights from seeds and apply to LoRA layers
            apply_start = time.time()
            
            # 🐛 调试：检查应用权重前的模型权重（采样第一个LoRA层的第一个权重）
            weight_before = None
            if self.lora_layers and len(self.lora_layers) > 0:
                first_layer = self.lora_layers[0]
                if hasattr(first_layer, "q_proj") and hasattr(first_layer.q_proj, "w_lora_A"):
                    weight_before = first_layer.q_proj.w_lora_A[0, 0].item()
            
            self._apply_seeds_to_layers(chromosome)
            apply_time = time.time() - apply_start
            
            # 🐛 调试：检查应用权重后的模型权重
            weight_after = None
            if self.lora_layers and len(self.lora_layers) > 0:
                first_layer = self.lora_layers[0]
                if hasattr(first_layer, "q_proj") and hasattr(first_layer.q_proj, "w_lora_A"):
                    weight_after = first_layer.q_proj.w_lora_A[0, 0].item()
            
            if weight_before is not None and weight_after is not None:
                weight_changed = abs(weight_before - weight_after) > 1e-6
                self._log_debug(f"Weight change check - before: {weight_before:.6f}, after: {weight_after:.6f}, changed: {weight_changed}")
            else:
                self._log_debug(f"Warning: Could not check weight change (before={weight_before}, after={weight_after})")
            
            # 使用worker缓存的train_loader
            if self.train_loader is None:
                raise RuntimeError("Train loader not available in worker. Please provide dataset during worker initialization.")
            
            # 使用worker保存的classnames和template
            if self.classnames is None:
                raise RuntimeError("Classnames not provided in worker.")
            
            # 创建轻量级dataset包装器用于evaluate_lora
            class DatasetWrapper:
                def __init__(self, classnames, template):
                    self.classnames = classnames
                    self.template = template
            
            dataset_wrapper = DatasetWrapper(self.classnames, self.template)
            
            # Prepare features if provided
            local_text_features = None
            if cached_text_features is not None:
                local_text_features = cached_text_features.to(self.device)
            
            # Evaluate fitness using process-specific model
            eval_start = time.time()
            
            fitness = evaluate_lora(
                self.model, 
                self.train_loader,  # 使用缓存的train_loader
                dataset_wrapper,  # 轻量级dataset包装器
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
        cache_hit = cache_key in self.weight_cache
        if cache_hit:
            self._log_debug(f"Cache hit for seeds: {cache_key[:2] if len(cache_key) > 2 else cache_key}")
            return self.weight_cache[cache_key].copy()
        
        if chromosome.seeds is None or len(chromosome.seeds) == 0:
            raise ValueError("No seeds available for weight computation")
        
        # 🐛 调试：记录权重重建过程
        self._log_debug(f"Computing weights from seeds (cache miss) - base_seed: {chromosome.seeds[0]}, num_mutations: {len(chromosome.seeds)-1}")
        
        # Start with base seed
        base_seed = chromosome.seeds[0]
        if isinstance(base_seed, tuple):
            # Base seed is itself a mutation chain
            theta = self._compute_from_seed_chain(base_seed, chromosome.num_params)
        else:
            # Single base seed
            theta = self.noise_table.get(base_seed, chromosome.num_params).copy()
        
        # 🐛 调试：记录基础权重的统计信息
        self._log_debug(f"Base weights - mean: {theta.mean():.6f}, std: {theta.std():.6f}, min: {theta.min():.6f}, max: {theta.max():.6f}")
        
        # Apply subsequent mutations
        for i, mutation in enumerate(chromosome.seeds[1:]):
            if isinstance(mutation, tuple) and len(mutation) == 2:
                idx, power = mutation
                mutation_noise = self.noise_table.get(idx, chromosome.num_params)
                theta = theta + power * mutation_noise
                self._log_debug(f"Applied mutation {i+1}: idx={idx}, power={power:.6f}, new_mean={theta.mean():.6f}")
            else:
                # Handle old seed format
                idx = mutation
                mutation_noise = self.noise_table.get(idx, chromosome.num_params)
                theta = theta + 0.005 * mutation_noise
        
        # Cache the result
        self.weight_cache[cache_key] = theta.copy()
        
        # 🐛 调试：记录最终权重的统计信息
        self._log_debug(f"Final weights - mean: {theta.mean():.6f}, std: {theta.std():.6f}, min: {theta.min():.6f}, max: {theta.max():.6f}")
        
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
        total_weights_applied = 0
        
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.lora_layers):
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
                    
                    # 🐛 调试：记录第一个权重的值（用于验证权重是否不同）
                    if applied_pairs == 0:
                        first_weight_sample = a_flat[0] if len(a_flat) > 0 else 0
                        self._log_debug(f"Applying weights - first weight sample: {first_weight_sample:.6f}, "
                                      f"weights_flat mean: {weights_flat.mean():.6f}, std: {weights_flat.std():.6f}")
                    
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
                    total_weights_applied += a_size + b_size
        
        # 🐛 调试：验证权重应用
        if applied_pairs == 0:
            self._log_debug(f"WARNING: No weights applied! Check LoRA layer configuration.")
        else:
            self._log_debug(f"Weights applied - {applied_pairs} LoRA pairs, {total_weights_applied} total parameters")

    
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
def worker_initializer(gpu_id, base_model_state, lora_layers_state, noise_table, worker_id, 
                      dataset=None, classnames=None, template=None, debug=True):
    """Initialize worker process with model and LoRA layers.
    
    🚀 性能优化：传递dataset以便在worker中创建并缓存train_loader
    """
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
        
        # 为每个worker分配唯一的ID（基于进程ID和GPU ID）
        import os
        process_id = os.getpid()
        unique_worker_id = gpu_id * 10000 + (process_id % 10000)
        
        # 🚀 创建worker实例，传递dataset以便创建train_loader
        worker = GPUWorkerProcess(
            worker_id=unique_worker_id,
            gpu_id=gpu_id,
            base_model=base_model,
            lora_layers=lora_layers,
            noise_table=noise_table,
            dataset=dataset,       # 传递dataset以便创建train_loader
            classnames=classnames, # 传递classnames
            template=template,     # 传递template
            debug=debug
        )
        
        # Store worker in process-local storage
        mp.current_process()._worker = worker
        
        if debug:
            print(f"[WORKER] Worker {unique_worker_id} (PID {process_id}) initialized on GPU{gpu_id}")
            
        return worker
        
    except Exception as e:
        print(f"[WORKER] Error initializing worker {worker_id} on GPU{gpu_id}: {e}")
        import traceback
        print(f"[WORKER] Traceback: {traceback.format_exc()}")
        raise

def evaluate_chromosome_worker(args):
    """Worker function for evaluating a single chromosome.
    
    🚀 性能优化：train_loader已在worker初始化时缓存，无需传递
    """
    chromosome_data, cached_text_features = args
    worker = getattr(mp.current_process(), '_worker', None)
    
    if worker is None:
        raise RuntimeError("Worker not initialized in process")
    
    # 🐛 调试：验证chromosome是否正确传递
    if hasattr(chromosome_data, 'seeds'):
        seeds_repr = "None"
        if chromosome_data.seeds:
            if len(chromosome_data.seeds) > 0:
                seeds_repr = f"base={chromosome_data.seeds[0]}, len={len(chromosome_data.seeds)}"
        if worker.debug:
            import os
            pid = os.getpid()
            print(f"[WORKER PID {pid}] Received chromosome - seeds: {seeds_repr}, num_params: {chromosome_data.num_params}")
    
    return worker.evaluate_single(
        chromosome_data, cached_text_features
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
                 dataset=None,
                 debug: bool = True):
        """
        Initialize process-based parallel evaluator.
        
        Args:
            base_model: Base model architecture
            list_lora_layers: LoRA layers to parallelize
            noise_table: Shared noise table for seed-based weight reconstruction
            gpu_ids: List of GPU device IDs to use
            processes_per_gpu: Number of processes per GPU for true parallel evaluation
            dataset: Dataset object with train_loader, test_loader, val_loader attributes
            debug: Enable debug logging
        """
        self.base_model = base_model
        self.list_lora_layers = list_lora_layers
        self.noise_table = noise_table
        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        self.processes_per_gpu = processes_per_gpu
        self.debug = debug
        self.evaluation_count = 0
        
        # 保存dataset引用（用于传递给worker）
        self._dataset = dataset
        
        # 从dataset提取必要信息（classnames, template）
        if dataset is not None:
            self.classnames = dataset.classnames if hasattr(dataset, 'classnames') else None
            self.template = getattr(dataset, 'template', None)
            # 注意：train_loader不再在主进程中保存，而是在每个worker中创建
        else:
            self.classnames = None
            self.template = None
        
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
        """Initialize process pools for each GPU.
        
        🚀 性能优化：传递dataset以便在worker中创建并缓存train_loader
        """
        self._log_debug(f"Initializing process pools for {len(self.gpu_ids)} GPUs...")
        start_time = time.time()
        
        # 获取dataset对象（用于在worker中创建train_loader）
        # 注意：这里需要保存原始的dataset对象引用
        dataset_for_workers = None
        if hasattr(self, '_dataset') and self._dataset is not None:
            dataset_for_workers = self._dataset
        
        for gpu_id in self.gpu_ids:
            try:
                # Create process pool for this GPU
                # 🚀 传递dataset以便在worker初始化时创建train_loader
                pool = ProcessPoolExecutor(
                    max_workers=self.processes_per_gpu,
                    initializer=worker_initializer,
                    initargs=(
                        gpu_id,
                        self.base_model,
                        self.list_lora_layers,
                        self.noise_table,
                        gpu_id,  # worker_id (will be made unique per process)
                        dataset_for_workers,  # 🚀 传递dataset以便创建train_loader
                        self.classnames,  # 传递classnames
                        self.template,    # 传递template
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
    
    def evaluate_population_parallel(self, population: List[Chromosome], 
                                   cached_text_features: Optional[torch.Tensor] = None) -> None:
        """
        Evaluate population using process-based parallelism with seed-based weight transfer.
        
        Args:
            population: List of chromosomes to evaluate (only seeds are transferred)
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
        
        # 检查dataset是否可用
        if self._dataset is None:
            raise RuntimeError("Dataset not available. Please provide dataset with train_loader in __init__.")
        
        # Fallback to sequential evaluation if no process pools available
        if not self.process_pools:
            self._log_debug("No process pools available, using sequential evaluation")
            self._evaluate_sequential(population, cached_text_features)
            return
        
        # Distribute chromosomes across GPUs (round-robin)
        tasks = self._distribute_tasks(population, cached_text_features)
        
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
    
    def _distribute_tasks(self, population: List[Chromosome], 
                         cached_text_features: Optional[torch.Tensor] = None) -> List[Tuple[int, Tuple]]:
        """
        Distribute chromosomes across GPUs in round-robin fashion.
        
        🚀 性能优化：train_loader已在worker中缓存，只需传递chromosome和cached_text_features
        
        Returns:
            List of (gpu_id, task_data) tuples
        """
        tasks = []
        gpu_ids = list(self.process_pools.keys())
        
        # 🐛 调试：检查population中chromosomes的seeds是否不同
        unique_seeds = set()
        seeds_list = []
        base_seeds = []
        for idx, chrom in enumerate(population):
            if chrom.need_update and chrom.seeds:
                # 创建seeds的hashable表示用于比较
                seeds_tuple = tuple(chrom.seeds) if isinstance(chrom.seeds, (list, tuple)) else (chrom.seeds,)
                unique_seeds.add(seeds_tuple)
                seeds_list.append((idx, seeds_tuple[:3] if len(seeds_tuple) > 3 else seeds_tuple))  # 只记录前3个
                # 记录base seed
                if len(seeds_tuple) > 0:
                    base_seed = seeds_tuple[0]
                    base_seeds.append(base_seed)
        
        self._log_debug(f"Population seeds check - total: {len(population)}, need_update: {sum(1 for c in population if c.need_update)}, unique_seeds: {len(unique_seeds)}")
        
        # 🐛 检查base seeds是否相同
        unique_base_seeds = set(base_seeds)
        if len(unique_base_seeds) == 1 and len(base_seeds) > 1:
            self._log_debug(f"ERROR: All chromosomes have the same base seed: {base_seeds[0] if base_seeds else 'None'}")
            self._log_debug(f"This means all chromosomes will have similar or identical weights!")
        
        if len(unique_seeds) < 3 and len(population) >= 3:
            self._log_debug(f"WARNING: Only {len(unique_seeds)} unique seeds for {len(population)} chromosomes!")
            self._log_debug(f"Sample seeds (first 10): {seeds_list[:10]}")
            self._log_debug(f"Base seeds (first 20): {base_seeds[:20]}")
        
        for idx, chrom in enumerate(population):
            if not chrom.need_update:
                continue
                
            # Round-robin distribution across GPUs
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            
            # 🐛 调试：记录chromosome的seeds信息
            chrom_seeds_preview = "None"
            if chrom.seeds:
                if len(chrom.seeds) > 0:
                    chrom_seeds_preview = f"{chrom.seeds[0]}" + (f"+{len(chrom.seeds)-1}muts" if len(chrom.seeds) > 1 else "")
            
            # 🚀 只需传递chromosome和cached_text_features
            # train_loader、classnames、template已在worker初始化时缓存
            task_data = (
                chrom,  # chromosome_data
                cached_text_features  # 只传递cached_text_features
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
    
    def _evaluate_sequential(self, population: List[Chromosome], 
                           cached_text_features: Optional[torch.Tensor] = None):
        """Fallback sequential evaluation when no process pools are available."""
        self._log_debug("Using sequential evaluation fallback")
        
        if self._dataset is None:
            raise RuntimeError("Dataset not available for sequential evaluation.")
        
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
            dataset=self._dataset,  # 传递dataset以便创建train_loader
            classnames=self.classnames,
            template=self.template,
            debug=self.debug
        )
        
        for i, chrom in enumerate(population):
            if chrom.need_update:
                try:
                    fitness = worker.evaluate_single(
                        chrom, cached_text_features  # train_loader已在worker中缓存
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
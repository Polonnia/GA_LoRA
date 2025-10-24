import random
import torch
import torch.nn.functional as F
from deap import base, creator, tools
import numpy as np
from tqdm import tqdm
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from contextlib import contextmanager
import math
from utils import *
from loralib.utils import mark_only_lora_as_trainable, apply_lora, save_lora, load_lora
from loralib import layers as lora_layers


# ------------------------------
# GA 可调参数（集中配置）
# ------------------------------
GA_POP_SIZE_DEFAULT = 100
GA_GENERATIONS_DEFAULT = 300

# 选择策略
GA_TOURNAMENT_SIZE = 2
# 精英保留数量
NUM_ELITES_DEFAULT = 4

# 交叉（SBX）
GA_CROSSOVER_PROB = 0.7
GA_CROSSOVER_ETA = 20  # SBX 的分布指数

# 变异（多项式变异，自适应）
GA_MUTATE_INDIVIDUAL_PROB = 0.3  # 以该概率对个体触发一次变异
GA_MUTATION_RATE = 1            # 基因元素级被选中变异的概率
GA_MUTATION_ETA = 20              # 多项式分布指数（eta）
LORA_WIDTH = 512
GA_MUTATION_SCALE = 0.1       # 初始步长系数，随代数线性衰减

# ------------------------------
# 工具：设备发现与上下文
# ------------------------------
def get_visible_devices():
    """返回可用 GPU 设备列表，如 ['cuda:0','cuda:1', ...]；若无 GPU 则回退到 ['cpu']"""
    if torch.cuda.is_available():
        return ['cuda:0']
    return ['cpu']

@contextmanager
def cuda_no_sync():
    """评估阶段的上下文（可扩展，例如 autocast 混合精度）"""
    with torch.no_grad():
        yield


# ------------------------------
# 工具：LoRA 层名提取与映射
# ------------------------------
def _collect_module_name_map(model):
    """返回 {模块对象id: 完整层名} 与 {完整层名: 模块对象} 两个映射"""
    id2name = {}
    name2mod = {}
    for n, m in model.named_modules():
        name2mod[n] = m
        id2name[id(m)] = n
    return id2name, name2mod

def get_lora_submodule_names(model, lora_layer_list):
    """返回启用的LoRA子模块(LinearLoRA)在模型中的完整名称列表。
    这里的子模块应为 PlainMultiheadAttentionLoRA 下的 q_proj/k_proj/v_proj/proj。
    """
    id2name, _ = _collect_module_name_map(model)
    names = []
    for layer in lora_layer_list:
        if hasattr(layer, 'q_proj'):
            names.append(id2name[id(layer.q_proj)])
        if hasattr(layer, 'k_proj'):
            names.append(id2name[id(layer.k_proj)])
        if hasattr(layer, 'v_proj'):
            names.append(id2name[id(layer.v_proj)])
        if hasattr(layer, 'proj'):
            names.append(id2name[id(layer.proj)])
    return names

def apply_individual_to_model(model, lora_layer_names, individual, device):
    """将个体参数按顺序写入每个LinearLoRA子模块的A/B矩阵。
    individual是一个扁平列表: [A_q, B_q, A_k, B_k, A_v, B_v, A_o, B_o, ...]
    lora_layer_names是LinearLoRA子模块名列表，与individual按照每子模块2个张量对齐。
    """
    _, name2mod = _collect_module_name_map(model)
    idx = 0
    for name in lora_layer_names:
        mod = name2mod[name]
        if hasattr(mod, 'w_lora_A') and hasattr(mod, 'w_lora_B'):
            a_t = individual[idx].to(device=device, dtype=mod.w_lora_A.dtype)
            b_t = individual[idx + 1].to(device=device, dtype=mod.w_lora_B.dtype)
            mod.w_lora_A.data.copy_(a_t)
            mod.w_lora_B.data.copy_(b_t)
            idx += 2


# ------------------------------
# 初始化种群
# ------------------------------
def initialize_population(pop_size, lora_layers):
    """个体以所有启用的LinearLoRA子模块的(A,B)按顺序扁平化组成: [A, B, A, B, ...]
    含一个“基线个体”（不加噪声，等价于当前模型 LoRA 参数），其余为带噪声的个体。
    """
    # 构造基线个体（无噪声）
    baseline_genes = []
    for layer in lora_layers:
        for sub in ['q_proj', 'k_proj', 'v_proj', 'proj']:
            if hasattr(layer, sub):
                mod = getattr(layer, sub)
                if hasattr(mod, 'w_lora_A') and hasattr(mod, 'w_lora_B'):
                    a = mod.w_lora_A.data.detach().clone().cpu()
                    b = mod.w_lora_B.data.detach().clone().cpu()
                    baseline_genes.append(a)
                    baseline_genes.append(b)

    population = []
    # 生成 pop_size-1 个带噪声的个体
    for _ in range(max(0, pop_size - 1)):
        genes = []
        for layer in lora_layers:
            for sub in ['q_proj', 'k_proj', 'v_proj', 'proj']:
                if hasattr(layer, sub):
                    mod = getattr(layer, sub)
                    if hasattr(mod, 'w_lora_A') and hasattr(mod, 'w_lora_B'):
                        n = mod.w_lora_A.size(1)
                        import math
                        std_dev = 10 / math.sqrt(n)
                        a = mod.w_lora_A.data.detach().clone().cpu()
                        b = mod.w_lora_B.data.detach().clone().cpu()
                        genes.append(a + torch.randn_like(a) * std_dev)
                        genes.append(b + torch.randn_like(b) * std_dev)
        population.append(creator.Individual(genes))

    # 将基线个体插入首位，保证至少一个个体等价于当前模型
    population.insert(0, creator.Individual(baseline_genes))
    # 防御式截断（理论上长度已为 pop_size）
    if len(population) > pop_size:
        population = population[:pop_size]
    return population


# ------------------------------
# SBX 与多项式变异
# ------------------------------
def simulated_binary_crossover(ind1, ind2, eta=GA_CROSSOVER_ETA):
    """逐基因(SBX)对A/B进行交叉。个体是[A, B, A, B, ...]扁平序列"""
    child1, child2 = [], []
    for g1, g2 in zip(ind1, ind2):
        if isinstance(g1, list) and len(g1) == 1:
            g1 = g1[0]
        if isinstance(g2, list) and len(g2) == 1:
            g2 = g2[0]
        rand = torch.rand_like(g1)
        beta = torch.where(
            rand <= 0.5,
            (2.0 * rand) ** (1.0 / (eta + 1)),
            (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1))
        )
        c1 = 0.5 * ((1 + beta) * g1 + (1 - beta) * g2)
        c2 = 0.5 * ((1 - beta) * g1 + (1 + beta) * g2)
        child1.append(c1)
        child2.append(c2)
    return child1, child2


def polynomial_mutation(individual, mutation_rate=GA_MUTATION_RATE, eta=GA_MUTATION_ETA, scale=GA_MUTATION_SCALE):
    """逐基因多项式变异，独立作用于每个A/B张量。"""
    mutated = []
    for gene in individual:
        g = gene.clone()
        mask = (torch.rand_like(g) < mutation_rate)
        if mask.any():
            r = torch.rand_like(g)
            delta = torch.where(
                r < 0.5,
                (2.0 * r) ** (1.0 / (eta + 1)) - 1.0,
                1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1))
            )
            g[mask] += scale * delta[mask]
        mutated.append(g)
    return mutated

def layer_aware_mutation(individual, mutation_rate=0.1, eta=20,
                         base_scale=0.2, layer_factors=None):
    if layer_factors is None:
        layer_factors = [1.0] * len(individual)
    mutated = []
    for layer, factor in zip(individual, layer_factors):
        mutated.append(polynomial_mutation([layer], mutation_rate, eta, base_scale * factor)[0])
    return mutated


# ------------------------------
# 自适应调整变异和交叉概率
# ------------------------------

def adaptive_mutation_rate(best_fitness_history, mutation_rate=GA_MUTATION_RATE):
    """根据适应度历史动态调整变异率"""
    if len(best_fitness_history) < 2:
        return mutation_rate  # 不足两代时返回当前变异率

    fitness_diff = best_fitness_history[-1] - best_fitness_history[-2]

    # 如果适应度没有提升，增加变异率
    if fitness_diff <= 0:
        mutation_rate = min(0.9, mutation_rate + 0.05)  # 增加变异率，不超过0.5

    # 如果适应度持续改进，减小变异率
    else:
        mutation_rate = max(0.1, mutation_rate - 0.02)  # 减小变异率，保持较小的搜索步伐

    return mutation_rate


def adaptive_crossover_prob(best_fitness_history, crossover_prob=GA_CROSSOVER_PROB):
    """根据适应度历史动态调整交叉概率"""
    if len(best_fitness_history) < 5:
        return crossover_prob  # 不足5代时返回当前交叉概率

    fitness_diff = best_fitness_history[-1] - best_fitness_history[-5]

    # 如果适应度没有提升，增加交叉概率
    if fitness_diff <= 0:
        crossover_prob = min(1.0, crossover_prob + 0.05)  # 增加交叉概率，不超过1

    # 如果适应度持续改进，减少交叉概率
    else:
        crossover_prob = max(0.6, crossover_prob - 0.05)  # 减少交叉概率，避免过多扰动

    return crossover_prob


# ------------------------------
# 核心：并行评估器
# ------------------------------
class ParallelEvaluator:
    """
    基于多 GPU 并行评估个体。
    用法：
      pe = ParallelEvaluator(base_model, lora_layer_names, train_loader, dataset, devices)
      fitness_list = pe.evaluate_many(pop_chunk)  # 返回与 pop_chunk 等长的分数数组
    """
    def __init__(self, base_model, lora_layer_names, train_loader, dataset, devices=None, use_amp=True, args=None):
        self.devices = devices or get_visible_devices()
        self.use_amp = use_amp and (self.devices and self.devices[0].startswith('cuda'))
        self.dataset = dataset
        self.train_loader = train_loader  # 原始 DataLoader（不在线程中直接使用）
        self.args = args
        # 为每块设备克隆独立模型
        self.models = []
        self.lora_layer_names = lora_layer_names
        base = base_model.module if isinstance(base_model, torch.nn.DataParallel) else base_model
        base_cpu = deepcopy(base).cpu()  # 先放 CPU，减少不必要的拷贝
        for dev in self.devices:
            m = deepcopy(base_cpu).to(dev)
            m.eval()
            self.models.append(m)
        # 为每个设备构建独立且无多进程的 DataLoader，避免在线程中共享多进程 DataLoader 导致崩溃
        try:
            from torch.utils.data import DataLoader as _DL
            batch_size = getattr(train_loader, 'batch_size', 32)
            collate_fn = getattr(train_loader, 'collate_fn', None)
            dataset = getattr(train_loader, 'dataset', None)
            drop_last = getattr(train_loader, 'drop_last', False)
            sampler = None  # 评估不需要分布式采样
            self.device_loaders = []
            for _ in self.devices:
                dl = _DL(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         sampler=sampler,
                         num_workers=0,
                         pin_memory=False,
                         collate_fn=collate_fn,
                         drop_last=drop_last)
                self.device_loaders.append(dl)
        except Exception:
            # 回退到共享 loader（不推荐，但保持兼容）
            self.device_loaders = [self.train_loader for _ in self.devices]

        # 构建每设备缓存：若仅微调 vision，则缓存文本特征；若仅微调 text，则缓存图像特征
        self.cached_text_features = [None] * len(self.devices)
        self.cached_image_batches = [None] * len(self.devices)
        try:
            mode = getattr(self.args, 'encoder', 'both') if self.args is not None else 'both'
            from utils import unwrap
            for idx, dev in enumerate(self.devices):
                model = self.models[idx]
                base = unwrap(model)
                # 缓存文本特征（仅调 vision 或 both!=text）
                if mode == 'vision':
                    template = self.dataset.template[0]
                    texts = [template.format(c.replace('_', ' ')) for c in self.dataset.classnames]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        tokenized = clip.tokenize(texts).to(dev)
                        class_embeddings = base.encode_text(tokenized)
                    tf = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                    self.cached_text_features[idx] = tf
                # 缓存图像特征（仅调 text）
                if mode == 'text':
                    cached_batches = []
                    with torch.no_grad():
                        for images, target in self.device_loaders[idx]:
                            images = images.to(dev, non_blocking=True)
                            target = target.to(dev, non_blocking=True)
                            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                                img_feat = base.encode_image(images)
                            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                            cached_batches.append((img_feat, target))
                    self.cached_image_batches[idx] = cached_batches
        except Exception as _e:
            # 缓存失败则跳过，不影响主流程
            pass

    def _eval_one(self, device_idx, individual):
        device = self.devices[device_idx]
        model = self.models[device_idx]
        # 把个体参数写到该设备模型
        apply_individual_to_model(model, self.lora_layer_names, individual, device)

        # 评估：你已有的 evaluate_lora，如果它内部会 to(device)，请把上面的 to(dev) 去掉
        from utils import evaluate_lora

        with cuda_no_sync():
            # 可选混合精度
            if self.use_amp and device.startswith('cuda'):
                with torch.cuda.amp.autocast():
                    score = evaluate_lora(
                        model,
                        self.device_loaders[device_idx],
                        self.dataset,
                        cached_text_features=self.cached_text_features[device_idx],
                        cached_image_batches=self.cached_image_batches[device_idx]
                    )
            else:
                score = evaluate_lora(
                    model,
                    self.device_loaders[device_idx],
                    self.dataset,
                    cached_text_features=self.cached_text_features[device_idx],
                    cached_image_batches=self.cached_image_batches[device_idx]
                )
        # 返回 float
        return float(score)

    def evaluate_many(self, individuals):
        """
        把 individuals（list[individual]）按设备数分发并行评估，返回同序列的 fitness 列表
        """
        if len(self.devices) == 1:
            # 单设备就顺序跑（带进度条）
            results_seq = []
            pbar = tqdm(total=len(individuals), desc="GA eval (1 device)", leave=False)
            for ind in individuals:
                results_seq.append(self._eval_one(0, ind))
                pbar.update(1)
            pbar.close()
            return results_seq

        results = [None] * len(individuals)
        # 分批提交，每批最多等于设备数
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=len(individuals), desc=f"GA eval ({len(self.devices)} devices)", leave=False)
        with ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
            idx = 0
            while idx < len(individuals):
                futures = {}
                for d in range(len(self.devices)):
                    if idx >= len(individuals):
                        break
                    fut = ex.submit(self._eval_one, d, individuals[idx])
                    futures[fut] = idx
                    idx += 1
                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = fut.result()
                    pbar.update(1)
        pbar.close()
        return results


# ------------------------------
# DEAP 集成：把 map 换成并行 map
# ------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def ga_optimize_lora(args, clip_model, lora_layers_list, train_loader, dataset,
                     generations=GA_GENERATIONS_DEFAULT, pop_size=GA_POP_SIZE_DEFAULT):
    """
    GA 优化 LoRA —— 使用多 GPU 并行评估个体。
    """
    # 准备 LoRA 层的子模块名（q, k, v, o）
    lora_layer_names = get_lora_submodule_names(clip_model, lora_layers_list)

    # 初始化种群（个体参数都放在 CPU，评估时拷到对应 GPU）
    population = initialize_population(pop_size, lora_layers_list)

    # 仅保留由 args.params 指定的子模块（例如只用 q 与 v）
    try:
        param2sub = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'proj'}
        allowed_subs = [param2sub[p] for p in getattr(args, 'params', ['q', 'v']) if p in param2sub]
        # 计算需要保留的子模块索引，与 initialize_population 的遍历顺序一致
        id2name, _ = _collect_module_name_map(clip_model)
        sub_order = ['q_proj', 'k_proj', 'v_proj', 'proj']
        selected_sub_indices = []
        selected_submodule_names = []
        running_index = 0
        for layer in lora_layers_list:
            for sub in sub_order:
                if hasattr(layer, sub):
                    mod = getattr(layer, sub)
                    # 仅当该子模块实际具有 LoRA 权重时，才参与基因序列与索引
                    if hasattr(mod, 'w_lora_A') and hasattr(mod, 'w_lora_B'):
                        if sub in allowed_subs:
                            selected_sub_indices.append(running_index)
                            selected_submodule_names.append(id2name[id(mod)])
                        running_index += 1

        if selected_sub_indices:
            # 过滤种群基因：每个子模块对应两个基因（A,B）
            filtered_population = []
            for ind in population:
                new_genes = []
                for si in selected_sub_indices:
                    a_idx = 2 * si
                    b_idx = a_idx + 1
                    if a_idx < len(ind) and b_idx < len(ind):
                        new_genes.append(ind[a_idx])
                        new_genes.append(ind[b_idx])
                filtered_population.append(creator.Individual(new_genes))
            population = filtered_population
            lora_layer_names = selected_submodule_names
    except Exception as _e:
        print(f"[GA][debug] Failed to filter params {getattr(args,'params',None)}: {_e}")

    # 调试：打印LoRA子模块与individual基因结构（仅一次，简洁输出）
    try:
        sample = population[0]
        num_pairs = min(len(lora_layer_names), len(sample) // 2)
        print("[GA][debug] Individual structure (LoRA submodules -> A/B shapes):")
        for i in range(num_pairs):
            name = lora_layer_names[i]
            a_shape = tuple(sample[2 * i].shape) if hasattr(sample[2 * i], 'shape') else 'n/a'
            b_shape = tuple(sample[2 * i + 1].shape) if hasattr(sample[2 * i + 1], 'shape') else 'n/a'
            print(f"  - {name}: A{a_shape}, B{b_shape}")
        if len(sample) != 2 * len(lora_layer_names):
            print(f"  [warn] gene count {len(sample)} != 2 * submodules {len(lora_layer_names)}")
    except Exception as e:
        print(f"[GA][debug] Failed to summarize individual: {e}")

    # 并行评估器：每块 GPU 一份独立模型
    devices = get_visible_devices()
    print(f"[GA] Using devices: {devices}")
    parallel_eval = ParallelEvaluator(
        base_model=clip_model,
        lora_layer_names=lora_layer_names,
        train_loader=train_loader,
        dataset=dataset,
        devices=devices,
        use_amp=True,
        args=args
    )

    # DEAP 基本设置
    toolbox = base.Toolbox()
    toolbox.clone = lambda x: deepcopy(x)
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: random.choice(population))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", simulated_binary_crossover)
    toolbox.register("mutate", polynomial_mutation)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)

    # 评估适配器：给 DEAP 用
    def evaluate_individual(individual):
        score = parallel_eval.evaluate_many([individual])[0]  # 使用并行评估器
        individual.fitness.values = (score,)  # 设置适应度值
        return (score,)

    toolbox.register("evaluate", evaluate_individual)

    # 自定义并行 map
    def gpu_parallel_map(func, iterable):
        inds = list(iterable)
        scores = parallel_eval.evaluate_many(inds)
        return [(s,) for s in scores]

    toolbox.map = gpu_parallel_map

    # GA 主循环
    best_fitness_history = []
    # 自适应参数（每代更新）
    current_individual_mutation_rate = GA_MUTATION_RATE
    current_crossover_prob = GA_CROSSOVER_PROB

    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")

        # 评估无效个体（并行）
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # 记录当代最佳
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        best_fitness_history.append(best_fitness)
        print(f"Best accuracy in generation {gen + 1}: {best_fitness:.4f}")

        # 将当代最佳写回模型并保存 LoRA
        try:
            target_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            _, name2mod = _collect_module_name_map(clip_model)
            idx_apply = 0
            for name in lora_layer_names:
                mod = name2mod[name]
                if hasattr(mod, 'w_lora_A') and hasattr(mod, 'w_lora_B') and idx_apply + 1 < len(best_individual):
                    a_t = best_individual[idx_apply].to(device=target_device, dtype=mod.w_lora_A.dtype)
                    b_t = best_individual[idx_apply + 1].to(device=target_device, dtype=mod.w_lora_B.dtype)
                    mod.w_lora_A.data.copy_(a_t)
                    mod.w_lora_B.data.copy_(b_t)
                    idx_apply += 2
            # 保存
            if getattr(args, 'save_path', None):
                orig_filename = getattr(args, 'filename', 'lora_weights')
                try:
                    args.filename = f"{orig_filename}_acc{best_fitness:.4f}"
                    save_lora(args, lora_layers_list)
                    print(f"[GA] Saved best of gen{gen+1} with acc={best_fitness:.4f}")
                finally:
                    args.filename = orig_filename
        except Exception as _e:
            print(f"[GA][warn] Failed to save best individual of gen{gen+1}: {_e}")

        # 基于历史自适应调整概率（作用于本代之后的遗传操作）
        current_individual_mutation_rate = adaptive_mutation_rate(best_fitness_history, current_individual_mutation_rate)
        current_crossover_prob = adaptive_crossover_prob(best_fitness_history, current_crossover_prob)
        print(f"[GA] gen={gen + 1} mut_rate={current_individual_mutation_rate:.3f} cross_prob={current_crossover_prob:.3f}")

        # 选择、克隆
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉（使用自适应交叉概率）
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < current_crossover_prob:
                nc1, nc2 = toolbox.mate(c1, c2)
                c1[:] = nc1
                c2[:] = nc2
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

        # 变异（使用自适应变异率传入算子；个体级触发概率不变）
        for m in offspring:
            if random.random() < GA_MUTATE_INDIVIDUAL_PROB:
                nm = toolbox.mutate(m, mutation_rate=current_individual_mutation_rate)
                m[:] = nm
                if hasattr(m.fitness, "values"):
                    del m.fitness.values

        # 精英保留
        # 保留4个精英
        num_elites = NUM_ELITES_DEFAULT
        elites = tools.selBest(population, num_elites)
        for i in range(num_elites):
            offspring[-(i + 1)] = toolbox.clone(elites[i])

        # 更新种群
        population[:] = offspring

    # 返回最优解与前K个精英个体
    best_solution = tools.selBest(population, 1)[0]
    topk_individuals = tools.selBest(population, min(5, len(population)))
    return best_solution, best_fitness_history, topk_individuals


def run_lora_ga(args, clip_model, dataset, train_loader, test_loader):
    """通过GA优化LoRA进行训练和评估"""
    VALIDATION = False

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() if torch.cuda.is_available() else clip_model

    if args.eval_only:
        load_lora(args, list_lora_layers)
        from utils import evaluate
        evaluate(clip_model, 'ga', test_loader, dataset, args.eval_datasets, args.result_path, args.seed, args.root_path)
        return

    mark_only_lora_as_trainable(clip_model)

    ga_start_ts = time.time()
    best_solution, fitness_history, topk_individuals = ga_optimize_lora(
        args, clip_model, list_lora_layers, train_loader, dataset)
    ga_end_ts = time.time()
    ga_duration_s = float(ga_end_ts - ga_start_ts)

    # 保存 GA 运行时间与历史
    try:
        result_dir = getattr(args, 'result_path', None)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            summary = {
                'seed': int(getattr(args, 'seed', 0)),
                'ga_pop_size': int(GA_POP_SIZE_DEFAULT),
                'ga_generations': int(GA_GENERATIONS_DEFAULT),
                'encoder': getattr(args, 'encoder', None),
                'params': list(getattr(args, 'params', [])) if getattr(args, 'params', None) is not None else None,
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ga_start_ts)),
                'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ga_end_ts)),
                'duration_seconds': ga_duration_s,
                'best_fitness_per_generation': [float(x) for x in fitness_history],
            }
            out_path = os.path.join(result_dir, f'ga_run_seed{getattr(args, "seed", 0)}.json')
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"[GA] Saved GA run summary to: {out_path}")
    except Exception as e:
        print(f"[GA][warn] Failed to save GA run summary: {e}")

    # 把最佳解写回主模型（按A/B写回LinearLoRA子模块）
    target_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _, name2mod = _collect_module_name_map(clip_model)
    lora_layer_names = get_lora_submodule_names(clip_model, list_lora_layers)
    idx = 0
    for name in lora_layer_names:
        mod = name2mod[name]
        if hasattr(mod, 'w_lora_A') and hasattr(mod, 'w_lora_B') and idx + 1 < len(best_solution):
            a_t = best_solution[idx].to(device=target_device, dtype=mod.w_lora_A.dtype)
            b_t = best_solution[idx + 1].to(device=target_device, dtype=mod.w_lora_B.dtype)
            mod.w_lora_A.data.copy_(a_t)
            mod.w_lora_B.data.copy_(b_t)
            idx += 2

    from utils import evaluate
    acc_test = evaluate(clip_model, 'ga', test_loader, dataset, args.eval_datasets, args.result_path, args.seed, args.root_path)

    if getattr(args, 'save_path', None):
        # 保存当前最佳到 save_path，文件名包含最终准确率
        args.filename = f"{args.filename}_ga_acc{acc_test:.4f}"
        save_lora(args, list_lora_layers)

        # 另外保存前5名个体的 LoRA 权重到 save_path/{shots}shots/seed{seed}/ga_top5，使用 save_lora
        try:
            save_root = getattr(args, 'save_path', None)
            shots_dir = f"{args.shots}shots/seed{args.seed}"
            save_dir = os.path.join(save_root, shots_dir, 'ga_top5')
            os.makedirs(save_dir, exist_ok=True)

            # 逐个体写回并导出 LoRA state_dict，文件名包含准确率
            orig_filename = getattr(args, 'filename', 'lora_weights')
            rank = 1
            for ind in topk_individuals:
                apply_individual_to_model(clip_model, lora_layer_names, ind, target_device)
                ind_acc = 0.0
                try:
                    ind_acc = float(ind.fitness.values[0])
                except Exception:
                    pass
                # 使用 save_lora 保存，临时改写 filename 为子路径 'ga_top5/...'
                try:
                    args.filename = f"ga_top5/lora_top{rank}_acc{ind_acc:.4f}"
                    save_lora(args, list_lora_layers)
                    print(f"[GA] Saved top-{rank} LoRA via save_lora with acc={ind_acc:.4f}")
                finally:
                    args.filename = orig_filename
                rank += 1
                if rank > 5:
                    break
        except Exception as e:
            print(f"[GA][warn] Failed to save top-5 individuals: {e}")

    return best_solution, fitness_history

# 全局配置参数
import torch

# 遗传算法参数
POPULATION_SIZE = 100
NUM_GENERATIONS = 4000
MUTATION_RATE = 0.30
MUTATION_RATIO = 0.20
NUM_ELITES = 10
NUM_PARENTS = 20
STD_DEV = 0.1
SEED = 42

# 并行参数
NUM_WORKERS = torch.cuda.device_count() if torch.cuda.is_available() else 1
BATCH_SIZE_PER_WORKER = 16

# 噪声表参数
NOISE_TABLE_SIZE = 100000000
NOISE_SEED = 42

# 调度器参数
MUTATION_SCHEDULE_TYPE = 'linear'  # 'constant', 'linear', 'exponential'
FINAL_MUTATION_POWER = 0.01  # 最终突变强度
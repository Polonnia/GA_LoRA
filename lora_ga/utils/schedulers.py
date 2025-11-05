import math
from typing import Optional

class MutationScheduler:
    """突变强度调度器"""
    def __init__(self, 
                 initial_power: float = 0.1,
                 schedule_type: str = 'constant',
                 final_power: Optional[float] = None,
                 schedule_steps: int = 1000):
        
        self.initial_power = initial_power
        self.schedule_type = schedule_type
        self.final_power = final_power if final_power is not None else initial_power
        self.schedule_steps = schedule_steps
        
    def get_power(self, iteration: int, timesteps_so_far: int) -> float:
        """根据进度获取突变强度"""
        if self.schedule_type == 'constant':
            return self.initial_power
        
        elif self.schedule_type == 'linear':
            fraction = min(float(timesteps_so_far) / self.schedule_steps, 1.0)
            return self.initial_power + fraction * (self.final_power - self.initial_power)
        
        elif self.schedule_type == 'exponential':
            fraction = min(float(timesteps_so_far) / self.schedule_steps, 1.0)
            log_initial = math.log(self.initial_power)
            log_final = math.log(self.final_power)
            return math.exp(log_initial + fraction * (log_final - log_initial))
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing
            fraction = min(float(timesteps_so_far) / self.schedule_steps, 1.0)
            return self.final_power + 0.5 * (self.initial_power - self.final_power) * (1 + math.cos(math.pi * fraction))
        
        else:
            return self.initial_power

class AdaptiveMutationScheduler(MutationScheduler):
    """自适应突变调度器"""
    def __init__(self, initial_power=0.1, min_power=0.01, max_power=0.5, 
                 improvement_threshold=0.01, adjustment_rate=1.1):
        super().__init__(initial_power, 'constant')
        self.min_power = min_power
        self.max_power = max_power
        self.improvement_threshold = improvement_threshold
        self.adjustment_rate = adjustment_rate
        self.last_best_fitness = None
        
    def update(self, current_best_fitness: float):
        """根据性能改进调整突变强度"""
        if self.last_best_fitness is None:
            self.last_best_fitness = current_best_fitness
            return self.initial_power
        
        improvement = current_best_fitness - self.last_best_fitness
        self.last_best_fitness = current_best_fitness
        
        if improvement < self.improvement_threshold:
            # 改进不足，增加突变强度
            self.initial_power = min(self.initial_power * self.adjustment_rate, self.max_power)
        else:
            # 有改进，减小突变强度
            self.initial_power = max(self.initial_power / self.adjustment_rate, self.min_power)
        
        return self.initial_power
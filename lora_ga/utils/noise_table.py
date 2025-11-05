import numpy as np
from typing import Union
import torch

class SharedNoiseTable:
    """共享噪声表，类似参考代码的实现"""
    def __init__(self, size: int = 100000000, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.noise = self.rng.randn(size).astype(np.float32)
        self.size = size
        
    def get(self, idx: int, dim: int) -> np.ndarray:
        """获取指定位置的噪声向量"""
        if idx + dim > self.size:
            # 循环使用噪声表
            idx = idx % (self.size - dim)
        return self.noise[idx:idx+dim]
    
    def sample_index(self, dim: int) -> int:
        """随机采样噪声索引"""
        return self.rng.randint(0, self.size - dim + 1)
    
    def get_as_tensor(self, idx: int, shape: tuple, dtype=torch.float32) -> torch.Tensor:
        """获取噪声并转换为指定形状的tensor"""
        dim = np.prod(shape)
        noise_array = self.get(idx, dim)
        return torch.from_numpy(noise_array).reshape(shape).to(dtype)
    
    def mutate_tensor(self, tensor: torch.Tensor, idx: int, power: float) -> torch.Tensor:
        """对tensor应用噪声突变"""
        noise_tensor = self.get_as_tensor(idx, tensor.shape, tensor.dtype)
        return tensor + noise_tensor * power
import json
import os
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def load_generations(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # expect list of {generation, best_fitness, avg_fitness}
    return data


def plot_ga(json_path: str, out_path: str = None):
    data = load_generations(json_path)
    gens = np.array([int(d.get('generation', i)) for i, d in enumerate(data)])
    best = [float(d.get('best_fitness', 0.0) if d.get('best_fitness') is not None else 0.0) for d in data]
    avg = [float(d.get('avg_fitness', 0.0) if d.get('avg_fitness') is not None else 0.0) for d in data]

    # 使用原始代数作为横坐标，不再使用对数
    x = gens

    plt.figure(figsize=(12, 6))
    plt.plot(x, best, label='best_fitness', marker='o', markersize=3, linewidth=1)
    plt.plot(x, avg, label='avg_fitness', marker='x', markersize=3, linewidth=1)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('GA Fitness Over Generations')
    plt.grid(True)
    plt.legend()

    # 设置x轴刻度：每100代一个刻度
    try:
        if len(gens) > 0:
            max_gen = max(gens)
            # 生成100, 200, 300...的刻度
            xticks = np.arange(0, max_gen + 100, 100)
            plt.xticks(xticks)
            
            # 如果数据点太多，可以稀疏显示标记点
            if len(gens) > 50:
                # 每10个点显示一个标记
                marker_indices = np.arange(0, len(gens), 10)
                lines = plt.gca().get_lines()
                for line in lines:
                    # 获取当前线的数据
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    # 清除所有标记
                    line.set_marker(None)
                    # 在稀疏的位置添加标记
                    for idx in marker_indices:
                        if idx < len(x_data):
                            plt.scatter(x_data[idx], y_data[idx], 
                                      marker=line.get_marker() if line.get_label() == 'best_fitness' else 'x',
                                      color=line.get_color(),
                                      s=30)
    except Exception as e:
        print(f"Error setting ticks: {e}")

    # 修复输出路径处理
    if out_path is None:
        # 使用默认目录
        default_dir = "/root/aProject/plot"
        os.makedirs(default_dir, exist_ok=True)  # 确保目录存在
        out_path = os.path.join(default_dir, 'ga_plot.png')
    else:
        # 如果传入的是目录，而不是完整文件路径
        if out_path.endswith('/') or os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, 'ga_plot.png')
        else:
            # 确保输出文件的目录存在
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()  # 关闭图形释放内存


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/plot_ga.py /path/to/ga_generations.json [out.png]')
        sys.exit(1)
    
    jp = sys.argv[1]
    op = sys.argv[2] if len(sys.argv) > 2 else None
    plot_ga(json_path=jp, out_path=op)
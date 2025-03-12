import numpy as np
from typing import List

def calculate_window_statistics(fitness_window: List[float], window_size: int = 100) -> dict:
    """Calculate statistics for current fitness window using vectorized operations"""
    window = fitness_window[-window_size:]
    if not window:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "best": 0.0, "worst": 0.0}
    
    arr = np.array(window)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "best": float(np.max(arr)),
        "worst": float(np.min(arr))
    }

def update_fitness_window(fitness_window: List[float], new_fitnesses: List[float], window_size: int) -> List[float]:
    """Update sliding window efficiently using deque-like behavior"""
    combined = (fitness_window[-window_size:] if fitness_window else []) + new_fitnesses
    return combined[-window_size:]

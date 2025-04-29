import numpy as np
from typing import Callable, Dict, Any, Tuple
from itertools import product

def grid_search(param_grid: Dict[str, list],
                backtest_func: Callable[..., Dict[str, Any]],
                data, maximize: str = 'final_equity') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    best_params = None
    best_metrics = None
    best_score = -np.inf
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        metrics = backtest_func(**params, data=data)
        score = metrics.get(maximize, 0)
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    return best_params, best_metrics

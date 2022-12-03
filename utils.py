import torch
import numpy as np
import random
import logging

log = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info(f'Global seed set to {seed}')
    
    return seed


def load_task(task_cfg):
    if task_cfg['name'] == 'qap':
        from benchmarks import QAPProblem
        return QAPProblem(3)
    elif task_cfg['name'].startswith('tsp'):
        from benchmarks import TSPProblem
        return TSPProblem(task_cfg['file_path'])
    elif task_cfg['name'].startswith('earth'):
        from benchmarks import EarthBenchmark
        return EarthBenchmark(task_cfg['file_path'])
    else:
        raise NotImplementedError
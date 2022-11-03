import hydra
import omegaconf
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    import torch
    import numpy as np
    import wandb
    from utils import seed_everything, load_task
    from algorithms import BO
    
    if cfg['seed'] is not None:
        seed_everything(cfg['seed'])
        
    alg_cfg = cfg['algorithm']
    task_cfg = cfg['task']
    wandb.init(
        project=cfg['project'],
        name=f"111",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    log.info(f'device: {device}')
    
    func = load_task(task_cfg['name'])
    dims = func.dims
    f = lambda x: -func(x)
    alg = hydra.utils.instantiate(alg_cfg['model'], dims=dims, lb=np.zeros(dims), ub=np.full(dims, dims-1))
    print(alg.init_sampler_type)
    # alg = BO(dims=dims, lb=0, ub=dims-1)
    
    log.info(f'func: {f}, alg: {alg}, dims: {dims}')
    
    xs = []
    ys = []
    x_best = []
    y_best = []
    for epoch in range(cfg['epochs']):
        cand = alg.ask()
        print(cand)
        y = f(cand)
        print(epoch, y)
        alg.tell(cand, y)


if __name__ == '__main__':
    main()
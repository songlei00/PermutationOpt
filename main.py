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
    import time
    from utils import seed_everything, load_task
    from algorithms import BO
    
    if cfg['seed'] is not None:
        seed_everything(cfg['seed'])
        
    # set parameters
    alg_cfg = cfg['algorithm']
    task_cfg = cfg['task']
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    wandb.init(
        project=cfg['project'],
        name='{}-{}-{}-{}'.format(task_cfg['name'], alg_cfg['name'], cfg['seed'], curr_time),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    log.info(f'device: {device}')
    
    func = load_task(task_cfg['name'])
    dims = func.dims
    f = lambda x: -func(x)
    alg = hydra.utils.instantiate(alg_cfg['model'], dims=dims, lb=np.zeros(dims), ub=np.full(dims, dims-1))
    # alg = BO(dims=dims, lb=0, ub=dims-1)
    
    log.info(f'func: {f}, alg: {alg}, dims: {dims}')
    
    # train
    xs = []
    ys = []
    x_best = None
    y_best = None
    for epoch in range(cfg['epochs']):
        cand = alg.ask()
        y = f(cand)
        alg.tell(cand, y)

        # log
        xs.append(cand)
        ys.append(y)
        if y_best is None or y > y_best:
            x_best = cand
            y_best = y
        log.info('Epoch: {}, y best: {}, x: {}, y: {}'.format(epoch, y_best, cand, y))

        # wandb log
        wandb.log({
            task_cfg['name']+'/epoch': epoch, 
            task_cfg['name']+'/y': y,
        })
        wandb.run.summary['y best'] = y


if __name__ == '__main__':
    main()
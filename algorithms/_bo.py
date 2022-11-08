import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import Kernel, MaternKernel
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP
from botorch.acquisition import ExpectedImprovement
import numpy as np
from collections import deque
from typing import List
import logging
from ._ea import EA
from ._base import BaseOptimizer
from ._fillin_strategy import PermutationRandomStrategy
from ._utils import get_init_samples, permutation_sampler

log = logging.getLogger(__name__)


class PositionKernel(Kernel):
    def forward(self, X, X2, **params):
        if X.dim() == 2:
            X = X[:, None, :]
        kernel_mat = torch.sum((X - X2)**2, axis=-1)
        log.debug('res shape: {}'.format(kernel_mat.shape))
        return torch.exp(- kernel_mat)
    

def discordant_cnt(x, y):
    assert len(x) == len(y)
    cnt = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            cnt += (x[i] < x[j]) * (y[i] > y[j]) + (x[i] > x[j]) * (y[i] < y[j])
    return cnt


class OrderKernel(Kernel):
    def forward(self, X, X2, **params):
        if X.dim() == 2:
            X = X[:, None, :]
        if X2.dim() == 3:
            assert len(X2) == 1
            X2 = X2[0]
        assert X.shape[1] == 1
        max_cnt = (X.shape[-1] * (X.shape[-1]-1)) / 2
        log.debug('Order kernel: X shape: {}, X2 shape: {}'.format(X.shape, X2.shape))
        mat = torch.zeros((len(X), len(X2)))
        for i in range(len(X)):
            x1 = X[i][0]
            for j in range(len(X2)):
                x2 = X2[j]
                log.debug('x1 shape: {}, x2 shape: {}'.format(x1.shape, x2.shape))
                mat[i][j] = (max_cnt - 2*discordant_cnt(x1, x2)) / max_cnt
        
        return mat

    
class VSKernel(Kernel):
    has_lengthscale = True
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.pos_kernel = PositionKernel()
        self.order_kernel = OrderKernel()
        self.mode = mode
        
    def forward(self, X, X2, **params):
        log.debug('X shape: {}'.format(X.shape))
        log.debug('X2 shape: {}'.format(X2.shape))

        pos_mat = self.pos_kernel(X, X2)
        order_mat = self.order_kernel(X, X2)
        assert pos_mat.shape == order_mat.shape

        return order_mat
        sum_mat = pos_mat + order_mat
        prod_mat = pos_mat * order_mat
        lam = 0.5
        mix_mat = lam * sum_mat + (1-lam) * prod_mat
        if self.mode == 'sum':
            return sum_mat
        elif self.mode == 'prod':
            return prod_mat
        elif self.mode == 'mix':
            return mix_mat
        else:
            raise NotImplementedError
        

class BO(BaseOptimizer):
    def __init__(
        self, dims, lb, ub, active_dims=3, n_init=10, batch_size=1, init_sampler_type='permutation', acqf_init_sampler_type='permutation', 
        acqf_type='EI', acqf_opt_type='random', kernel_type='vs', fillin_type='random', device='cpu'
    ):
        self.dims = dims
        self.active_dims = active_dims
        self.lb = np.ones(self.dims) * lb
        self.ub = np.ones(self.dims) * ub
        self.n_init = n_init
        self.batch_size = batch_size
        self.init_sampler_type = init_sampler_type
        self.acqf_init_sampler_type = acqf_init_sampler_type
        self.acqf_type = acqf_type
        self.acqf_opt_type = acqf_opt_type
        self.kernel_type = kernel_type
        fillin_strategy_factory = {
            'random': PermutationRandomStrategy(dims, lb, ub),
        }
        self.fillin_strategy = fillin_strategy_factory[fillin_type]
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        log.info('Device: {}'.format(self.device))

        self.cache_X = deque()
        self.train_X = []
        self.train_Y = []
        
    def _init_samples(self, init_sampler_type, n) -> List[np.array]:
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [points[i] for i in range(len(points))]
        return points
        
    def _get_kernel(self, kernel_type):
        if kernel_type == 'matern':
            kernel = MaternKernel()
        elif kernel_type == 'vs':
            kernel = VSKernel('mix')
        else:
            raise NotImplementedError
        return kernel
        
    def _get_acqf(self, acqf_type, model):
        if acqf_type == 'EI':
            AF = ExpectedImprovement(model, best_f=np.max(self.train_Y).item()).to(self.device)
        else:
            raise NotImplementedError
        return AF
    
    def _select(self, dims, active_dims):
        idx = np.random.choice(range(dims), active_dims, replace=False)
        return idx
    
    def _get_subset(self, train_X, idx):
        # return the position of idx in train_X
        subset_X = torch.zeros((len(train_X), len(idx)))
        for i, j in enumerate(idx):
            pos = torch.where(train_X == j)
            subset_X[:, i] = pos[1]
        return subset_X
        
    def _init_model(self, train_X: Tensor, train_Y: Tensor):
        Y_var = torch.full_like(train_Y, 0.01)
        kernel = self._get_kernel(self.kernel_type)
        model = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        return mll, model
    
    def _optimize_acqf_random(self, dims, AF, lb, ub, n=1):
        cand_X = permutation_sampler(1024, dims, list(range(self.dims)))
        cand_X = torch.from_numpy(cand_X).float().to(self.device)
        cand_Y = torch.cat([AF(X_) for X_ in cand_X.split(1)]).reshape(-1)
        # cand_Y = AF(cand_X.unsqueeze(1))
        indices = torch.argsort(cand_Y)[-n: ]
        proposed_X, proposed_Y = cand_X[indices], cand_Y[indices]
        return proposed_X, proposed_Y

    def _optimize_acqf_ea(self, dims, AF, lb, ub, n=1):
        ea_alg = EA(dims, lb, ub, pop_size=10, init_sampler_type='permutation', mutation_type='swap', crossover_type='order')
        
        for _ in range(100):
            cands = ea_alg.ask()
            cands_tensor = [torch.from_numpy(cand) for cand in cands]
            cands_y_tensor = [AF(cand.unsqueeze(0)) for cand in cands_tensor]
            cands_y = [y.cpu().detach().numpy().item() for y in cands_y_tensor]
            ea_alg.tell(cands, cands_y)
        
        indices = np.argsort(ea_alg.fitness)[-n: ]
        proposed_X = [torch.from_numpy(ea_alg.population[idx]) for idx in indices]
        proposed_Y = [ea_alg.fitness[idx] for idx in indices]
        return proposed_X, proposed_Y
    
    def _optimize_acqf(self, dims, AF, lb, ub, n=1):
        if self.acqf_opt_type == 'random':
            proposed_X, proposed_Y = self._optimize_acqf_random(dims, AF, lb, ub, n)
        elif self.acqf_opt_type == 'ea':
            proposed_X, proposed_Y = self._optimize_acqf_ea(dims, AF, lb, ub, n)
        else:
            raise NotImplementedError
        return proposed_X, proposed_Y
    
    def ask(self) -> np.array:
        # init
        if len(self.cache_X) + len(self.train_X) < self.n_init:
            points = self._init_samples(self.init_sampler_type, self.n_init)
            self.cache_X.extend(points)
            
        # unevaluated points
        if len(self.cache_X) > 0:
            return [self.cache_X.popleft()]
        
        # prepare train data
        train_X_tensor = torch.vstack(self.train_X).float().to(self.device)
        idx = self._select(self.dims, self.active_dims)
        subset_X = self._get_subset(train_X_tensor, idx)
        train_Y_tensor = torch.from_numpy(np.vstack(self.train_Y)).to(self.device)
        train_Y_tensor = (train_Y_tensor - train_Y_tensor.mean()) / train_Y_tensor.std()

        # train model
        mll, model = self._init_model(subset_X, train_Y_tensor)
        fit_gpytorch_model(mll)
        
        # optimize acquisition function
        AF = self._get_acqf(self.acqf_type, model)
        proposed_X, _ = self._optimize_acqf(self.active_dims, AF, self.lb, self.ub, 1)
        assert len(proposed_X) == 1
        proposed_X = [proposed_X[i].cpu().detach().numpy() for i in range(len(proposed_X))]

        # fill in
        new_X = []
        for i in range(len(proposed_X)):
            fixed_vars = {j: pos for j, pos in zip(idx, proposed_X[i])}
            log.debug('fixed variables: {}'.format(fixed_vars))
            new_x = self.fillin_strategy.fillin(fixed_vars)
            new_x = new_x.astype(np.int)
            log.debug('new x: {}'.format(new_x))
            new_X.append(new_x)
        
        self.cache_X.extend(new_X)
        
        return [self.cache_X.popleft()]
    
    def tell(self, X: List[np.array], Y):
        X = [torch.as_tensor(x) for x in X]
        self.train_X.extend(X)
        self.train_Y.extend(Y)
        
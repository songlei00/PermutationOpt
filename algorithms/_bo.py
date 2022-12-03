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
import time
import logging
from ._ea import EA
from ._base import BaseOptimizer
from ._fillin_strategy import PermutationRandomStrategy, PermutationBestKPosStrategy
from ._utils import get_init_samples, permutation_sampler, select, get_subset, featurize

log = logging.getLogger(__name__)


class FeatureCache:
    def __init__(self):
        self.cache = dict()

    def _get_key(self, x):
        return tuple(x.tolist())

    def push(self, x):
        feature = self.get(x)
        if feature is None:
            feature = featurize(x, 'torch')
            self.cache[self._get_key(x)] = feature
        return feature

    def get(self, x):
        return self.cache.get(self._get_key(x), None)

feature_cache = FeatureCache()


class PositionKernel(Kernel):
    def forward(self, X, X2, **params):
        if X.dim() == 2:
            X = X[:, None, :]
        kernel_mat = torch.sum((X - X2)**2, axis=-1)
        # log.debug('res shape: {}'.format(kernel_mat.shape))
        return torch.exp(- kernel_mat)
    

def discordant_cnt(x, y):
    assert len(x) == len(y)
    cnt = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            cnt += (x[i] < x[j]) * (y[i] > y[j]) + (x[i] > x[j]) * (y[i] < y[j])
    return cnt


class OrderKernel(Kernel):
    has_lengthscale = True
    # def forward(self, X, X2, **params):
    #     if X.dim() == 2:
    #         X = X[:, None, :]
    #     if X2.dim() == 3:
    #         assert len(X2) == 1
    #         X2 = X2[0]
    #     assert X.shape[1] == 1
    #     max_cnt = (X.shape[-1] * (X.shape[-1]-1)) / 2
    #     log.debug('Order kernel: X shape: {}, X2 shape: {}'.format(X.shape, X2.shape))
    #     mat = torch.zeros((len(X), len(X2)))
    #     log.debug('mat shape: {}'.format(mat.shape))
    #     for i in range(len(X)):
    #         x1 = X[i][0]
    #         for j in range(len(X2)):
    #             x2 = X2[j]
    #             mat[i][j] = (max_cnt - 2*discordant_cnt(x1, x2)) / max_cnt
        
    #     return mat

    def forward(self, X, X2, **params):
        if X.dim() == 2:
            X = X[:, None, :]
        if X2.dim() == 3:
            assert len(X2) == 1
            X2 = X2[0]
        assert X.shape[1] == 1
        mat = torch.zeros((len(X), len(X2))).to(X)
        for i in range(len(X)):
            x1 = X[i][0]
            x1 = feature_cache.push(x1)
            # x1 = featurize(x1)
            for j in range(len(X2)):
                x2 = X2[j]
                x2 = feature_cache.push(x2)
                # x2 = featurize(x2)
                mat[i][j] = torch.sum((x1 - x2)**2)
        
        return torch.exp(- self.lengthscale * mat)

    
class VSKernel(Kernel):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.pos_kernel = PositionKernel()
        self.order_kernel = OrderKernel()
        self.mode = mode
        
    def forward(self, X, X2, **params):
        # log.debug('X shape: {}'.format(X.shape))
        # log.debug('X2 shape: {}'.format(X2.shape))

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
        self, dims, lb, ub, active_dims=3, n_init=10, batch_size=1, init_sampler_type='permutation', 
        acqf_init_sampler_type='permutation', acqf_type='EI', acqf_opt_type='random', kernel_type='vs', 
        fillin_type='random', device='cpu'
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
            'best_pos': PermutationBestKPosStrategy(dims, lb, ub, 10),
        }
        self.fillin_strategy = fillin_strategy_factory[fillin_type]
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        log.info('Device: {}'.format(self.device))

        self.cache_X = deque()
        self.train_X = []
        self.train_Y = []
        
    def _init_samples(self, init_sampler_type, n) -> List[np.ndarray]:
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
        
    def _get_acqf(self, acqf_type, model, train_X: Tensor, train_Y: Tensor):
        if acqf_type == 'EI':
            AF = ExpectedImprovement(model, best_f=train_Y.max().item()).to(self.device)
        else:
            raise NotImplementedError
        return AF
        
    def _init_model(self, train_X: Tensor, train_Y: Tensor):
        Y_var = torch.full_like(train_Y, 0.01).to(self.device)
        kernel = self._get_kernel(self.kernel_type).to(self.device)
        model = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)
        return mll, model
    
    def _optimize_acqf_random(self, dims, AF, lb, ub, n=1):
        cand_X = permutation_sampler(1024, dims, list(range(self.dims)))
        cand_X = torch.from_numpy(cand_X).float().to(self.device)
        cand_Y = torch.cat([AF(X_) for X_ in cand_X.split(1)]).reshape(-1)
        # cand_Y = AF(cand_X.unsqueeze(1))
        indices = torch.argsort(cand_Y)[-n: ]
        proposed_X, proposed_Y = cand_X[indices], cand_Y[indices]
        return proposed_X, proposed_Y

    def _optimize_acqf_local_search(self, dims, AF, lb, ub, n=1, n_restart = 10):
        assert n == 1
        init_X, init_Y = self._optimize_acqf_random(dims, AF, lb, ub, n_restart)
        best_cand = None
        best_vals = None
        
        for i in range(n_restart):
            x = init_X[i].cpu()
            vals = init_Y[i].cpu()
            while True:
                all_cands = []
                all_vals = []
                # generate neighbors
                for i in range(dims):
                    for j in range(i+1, dims):
                        next_x = x.clone()
                        tmp = next_x[i].item()
                        next_x[i] = next_x[j].item()
                        next_x[j] = tmp
                        all_cands.append(next_x)
                        with torch.no_grad():
                            all_vals.append(AF(next_x.unsqueeze(0).to(self.device)))
                idx = torch.argmax(torch.cat(all_vals))
                if all_vals[idx] > vals:
                    x = all_cands[idx]
                    vals = all_vals[idx]
                else:
                    break
            if best_vals is None or vals > best_vals:
                best_cand = x
                best_vals = vals
            # log.debug('-----------------------')
            # log.debug('x: {}'.format(x))
            # log.debug('vals: {}'.format(vals))
            # log.debug('-----------------------')
            
        return [best_cand], [best_vals]

    def _optimize_acqf_ea(self, dims, AF, lb, ub, n=1):
        ea_alg = EA(dims, lb, ub, pop_size=10, init_sampler_type='permutation', mutation_type='swap', crossover_type='order')
        
        for _ in range(500):
            cands = ea_alg.ask()
            cands_tensor = [torch.from_numpy(cand).float() for cand in cands]
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
        elif self.acqf_opt_type == 'ls':
            proposed_X, proposed_Y = self._optimize_acqf_local_search(dims, AF, lb, ub, n)
        elif self.acqf_opt_type == 'ea':
            proposed_X, proposed_Y = self._optimize_acqf_ea(dims, AF, lb, ub, n)
        else:
            raise NotImplementedError
        return proposed_X, proposed_Y
    
    def ask(self) -> List[np.ndarray]:
        # init
        if len(self.cache_X) + len(self.train_X) < self.n_init:
            points = self._init_samples(self.init_sampler_type, self.n_init)
            self.cache_X.extend(points)
            
        # unevaluated points
        if len(self.cache_X) > 0:
            return [self.cache_X.popleft()]
        
        # prepare train data
        train_X_tensor = torch.vstack(self.train_X).float()
        idx = select(self.dims, self.active_dims)
        log.debug('selected idx: {}'.format(idx))
        subset_X = get_subset(train_X_tensor, idx).to(self.device)
        train_Y_tensor = torch.from_numpy(np.vstack(self.train_Y)).to(self.device)
        train_Y_tensor = (train_Y_tensor - train_Y_tensor.mean()) / (train_Y_tensor.std() + 1e-6)

        # train model
        st = time.time()
        mll, model = self._init_model(subset_X, train_Y_tensor)
        fit_gpytorch_model(mll)

        log.info('Train time: {}'.format(time.time() - st))
        st = time.time()
        
        # optimize acquisition function
        AF = self._get_acqf(self.acqf_type, model, subset_X, train_Y_tensor)
        proposed_X, _ = self._optimize_acqf(self.active_dims, AF, self.lb, self.ub, 1)
        log.debug('Proposed X: {}'.format(proposed_X))
        assert len(proposed_X) == 1
        proposed_X = [proposed_X[i].cpu().detach().numpy() for i in range(len(proposed_X))]

        log.info('Optimize acquisition function time: {}'.format(time.time() - st))
        st = time.time()

        # fill in
        new_X = []
        for i in range(len(proposed_X)):
            fixed_vars = {j: pos for j, pos in zip(idx, proposed_X[i])}
            log.debug('fixed variables: {}'.format(fixed_vars))
            new_x = self.fillin_strategy.fillin(fixed_vars)
            log.debug('new x: {}'.format(new_x))
            new_X.append(new_x)
        
        self.cache_X.extend(new_X)
        
        return [self.cache_X.popleft()]
    
    def tell(self, X: List[np.ndarray], Y):
        X = [torch.as_tensor(x) for x in X]
        self.train_X.extend(X)
        self.train_Y.extend(Y)
        for x, y in zip(X, Y):
            if isinstance(x, Tensor):
                x = x.cpu().detach().numpy()
            self.fillin_strategy.update(x.reshape(1, -1), y)
        
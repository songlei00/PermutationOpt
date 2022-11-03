from hashlib import new
import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import Kernel, MaternKernel
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP
from botorch.acquisition import ExpectedImprovement
import numpy as np
from collections import deque
from ._base import BaseOptimizer
from ._fillin_strategy import PermutationRandomStrategy
from ._utils import get_init_samples, permutation_sampler


class PositionKernel(Kernel):
    has_lengthscale = True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, X, X2, **params):
        # print('length scale shape', self.lengthscale.shape)
        # print('X shape', X.shape)
        # print('X2 shape', X2.shape)
        if X.dim() == 2:
            X = X[:, None, :]
        elif X.dim() == 3 and X2.dim() == 3:
            pass
        else:
            assert 0
        kernel_mat = torch.sum((X - X2)**2, axis=-1)
        # print('res shape', kernel_mat.shape)
        return torch.exp(- kernel_mat)
    
    
class OrderKernel(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, X, X2, **params):
        return 1

    
class VSKernel(Kernel):
    has_lengthscale = True
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.pos_kernel = PositionKernel()
        self.order_kernel = OrderKernel()
        self.mode = mode
        
    def forward(self, X, X2, **params):
        # print('X shape', X.shape)
        # print('X2 shape', X2.shape)
        
        pos_sim = self.pos_kernel(X, X2)
        # order_sim = self.order_kernel(X, X2)
        
        if self.mode == 'sum':
            pass
        elif self.mode == 'prod':
            pass
        else:
            raise NotImplementedError

        return pos_sim
        

class BO(BaseOptimizer):
    def __init__(self, dims, lb, ub, n_init=10, init_sampler_type='sobel', acqf_init_sampler_type='sobel', acqf_type='EI', acqf_opt_type='random', kernel_type='matern'):
        self.dims = dims
        self.active_dims = 3
        self.lb = np.ones(self.dims) * lb
        self.ub = np.ones(self.dims) * ub
        self.n_init = n_init
        self.init_sampler_type = init_sampler_type
        self.acqf_init_sampler_type = acqf_init_sampler_type
        self.acqf_type = acqf_type
        self.acqf_opt_type = acqf_opt_type
        self.kernel_type = kernel_type
        self.fillin_strategy = PermutationRandomStrategy(self.dims)
        self.cache_X = deque()
        self.train_X = []
        self.train_Y = []
        
    def _init_samples(self, init_sampler_type, n):
        points = get_init_samples(init_sampler_type, n, self.dims, self.lb, self.ub)
        points = [torch.from_numpy(points[i]) for i in range(len(points))]
        return points
        
    def _get_kernel(self, kernel_type):
        if kernel_type == 'matern':
            kernel = MaternKernel()
        elif kernel_type == 'vs':
            kernel = VSKernel('sum')
        else:
            raise NotImplementedError
        return kernel
        
    def _get_acqf(self, acqf_type, model):
        if acqf_type == 'EI':
            AF = ExpectedImprovement(model, best_f=np.max(self.train_Y).item())
        else:
            raise NotImplementedError
        return AF
    
    def _select(self, dims, active_dims):
        idx = np.random.choice(range(dims), active_dims, replace=False)
        return idx
    
    def _get_subset(self, train_X, idx):
        subset_X = torch.zeros((len(train_X), len(idx)))
        for i, j in enumerate(idx):
            pos = torch.where(train_X == j)
            subset_X[:, i] = pos[1]
        return subset_X
        
    def _init_model(self, train_X: Tensor, train_Y: Tensor):
        Y_var = torch.full_like(train_Y, 0.01)
        
        kernel = self._get_kernel(self.kernel_type)
        model = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        return mll, model
    
    def _optimize_acqf_random(self, dims, AF, lb, ub, n=1, idx=None):
        cand_X = permutation_sampler(1024, dims, list(range(self.dims)))
        cand_X = torch.from_numpy(np.vstack(cand_X)).float()
        cand_Y = torch.cat([AF(X_) for X_ in cand_X.split(1)]).reshape(-1)
        # cand_Y = AF(cand_X.unsqueeze(1))
        indices = torch.argsort(cand_Y)[-n: ]
        proposed_X, proposed_Y = cand_X[indices], cand_Y[indices]
        return proposed_X, proposed_Y

    def _optimize_acqf_ea_permutation(self, dims, AF, lb, ub, n=1, idx=None):
        pass
    
    def _optimize_acqf(self, dims, AF, lb, ub, n=1, idx=None):
        if self.acqf_opt_type == 'random':
            proposed_X, proposed_Y = self._optimize_acqf_random(dims, AF, lb, ub, 1, idx)
        else:
            raise NotImplementedError
        return proposed_X, proposed_Y
    
    def ask(self):
        if len(self.cache_X) + len(self.train_X) < self.n_init:
            points = self._init_samples(self.init_sampler_type, self.n_init)
            self.cache_X.extend(points)
            
        if len(self.cache_X) > 0:
            return self.cache_X.popleft()
        
        # prepare train data
        train_X_tensor = torch.vstack(self.train_X).float()
        idx = self._select(self.dims, self.active_dims)
        subset_X = self._get_subset(train_X_tensor, idx)
        train_Y_tensor = torch.tensor(np.vstack(self.train_Y))
        train_Y_tensor = (train_Y_tensor - train_Y_tensor.mean()) / train_Y_tensor.std()

        # train model
        mll, model = self._init_model(subset_X, train_Y_tensor)
        fit_gpytorch_model(mll)
        
        # optimize acquisition function
        AF = self._get_acqf(self.acqf_type, model)
        proposed_X, _ = self._optimize_acqf(self.active_dims, AF, self.lb, self.ub, 1, None)
        assert len(proposed_X) == 1
        proposed_X = proposed_X[0].cpu().detach().numpy()

        # fill in
        fixed_vars = {i: pos for i, pos in zip(idx, proposed_X)}
        print(fixed_vars)
        new_x = self.fillin_strategy.fillin(fixed_vars, self.lb, self.ub)
        new_x = new_x.astype(np.int)
        print(new_x)
        
        self.cache_X.append(new_x)
        
        return self.cache_X.popleft()
    
    def tell(self, x, y):
        x = torch.as_tensor(x)
        self.train_X.append(x)
        self.train_Y.append(y)
        
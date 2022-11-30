import torch
from torch.quasirandom import SobolEngine
import numpy as np


# ================== init function =================
def from_unit_cube(points, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert points.ndim  == 2
    new_points = points * (ub - lb) + lb
    return new_points


def sobel_sampler(n, dims) -> np.ndarray:
    seed = np.random.randint(int(5e5))
    sobol = SobolEngine(dims, scramble=True, seed=seed)
    points = sobol.draw(n).to(dtype=torch.float64).cpu().detach().numpy()
    return points


def lhs_sampler(n, dims) -> np.ndarray:
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points


def permutation_sampler(n, dims, choices=None) -> np.ndarray:
    if choices is None:
        choices = range(dims)
    points = [np.random.choice(choices, dims, replace=False) for _ in range(n)]
    points = np.vstack(points)
    return points


def get_init_samples(sampler_type, n, dims, lb, ub):
    if sampler_type == 'sobel':
        points = sobel_sampler(n, dims)
        points = from_unit_cube(points, lb, ub)
    elif sampler_type == 'lhs':
        points = lhs_sampler(n, dims)
        points = from_unit_cube(points, lb, ub)
    elif sampler_type == 'permutation':
        points = permutation_sampler(n, dims)
    else:
        raise NotImplementedError
    return points


# ================== init function =================
def select(dims, active_dims):
    idx = np.random.choice(range(dims), active_dims, replace=False)
    idx = np.sort(idx)
    return idx

def get_subset(train_X, idx):
    # return the position of idx in train_X
    if isinstance(train_X, np.ndarray):
        zeros_fn = np.zeros
        where_fn = np.where
    elif isinstance(train_X, torch.Tensor):
        zeros_fn = torch.zeros
        where_fn = torch.where
    subset_X = zeros_fn((len(train_X), len(idx)))
    for i, j in enumerate(idx):
        pos = where_fn(train_X == j)
        subset_X[:, i] = pos[1]
    return subset_X


if __name__ == '__main__':
    n, dims = 10, 5
    points = sobel_sampler(n, dims)
    print(type(points))
    print(points)
    points = lhs_sampler(n, dims)
    print(type(points))
    print(points)
    points = permutation_sampler(n, dims)
    print(type(points))
    print(points)
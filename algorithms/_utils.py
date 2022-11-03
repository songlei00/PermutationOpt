import torch
from torch.quasirandom import SobolEngine
import numpy as np


def from_unit_cube(points, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert points.ndim  == 2
    new_points = points * (ub - lb) + lb
    return new_points


def sobel_sampler(n, dims) -> np.array:
    seed = np.random.randint(int(5e5))
    sobol = SobolEngine(dims, scramble=True, seed=seed)
    points = sobol.draw(n).to(dtype=torch.float64).cpu().detach().numpy()
    return points


def lhs_sampler(n, dims) -> np.array:
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points


def permutation_sampler(n, dims, choices=None) -> np.array:
    if choices is None:
        points = [np.random.permutation(dims) for _ in range(n)]
    else:
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
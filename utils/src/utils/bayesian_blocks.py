import math
from typing import SupportsFloat
from collections.abc import Iterable, Sized

import torch
from tqdm import tqdm

def _prior(
    p0: float, 
    gamma: float | None, 
    N : int, 
    **kwargs
):
    if gamma is not None:
        return torch.full(N, -math.log(gamma), **kwargs)
    else:
        # eq. 21 from Scargle 2012
        return torch.arange(1, N+1, **kwargs).pow_(0.478).div_(73.53 * p0).log_().add(4)
        #return 4 - torch.log(73.53 * p0 * (torch.arange(1, N+1)**-0.478))

def _process_data(
    data : Iterable, 
    weights : Iterable | None,
    /,
    ranges : tuple[SupportsFloat | None, SupportsFloat | None] | None = None, 
    undersample : float | int | None = None,
    generator : torch.Generator | None = None,
    device : str = "cpu",
    dtype : torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor] :
    data = torch.as_tensor(data, device = device, dtype = dtype,)
    assert data.ndim == 1

    weights = torch.as_tensor(weights, device = device, dtype = dtype,) \
                    if weights is not None \
                    else torch.ones_like(data)
    assert data.shape == weights.shape

    if ranges is not None:
        #indices = torch.arange(0,data.numel(), **kwargs) 
        data_min, data_max = ranges
        mask = torch.full_like(data, True, dtype=torch.bool)
        if data_min is not None:
            mask = mask.logical_and_(data.gt(data_min))
        if data_max is not None:
            mask = mask.logical_and_(data.lt(data_max))
        data = data[mask]
        weights = weights[mask]

    if undersample is not None:
        input_size= data.numel()
        sample_size = undersample if isinstance(undersample, int) \
                                     else math.floor(input_size*undersample)
        assert sample_size <= input_size 
        indices = torch.randperm(input_size, generator=generator)[:sample_size]
        data = data[indices]
        weights = weights[indices]

    group_data, group_indices = torch.unique(
        data, sorted=True, return_inverse=True,
    )
    group_weights = torch.zeros_like(group_data)
    
    return group_data, group_weights.index_add_(0, group_indices, weights)

def bayesian_blocks(
    data: Iterable,
    /, *,
    weights: Iterable | None = None,
    p0: float = 0.05,
    gamma : float | None = None,
    **kwargs,
) -> torch.Tensor:
    # validate input data
    data, weights = _process_data(data, weights, **kwargs)
    edges = torch.cat([data[:1], 0.5 * (data[1:] + data[:-1]), data[-1:]])

    # arrays to store the best configuration
    best = torch.zeros_like(weights)
    last = torch.zeros_like(weights, dtype=torch.long)
    
    N = weights.size(0)
    prior_calc = _prior(
        p0, gamma, N, 
        device=weights.device, dtype=weights.dtype,
    )
    # -----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    # -----------------------------------------------------------------
    # last = core_loop(N, edges, weights, fitfunc, best, last)

    csum = torch.cumsum(weights, 0)
    subtracted_csum = weights.sub_(csum)
    
    for R in tqdm(range(N)):
        T_k = edges[R + 1] - edges[: R + 1]
        N_k = subtracted_csum[: R+1] + csum[R] 

        A_R = N_k.xlogy_(N_k / T_k).sub_(prior_calc[R])
        A_R[1:] += best[:R]

        i_max = torch.argmax(A_R)
        last[R] = i_max
        best[R] = A_R[i_max]

    # -----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    # -----------------------------------------------------------------
    change_points = torch.zeros_like(last)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    return edges[change_points[i_cp:]].cpu()


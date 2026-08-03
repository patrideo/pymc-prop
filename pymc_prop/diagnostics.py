import numpy as np
from typing import Callable, Optional, Tuple


def _get_bandwidth(sq_dists: np.ndarray, N: int, lengthscale: Optional[float] = None) -> float:
    """Helper to compute bandwidth via median heuristic or manual choice."""
    if lengthscale is None:
        off_diag = sq_dists[np.triu_indices(N, k=1)]
        median_sq = np.median(off_diag) if off_diag.size else 0.0
        # Adapt bandwidth to particle count
        bandwidth = median_sq / np.log(N) if median_sq > 0 else 1.0
        return max(bandwidth, 1e-12)
    return max(lengthscale ** 2, 1e-12)

def rbf_kernel(sq_dists: np.ndarray, d: int, lengthscale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes components for the Gaussian RBF Kernel: K(x,y) = exp(-||x-y||^2 / h)
    Returns: (K, cross_coeff, trace_term)
    """
    N = sq_dists.shape[0]
    h = _get_bandwidth(sq_dists, N, lengthscale)
    
    K = np.exp(-sq_dists / h)
    
    # -2 * f'(r^2)
    cross_coeff = (2.0 / h) * K 
    # -2d * f'(r^2) - 4r^2 * f''(r^2)
    trace_term = K * ((2.0 * d / h) - (4.0 / (h ** 2)) * sq_dists) 
    
    return K, cross_coeff, trace_term

def imq_kernel(sq_dists: np.ndarray, d: int, lengthscale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes components for the Inverse Multi-Quadric Kernel: K(x,y) = (1 + ||x-y||^2 / h)^{-1/2}
    Returns: (K, cross_coeff, trace_term)
    """
    N = sq_dists.shape[0]
    h = _get_bandwidth(sq_dists, N, lengthscale)
    
    K = (1.0 + sq_dists / h) ** (-0.5)
    
    # Precompute powers of K to avoid repeated expensive operations
    K2 = K*K
    K3 = K2*K
    K5 = K2*K3
    
    # -2 * f'(r^2)
    cross_coeff = (1.0 / h) * K3 
    # -2d * f'(r^2) - 4r^2 * f''(r^2)
    trace_term = (d / h) * K3 - (3.0 / (h ** 2)) * K5 * sq_dists 
    
    return K, cross_coeff, trace_term

def compute_kgd(
    particles: np.ndarray, 
    forces: np.ndarray, 
    kernel_fn: Callable = rbf_kernel,
    lengthscale: Optional[float] = None
) -> float:
    """
    Computes the full 4-term Kernel Gradient Discrepancy (KGD^2) estimator 
    using a generalized kernel function and Stein's Identity.
    """
    N, d = particles.shape
    if N <= 1:
        return 0.0
        
    # 1. Pairwise differences and squared distances
    # diffs shape: (N, N, d) -> diffs[j, l] = x_j - x_l
    diffs = particles[:, None, :] - particles[None, :, :]  
    sq_dists = np.sum(diffs ** 2, axis=-1)                
    
    # 2. Get matrix components from the generalized kernel function
    K, cross_coeff, trace_term = kernel_fn(sq_dists, d, lengthscale)
    
    # Base Interaction b(x)^T K(x,y) b(y)
    term1 = (forces @ forces.T) * K 
    
    # Terms 2 & 3: The Cross Terms
    force_diffs = forces[:, None, :] - forces[None, :, :]
    cross_dot = np.sum(force_diffs * diffs, axis=-1)
    
    term23 = cross_coeff * cross_dot 
    
    #  Combine into the full Stein Kernel matrix (H)
    H = term1 - term23 + trace_term
    
    # Exclude diagonal (j=l) to compute the unbiased U-statistic
    np.fill_diagonal(H, 0.0)
    
    # Final KGD^2 Expectation, normalized by N(N-1)
    kgd_sq = np.sum(H) / (N * (N - 1))
    kgd_val = float(max(kgd_sq, 0.0))
    
    return float(np.sqrt(kgd_val))
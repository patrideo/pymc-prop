"""Simulation loop for PrO particles."""

from __future__ import annotations

from typing import List

import numpy as np

from pymc_prop.compile import compile_batched_prior_grad
from pymc_prop.fuse import (
    FUSE_GRADIENT_ENERGY_STAT,
    FUSE_HALF_STEP_DISTANCE_SQ_STAT,
    FUSE_STEP_SIZE_STAT,
    FuseState,
    fuse_adaptive_step,
    fuse_initial_step,
)
from pymc_prop.particles import initialize_particles, time_step, scaled_drift
from pymc_prop.points import PointMapper
from pymc_prop.scoring import LogScore, ScoringRule
from pymc_prop.diagnostics import compute_kgd_squared, get_bandwidth, imq_kernel


def run_sampler(
    model,
    mapper: PointMapper,
    scoring_rule: ScoringRule,
    n_particles: int,
    n_steps: int,
    tune: int,
    step_size: float | None,
    learning_rate: float,
    random_seed: int | None,
    r_eps: float = 1e-5,
    fuse_diagnostics: dict[str, list[float]] | None = None,
    flow_stats: dict[str, list[float]] = None,
    kgd_interval: int | None = None,
    bandwidth: float | None = None,
    biased: bool = False,
) -> np.ndarray:
    """Run the PrO particle simulation loop.

    Returns retained particle arrays with shape ``(n_steps, n_particles,
    n_params)``. The loop runs ``tune + n_steps`` Euler-Maruyama steps; each draw after
    warmup is an empirical particle measure at a retained simulation step.

    When ``step_size`` is ``None``, the tuning-free FUSE schedule (Sharrock &
    Nemeth 2025) adapts ``η_t`` from raw and scaled drift fields; ``r_eps`` is
    the schedule floor. FUSE state persists across the full ``tune + n_steps``
    loop (no reset at the tune boundary).
    """
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if step_size is None:
        if r_eps <= 0:
            raise ValueError("r_eps must be positive when step_size is None (FUSE).")
    elif step_size <= 0:
        raise ValueError("step_size must be positive.")

    rng = np.random.default_rng(random_seed)

    # Sec. 5: particles in unconstrained value_vars space
    particles = initialize_particles(model, mapper, n_particles, rng)

    # LogScore: compile_drift once (fused wgf+prior). Do not also call compile_wgf --
    # that rebuilds the same fused graph and discards prior_grad.
    wgf_fn = None
    batched_prior_grad_fn = None
    drift_fn = None
    if isinstance(scoring_rule, LogScore):
        drift_fn = scoring_rule.compile_drift(model, mapper)
    else:
        wgf_fn = scoring_rule.compile_wgf(model, mapper)
        batched_prior_grad_fn = compile_batched_prior_grad(mapper, model)

    use_fuse = step_size is None
    # FUSE Sec. 5.1.1: None until initial schedule step (t=0); then mutable state
    fuse_state: FuseState | None = None

    retained: List[np.ndarray] = []

    initial_diffs = particles[:, None, :] - particles[None, :, :]
    initial_sq_dists = np.sum(initial_diffs**2, axis=-1)
    bandwidth = get_bandwidth(initial_sq_dists, bandwidth)

    def evaluate_gradients(current_particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate both gradient components at one particle cloud."""
        if drift_fn is not None:
            return drift_fn(current_particles)

        assert wgf_fn is not None
        assert batched_prior_grad_fn is not None
        current_wgf_grad = wgf_fn(current_particles)
        current_prior_grad = np.asarray(
            batched_prior_grad_fn(current_particles), dtype=float
        )
        return current_wgf_grad, current_prior_grad

    total_steps = tune + n_steps
    wgf_grad, prior_grad = evaluate_gradients(particles)

    for step in range(total_steps):
        if use_fuse:
            if fuse_state is None:
                # t = 0: η_0 = r_ε, freeze reference half-step x_{1/2}
                step_size, fuse_state, diag = fuse_initial_step(
                    particles,
                    wgf_grad,
                    prior_grad,
                    learning_rate,
                    r_eps,
                )
            else:
                # t ≥ 1: η_t = r̄_t / sqrt(G_t), update r̄_t from half-step distances
                step_size, fuse_state, diag = fuse_adaptive_step(
                    particles,
                    wgf_grad,
                    prior_grad,
                    learning_rate,
                    fuse_state,
                    r_eps,
                )
            if fuse_diagnostics is not None and step >= tune:
                fuse_diagnostics.setdefault(FUSE_GRADIENT_ENERGY_STAT, []).append(diag.gradient_energy)
                fuse_diagnostics.setdefault(FUSE_HALF_STEP_DISTANCE_SQ_STAT, []).append(diag.half_step_distance_sq)
                fuse_diagnostics.setdefault(FUSE_STEP_SIZE_STAT, []).append(diag.step_size)

        particles = time_step(
            particles, prior_grad, wgf_grad, step_size, learning_rate, rng
        )

        kgd_due = (
            flow_stats is not None
            and kgd_interval is not None
            and step >= tune
            and (step - tune) % kgd_interval == 0
        )
        has_next_step = step + 1 < total_steps

        # Evaluate at x[t+1]. These gradients give an aligned KGD for the
        # updated cloud and are cached for the next Euler-Maruyama step.
        if has_next_step or kgd_due:
            next_wgf_grad, next_prior_grad = evaluate_gradients(particles)

        if flow_stats is not None and kgd_interval is not None and step>= tune:
            if kgd_due:
                potential_gradient = scaled_drift(
                    next_wgf_grad, next_prior_grad, learning_rate
                )
                score = -potential_gradient
                kgd_squared = compute_kgd_squared(
                    particles,
                    score,
                    kernel_fn=imq_kernel,
                    bandwidth=bandwidth,
                    biased=biased,
                )
            else:
                kgd_squared = np.nan

            flow_stats.setdefault("kgd_squared", []).append(kgd_squared)

        if step >= tune:
            retained.append(particles.copy())

        if has_next_step:
            wgf_grad, prior_grad = next_wgf_grad, next_prior_grad

    if retained:
        return np.stack(retained, axis=0)
    return np.empty((0, n_particles, particles.shape[1]))

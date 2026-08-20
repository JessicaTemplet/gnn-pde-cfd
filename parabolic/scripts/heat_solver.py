"""
Reference "simulation" for the parabolic stage: 1D heat equation

    u_t = nu * u_xx,   x in [0, L) periodic

solved exactly with a Fourier spectral method. Unlike the hyperbolic
Burgers' solver, there is no nonlinear term and no stability constraint on
dt -- diffusion is unconditionally stable with an exponential integrating
factor, and since the equation is linear the spectral solution is exact (not
just a high-order approximation). The truncation error is purely from
representing the initial condition in a finite Fourier basis.

Each Fourier mode k evolves as:

    u_hat(k, t) = u_hat(k, 0) * exp(-nu * (2*pi*k/L)^2 * t)

High modes decay faster than low modes (proportional to k^2), so smooth
initial conditions remain smooth but structure is gradually erased from small
to large scales. This is the defining character of parabolic equations and
the pedagogical contrast with hyperbolic: diffusion damps errors rather than
advecting them, which is why a naive one-step trained GNN is expected to be
rollout-stable here in a way it was not for Burgers'.

viscosity nu=0.01 was chosen so that:
  - mode k=1 (wavelength L) decays by factor exp(-nu*(2*pi)^2 * 0.5) ~= 0.82
    by the end of a trajectory -- still visible, the solution is not fully
    relaxed to zero.
  - mode k=4 decays by factor ~= 0.04 -- essentially gone, giving interesting
    multi-scale structure in the first portion of each trajectory.
  - no aliasing issues at nx=64: the highest mode (k=32) decays by factor
    ~= 3e-54 within the first timestep, so the grid is more than fine enough.
"""
import numpy as np


def make_wavenumbers(nx: int, L: float) -> np.ndarray:
    """Wavenumbers for rfft layout: k = 2*pi*n/L for n=0,1,...,nx//2."""
    return 2 * np.pi * np.fft.rfftfreq(nx, d=L / nx)


def solve_heat(u0: np.ndarray, L: float, nu: float, dt: float, n_steps: int,
               save_every: int = 1) -> np.ndarray:
    """Roll out the heat equation from initial condition u0 (shape (nx,)).

    Uses the exact spectral propagator: multiply each Fourier mode by the
    appropriate exponential decay factor at each timestep. Returns trajectory
    of shape (n_saved + 1, nx), where n_saved = n_steps // save_every.

    Args:
        u0: initial condition, shape (nx,), real-valued
        L: domain length (periodic)
        nu: diffusion coefficient
        dt: timestep (no stability constraint -- any dt works)
        n_steps: total number of timesteps to integrate
        save_every: save a snapshot every this many steps
    """
    nx = u0.shape[0]
    k = make_wavenumbers(nx, L)
    # precompute decay factor per mode per dt: exp(-nu * k^2 * dt)
    decay = np.exp(-nu * k**2 * dt)

    u_hat = np.fft.rfft(u0)
    traj = [u0.copy().astype(np.float32)]
    for step in range(n_steps):
        u_hat = u_hat * decay
        if (step + 1) % save_every == 0:
            u = np.fft.irfft(u_hat, n=nx).astype(np.float32)
            traj.append(u)
    return np.array(traj)   # (n_saved + 1, nx)


def random_initial_condition(nx: int, L: float, rng: np.random.Generator,
                              n_modes: int = 6, decay: float = 1.3) -> np.ndarray:
    """Smooth periodic IC as a random truncated Fourier series.

    Same recipe as the hyperbolic stage: amplitude decays with mode number
    so energy is concentrated at large scales, giving smooth initial
    conditions that evolve into interesting multi-scale transients under
    diffusion. The amplitude scaling and normalization are identical to the
    hyperbolic ICs so the two stages are directly comparable.
    """
    x = np.linspace(0.0, L, nx, endpoint=False)
    u0 = np.zeros(nx)
    for j in range(1, n_modes + 1):
        amp = rng.uniform(-1.0, 1.0) / j**decay
        phase = rng.uniform(0, 2 * np.pi)
        u0 += amp * np.sin(2 * np.pi * j * x / L + phase)
    scale = rng.uniform(1.5, 3.0)
    u0 = scale * u0 / (np.abs(u0).max() + 1e-8)
    return u0.astype(np.float32)

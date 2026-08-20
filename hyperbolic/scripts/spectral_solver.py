"""
Reference "simulation" for the hyperbolic stage: 1D viscous Burgers' equation

    u_t + u u_x = nu * u_xx,   x in [0, L) periodic

solved with a Fourier pseudo-spectral method. The stiff linear diffusion
term is handled exactly via an integrating factor (unconditionally stable,
no dt restriction from nu); the nonlinear advection term u*u_x - written in
conservative flux form -(0.5*u^2)_x, which is what actually forms the shock
as u steepens - is stepped explicitly with RK4, so the only stability limit
is the usual CFL condition on the advection speed. 2/3-rule dealiasing is
applied to the nonlinear term, which matters here specifically because
without it the steepening gradients near shock formation alias onto low
wavenumbers and blow up the reference solution itself - not something you'd
notice on a smooth low-amplitude test case, but this benchmark is chosen
exactly because it isn't smooth for long.
"""
import numpy as np


def make_wavenumbers(nx: int, L: float) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)


def dealias_mask(nx: int) -> np.ndarray:
    """2/3 rule: zero the top third of wavenumbers (by index) to prevent
    aliasing errors from the quadratic nonlinearity."""
    k_idx = np.fft.fftfreq(nx, d=1.0 / nx)
    cutoff = nx / 3.0
    return (np.abs(k_idx) < cutoff).astype(np.float64)


def nonlinear_rhs_hat(u_hat: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    u = np.fft.ifft(u_hat).real
    flux_hat = np.fft.fft(0.5 * u * u) * mask
    return -1j * k * flux_hat


def rk4_nonlinear_step(u_hat: np.ndarray, k: np.ndarray, mask: np.ndarray, dt: float) -> np.ndarray:
    k1 = nonlinear_rhs_hat(u_hat, k, mask)
    k2 = nonlinear_rhs_hat(u_hat + dt / 2 * k1, k, mask)
    k3 = nonlinear_rhs_hat(u_hat + dt / 2 * k2, k, mask)
    k4 = nonlinear_rhs_hat(u_hat + dt * k3, k, mask)
    return u_hat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_burgers(u0: np.ndarray, L: float, nu: float, dt: float, n_steps: int,
                   save_every: int = 1) -> np.ndarray:
    """Roll out from initial condition u0 (shape (nx,)) for n_steps of size
    dt, saving every `save_every` steps (plus the initial condition).
    Returns trajectory of shape (n_saved + 1, nx).
    """
    nx = u0.shape[0]
    k = make_wavenumbers(nx, L)
    mask = dealias_mask(nx)
    diff_factor_half = np.exp(-nu * k**2 * dt / 2)

    u_hat = np.fft.fft(u0)
    traj = [u0.copy()]
    for step in range(n_steps):
        u_hat *= diff_factor_half
        u_hat = rk4_nonlinear_step(u_hat, k, mask, dt)
        u_hat *= diff_factor_half
        if (step + 1) % save_every == 0:
            u = np.fft.ifft(u_hat).real
            if not np.all(np.isfinite(u)):
                raise FloatingPointError(f"solver blew up at step {step + 1}")
            traj.append(u)
    return np.array(traj)


def random_initial_condition(nx: int, L: float, rng: np.random.Generator,
                              n_modes: int = 6, decay: float = 1.3) -> np.ndarray:
    """Smooth periodic IC as a random truncated Fourier series - amplitude
    decays with mode number so energy is concentrated at large scales, like
    the initial conditions used in neural-operator Burgers benchmarks. These
    smooth ICs are what steepen into shocks under the nonlinear advection.
    """
    x = np.linspace(0.0, L, nx, endpoint=False)
    u0 = np.zeros(nx)
    for j in range(1, n_modes + 1):
        amp = rng.uniform(-1.0, 1.0) / j**decay
        phase = rng.uniform(0, 2 * np.pi)
        u0 += amp * np.sin(2 * np.pi * j * x / L + phase)
    scale = rng.uniform(1.5, 3.0)
    u0 = scale * u0 / (np.abs(u0).max() + 1e-8)
    return u0

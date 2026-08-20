"""
Classical finite-difference operators, used two ways in this workflow:
  1. as the reference "simulation" that generates training targets and that
     we validate the GNN against
  2. as an analytic physics-residual term added to the training loss, so the
     GNN is nudged toward solutions that actually satisfy the PDE, not just
     ones that match the reference pointwise
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def _laplacian_matrix(nx: int, ny: int, dx: float, dy: float) -> sp.csr_matrix:
    """Sparse 5-point discrete Laplacian for an nx-by-ny grid, row-major
    indexing i = row * nx + col, homogeneous (interior) stencil everywhere;
    boundary rows get overwritten by the caller to enforce Dirichlet BCs.
    """
    n = nx * ny
    A = sp.lil_matrix((n, n))
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    for row in range(ny):
        for col in range(nx):
            i = row * nx + col
            A[i, i] = -2 * inv_dx2 - 2 * inv_dy2
            if col > 0:
                A[i, i - 1] = inv_dx2
            if col < nx - 1:
                A[i, i + 1] = inv_dx2
            if row > 0:
                A[i, i - nx] = inv_dy2
            if row < ny - 1:
                A[i, i + nx] = inv_dy2
    return A.tocsr()


def build_poisson_operator(nx: int, ny: int, dx: float, dy: float, boundary_mask: np.ndarray):
    """Build and LU-factorize the discrete Poisson system once so many
    samples (same grid, same BC locations, different f / BC values) can be
    solved cheaply. Returns a `solve(f, bc_value) -> u` closure.
    """
    L = _laplacian_matrix(nx, ny, dx, dy)
    A = (-L).tolil()
    bidx = np.where(boundary_mask)[0]
    for i in bidx:
        A.rows[i] = [i]
        A.data[i] = [1.0]
    A_csr = A.tocsr()
    lu = spla.splu(A_csr.tocsc())

    def solve(f: np.ndarray, bc_value: np.ndarray) -> np.ndarray:
        b = f.copy().astype(np.float64)
        b[bidx] = bc_value[bidx]
        return lu.solve(b)

    return solve


def poisson_fd_solve(f: np.ndarray, nx: int, ny: int, dx: float, dy: float,
                      boundary_mask: np.ndarray, bc_value: np.ndarray) -> np.ndarray:
    """Convenience one-shot solve (builds+factorizes a fresh operator every
    call). Use build_poisson_operator directly when solving many samples on
    the same grid.
    """
    solve = build_poisson_operator(nx, ny, dx, dy, boundary_mask)
    return solve(f, bc_value)


def discrete_laplacian_torch(u: torch.Tensor, edge_index: torch.Tensor,
                              dx: float, dy: float) -> torch.Tensor:
    """Apply the same 5-point Laplacian stencil to a predicted field u,
    using the graph edges, so it works whether u is a single sample (N,) or
    a batch stacked along dim 0 (N,) with a batched edge_index from PyG.

    Returns Laplacian(u) at every node (boundary nodes will be wrong here
    since we don't special-case them - callers should mask to interior only).
    """
    src, dst = edge_index[0], edge_index[1]
    diff = u[dst] - u[src]
    # average 1/dx^2 and 1/dy^2 since grid is uniform in each direction and
    # every node has up to 4 unit-stencil neighbors; using dx==dy keeps this
    # simple, callers should ensure dx == dy for exact correctness.
    inv_h2 = 1.0 / dx**2
    agg = torch.zeros_like(u)
    agg.index_add_(0, src, diff * inv_h2)
    return agg


def poisson_residual_scaled(u: torch.Tensor, edge_index: torch.Tensor,
                             f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """PDE residual for -Laplacian(u) = f, but written in the *scaled* form

        -sum_neighbors(u_j - u_i)  -  f * dx * dy  =  0

    i.e. the discrete equation multiplied through by dx^2, instead of
    dividing by it. Mathematically identical to the residual you'd get from
    discrete_laplacian_torch, but numerically well-conditioned: on a fine
    grid, 1/dx^2 can be several orders of magnitude (e.g. ~960 on a 32x32
    unit-square grid), which blows up the raw residual and its loss gradient
    long before the network has learned anything - this form keeps the
    residual in the same O(u) scale as the field itself, so it trains
    alongside the supervised loss instead of drowning it out.
    """
    src, dst = edge_index[0], edge_index[1]
    diff = u[dst] - u[src]
    agg = torch.zeros_like(u)
    agg.index_add_(0, src, diff)
    return -agg - f * dx * dy

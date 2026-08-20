"""
Structured grid -> graph utilities shared across elliptic, parabolic, and
hyperbolic stages. A regular grid is used here (so we have a cheap, exact
finite-difference reference solution to train / validate against), but the
graph representation is the same one you'd use for an unstructured CFD mesh -
nodes with positions, edges with relative-displacement features. Swapping in
a real unstructured mesh later only means changing how edge_index/pos are
built, not the model or training loop.

Two edge sets are built:
  - stencil edge_index: plain 4-connectivity, unit grid spacing. Used for the
    physics-residual loss, since the discrete Laplacian stencil is only
    valid on direct neighbors.
  - multiscale edge_index: stencil edges PLUS long-range "shortcut" edges at
    strides 2, 4, 8, ... (same idea as GraphCast's multi-mesh, or a dilated
    convolution). Elliptic PDEs like Poisson have genuinely global coupling -
    every node's value depends on the source term everywhere in the domain -
    so a message-passing GNN with only local edges needs as many layers as
    the grid diameter to represent that. A handful of long-range edges lets
    a shallow GNN reach the whole domain in just a few hops, which is what
    the model actually trains on.
"""
import numpy as np
import torch


def build_periodic_chain_graph(nx: int, L: float, strides=(1, 2, 4, 8)):
    """Build a 1D periodic chain graph with multiscale shortcut edges.

    Nodes are evenly spaced on [0, L) with periodic wrap-around. Edges
    connect each node to its neighbors at distances given by `strides`
    in both directions, wrapping modulo nx.

    The multiscale shortcut edges serve the same purpose as in the 2D
    grid case: they give a shallow GNN a full-domain receptive field in
    a small number of hops, which matters for capturing wave propagation
    over the whole domain.

    Args:
        nx: number of grid nodes
        L: domain length (periodic, so node i is at x = i * L / nx)
        strides: hop distances to include as edges (both directions)

    Returns:
        pos: (nx, 1) float tensor of node x-coordinates
        edge_index: (2, E) long tensor
        edge_attr: (E, 2) float tensor of [signed_dx, |dx|], where dx
                   is the shortest (wrapped) signed displacement along
                   the periodic domain
        dx: grid spacing L / nx (float)
    """
    dx = L / nx
    x = np.linspace(0.0, L, nx, endpoint=False)  # shape (nx,)

    seen = set()
    src_list, dst_list = [], []
    for s in strides:
        for i in range(nx):
            for sign in (+1, -1):
                j = (i + sign * s) % nx
                if (i, j) not in seen:
                    seen.add((i, j))
                    src_list.append(i)
                    dst_list.append(j)

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    edge_index = np.stack([src, dst])  # (2, E)

    # Signed displacement, wrapped to the half-open interval (-L/2, L/2].
    # This gives the model a consistent notion of "which direction and how
    # far" regardless of whether the edge wraps around the periodic boundary.
    dx_ij = x[dst] - x[src]
    dx_ij = (dx_ij + L / 2) % L - L / 2
    ea = np.stack([dx_ij, np.abs(dx_ij)], axis=1).astype(np.float32)  # (E, 2)

    return (
        torch.tensor(x[:, None], dtype=torch.float32),   # (nx, 1)
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(ea, dtype=torch.float32),
        float(dx),
    )


def build_grid_graph(nx: int, ny: int, extent=(0.0, 1.0, 0.0, 1.0), strides=(1, 2, 4, 8, 16)):
    """Build a regular nx-by-ny grid as a multiscale graph.

    Returns:
        pos: (N, 2) float tensor of (x, y) node coordinates
        edge_index: (2, E) long tensor, multiscale (for the GNN), both directions
        edge_attr: (E, 3) float tensor of [dx, dy, dist] for edge_index
        stencil_edge_index: (2, E_s) long tensor, direct 4-neighbors only (for the PDE residual)
        boundary_mask: (N,) bool tensor, True on the domain boundary
        dx, dy: grid spacing (floats)
        shape: (ny, nx) for reshaping node arrays back to images
    """
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # shape (ny, nx)
    pos = np.stack([xx.ravel(), yy.ravel()], axis=1)  # node i = row-major (row, col)

    def idx(row, col):
        return row * nx + col

    def stencil_edges():
        edges = []
        for row in range(ny):
            for col in range(nx):
                i = idx(row, col)
                if col + 1 < nx:
                    j = idx(row, col + 1)
                    edges.append((i, j))
                    edges.append((j, i))
                if row + 1 < ny:
                    j = idx(row + 1, col)
                    edges.append((i, j))
                    edges.append((j, i))
        return edges

    def multiscale_edges():
        edges = set()
        for s in strides:
            if s >= max(nx, ny):
                continue
            for row in range(ny):
                for col in range(nx):
                    i = idx(row, col)
                    if col + s < nx:
                        j = idx(row, col + s)
                        edges.add((i, j))
                        edges.add((j, i))
                    if row + s < ny:
                        j = idx(row + s, col)
                        edges.add((i, j))
                        edges.add((j, i))
        return list(edges)

    def to_tensors(edge_list):
        ei = np.array(edge_list, dtype=np.int64).T  # (2, E)
        src, dst = ei[0], ei[1]
        rel = pos[dst] - pos[src]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        ea = np.concatenate([rel, dist], axis=1)
        return torch.tensor(ei, dtype=torch.long), torch.tensor(ea, dtype=torch.float32)

    stencil_edge_index, _ = to_tensors(stencil_edges())
    edge_index, edge_attr = to_tensors(multiscale_edges())

    boundary_mask = np.zeros(nx * ny, dtype=bool)
    boundary_mask[[idx(0, c) for c in range(nx)]] = True
    boundary_mask[[idx(ny - 1, c) for c in range(nx)]] = True
    boundary_mask[[idx(r, 0) for r in range(ny)]] = True
    boundary_mask[[idx(r, nx - 1) for r in range(ny)]] = True

    return (
        torch.tensor(pos, dtype=torch.float32),
        edge_index,
        edge_attr,
        stencil_edge_index,
        torch.tensor(boundary_mask, dtype=torch.bool),
        float(dx),
        float(dy),
        (ny, nx),
    )

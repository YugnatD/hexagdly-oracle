"""Pure-Python first-principles oracle for hexagdly's share_neighbors modes.

Recomputes expected conv output directly from a grid, a weight dict, and one
of geometry.py's neighbor/offset tables -- without going through any
framework's weight-materialization code path. Stride=1 only (see the package
README).
"""


def oracle(grid, weights, neighbor_table):
    """kernel_size=1 oracle: center handled separately (not in neighbor_table)."""
    n_rows, n_cols = len(grid), len(grid[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for row in range(n_rows):
        for col in range(n_cols):
            parity = col % 2
            out[row][col] += weights[0] * grid[row][col]
            for (dr, dc), g in neighbor_table[parity].items():
                ir, ic = row + dr, col + dc
                if 0 <= ir < n_rows and 0 <= ic < n_cols:
                    out[row][col] += weights[g] * grid[ir][ic]
    return out


def oracle_k2(grid, weights, neighbor_table):
    """kernel_size=2 oracle: neighbor_table already includes the center (0,0):0."""
    n_rows, n_cols = len(grid), len(grid[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for row in range(n_rows):
        for col in range(n_cols):
            parity = col % 2
            for (dr, dc), g in neighbor_table[parity].items():
                ir, ic = row + dr, col + dc
                if 0 <= ir < n_rows and 0 <= ic < n_cols:
                    out[row][col] += weights[g] * grid[ir][ic]
    return out


def oracle_noshare(grid, weights, offsets):
    """No weight sharing: `offsets` is geometry.py's NOSHARE(2)_OFFSETS,
    keyed by (sub_kernel, row, col) -> per-parity (dr, dc)."""
    n_rows, n_cols = len(grid), len(grid[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for row in range(n_rows):
        for col in range(n_cols):
            parity = col % 2
            for cell, parity_map in offsets.items():
                dr, dc = parity_map[parity]
                ir, ic = row + dr, col + dc
                if 0 <= ir < n_rows and 0 <= ic < n_cols:
                    out[row][col] += weights[cell] * grid[ir][ic]
    return out

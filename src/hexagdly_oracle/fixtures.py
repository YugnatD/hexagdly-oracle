"""Deterministic test grids and weight dicts shared by both suites' oracle tests.

Grid values are simple, non-repeating-ish deterministic patterns (not
hand-transcribed) so a routing bug shows up as a numeric mismatch instead of
being masked by coincidentally-equal neighbor values. Weight dicts use
distinct, easily-traceable values (small primes for the k=2 diag/sym cases)
so a mis-assigned weight group is obvious in a failing diff.
"""

GRID_SMALL = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

GRID_LARGE = [
    [2.0, 1.0, 5.0, 4.0],
    [4.0, 3.0, 2.0, 9.0],
    [2.0, 7.0, 1.0, 1.0],
    [2.0, 4.0, 4.0, 9.0],
    [1.0, 9.0, 4.0, 9.0],
]

GRID_K2 = [
    [2.0, 1.0, 5.0, 4.0, 3.0, 7.0],
    [4.0, 3.0, 2.0, 9.0, 1.0, 5.0],
    [2.0, 7.0, 1.0, 1.0, 8.0, 2.0],
    [2.0, 4.0, 4.0, 9.0, 3.0, 6.0],
    [1.0, 9.0, 4.0, 9.0, 2.0, 4.0],
    [3.0, 5.0, 2.0, 6.0, 7.0, 1.0],
    [4.0, 2.0, 8.0, 3.0, 1.0, 9.0],
]

WEIGHTS_RING = {0: 2.0, 1: 3.0}
WEIGHTS_DIAG = {0: 2.0, 1: 3.0, 2: 5.0, 3: 7.0}
WEIGHTS_SYM = {0: 2.0, 1: 3.0, 2: 5.0, 3: 7.0}
WEIGHTS_NOSHARE = {
    (0, 0, 0): 2.0,
    (0, 1, 0): 11.0,
    (0, 2, 0): 3.0,
    (1, 0, 0): 5.0,
    (1, 0, 1): 7.0,
    (1, 1, 0): 13.0,
    (1, 1, 1): 17.0,
}

WEIGHTS_RING2 = {0: 2.0, 1: 3.0, 2: 5.0}
WEIGHTS_DIAG2 = {
    i: p for i, p in enumerate([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])
}
WEIGHTS_SYM2 = dict(WEIGHTS_DIAG2)
WEIGHTS_NOSHARE2 = {
    (0, 0, 0): 2.0,
    (0, 1, 0): 3.0,
    (0, 2, 0): 5.0,
    (0, 3, 0): 7.0,
    (0, 4, 0): 11.0,
    (1, 0, 0): 13.0,
    (1, 0, 1): 17.0,
    (1, 1, 0): 19.0,
    (1, 1, 1): 23.0,
    (1, 2, 0): 29.0,
    (1, 2, 1): 31.0,
    (1, 3, 0): 37.0,
    (1, 3, 1): 41.0,
    (2, 0, 0): 43.0,
    (2, 0, 1): 47.0,
    (2, 1, 0): 53.0,
    (2, 1, 1): 59.0,
    (2, 2, 0): 61.0,
    (2, 2, 1): 67.0,
}

"""Self-consistency tests for hexagdly-oracle: no framework dependency, no
hexagdly install required. These validate the package's own internal
plumbing (tables + oracle functions + fixtures agree with each other);
actually checking a real hex conv layer's output against this oracle is each
consuming repo's job (keras-hexagdly / pytorch-hexagdly test suites).
"""

from hexagdly_oracle import (
    DIAG_NEIGHBORS,
    DIAG2_NEIGHBORS,
    NOSHARE_OFFSETS,
    NOSHARE2_OFFSETS,
    RING_NEIGHBORS,
    RING2_NEIGHBORS,
    SYM_NEIGHBORS,
    SYM2_NEIGHBORS,
    oracle,
    oracle_k2,
    oracle_noshare,
)
from hexagdly_oracle.fixtures import (
    GRID_LARGE,
    GRID_K2,
    GRID_SMALL,
    WEIGHTS_DIAG,
    WEIGHTS_DIAG2,
    WEIGHTS_NOSHARE,
    WEIGHTS_NOSHARE2,
    WEIGHTS_RING,
    WEIGHTS_RING2,
    WEIGHTS_SYM,
    WEIGHTS_SYM2,
)
from hexagdly_oracle.testing import check_grid
from hexagdly_oracle.weight_maps import HARDCODED_MAPS


def test_ring_uniform_counts_neighbors():
    # Uniform grid of 1s, w0=0, w1=1: output[r,c] = number of valid
    # (in-bounds) ring-1 neighbors.
    grid = [[1.0] * 4 for _ in range(5)]
    got = oracle(grid, {0: 0.0, 1: 1.0}, RING_NEIGHBORS)
    assert abs(got[2][1] - 6.0) < 1e-9  # interior pixel: all 6 neighbors in bounds
    assert abs(got[0][0] - 2.0) < 1e-9  # corner: only 2 of 6 neighbors in bounds


def test_diag_antipodal_symmetry():
    grid = [[1.0] * 6 for _ in range(7)]
    got = oracle(grid, {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}, DIAG_NEIGHBORS)
    for r in range(1, 6):
        for c in range(1, 5):
            assert abs(got[r][c] - 6.0) < 1e-9


def test_oracle_runs_for_every_mode_k1():
    for weights, table in (
        (WEIGHTS_RING, RING_NEIGHBORS),
        (WEIGHTS_DIAG, DIAG_NEIGHBORS),
        (WEIGHTS_SYM, SYM_NEIGHBORS),
    ):
        out = oracle(GRID_SMALL, weights, table)
        check_grid(out, out, "self", tol=1e-9)


def test_oracle_k2_runs_for_every_mode():
    for weights, table in (
        (WEIGHTS_RING2, RING2_NEIGHBORS),
        (WEIGHTS_DIAG2, DIAG2_NEIGHBORS),
        (WEIGHTS_SYM2, SYM2_NEIGHBORS),
    ):
        out = oracle_k2(GRID_K2, weights, table)
        check_grid(out, out, "self", tol=1e-9)


def test_oracle_noshare_k1_and_k2():
    out1 = oracle_noshare(GRID_LARGE, WEIGHTS_NOSHARE, NOSHARE_OFFSETS)
    out2 = oracle_noshare(GRID_K2, WEIGHTS_NOSHARE2, NOSHARE2_OFFSETS)
    check_grid(out1, out1, "self", tol=1e-9)
    check_grid(out2, out2, "self", tol=1e-9)


def test_check_grid_detects_mismatch():
    try:
        check_grid([[1.0]], [[2.0]], "mismatch check")
    except AssertionError:
        return
    raise AssertionError("check_grid failed to detect a real mismatch")


def test_hardcoded_maps_ring_group_counts():
    # ring r holds 1 (r=0) or 6*r cells.
    maps, num_groups = HARDCODED_MAPS[("ring", 2, 0)]
    assert num_groups == 3
    counts = [0, 0, 0]
    for m in maps:
        for g in range(3):
            counts[g] += int((m == g).sum())
    assert counts == [1, 6, 12]


def test_hardcoded_maps_both_parities_present_for_k1_and_k2():
    for mode in ("ring", "diag", "sym"):
        for n in (1, 2):
            assert (mode, n, 0) in HARDCODED_MAPS
            assert (mode, n, 1) in HARDCODED_MAPS

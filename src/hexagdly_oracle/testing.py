"""Tiny comparison helper shared by both suites' oracle tests."""


def check_grid(expected, got, label, tol=1e-2):
    assert len(expected) == len(got), f"{label}: row count mismatch"
    for r in range(len(expected)):
        assert len(expected[r]) == len(got[r]), f"{label}: row {r} length mismatch"
        for c in range(len(expected[r])):
            diff = abs(expected[r][c] - got[r][c])
            assert diff < tol, (
                f"{label}: mismatch at ({r},{c}): expected {expected[r][c]}, got {got[r][c]}"
            )

"""Hardcoded, hand-verified weight-group tables for share_neighbors modes.

`HARDCODED_MAPS[(mode, kernel_size, parity)]` gives `(sub_kernel_group_arrays,
num_weights)`: for each hexbase sub-kernel, an array of the same shape as
that sub-kernel where each cell holds its weight-group index (matching
geometry.py's *_NEIGHBORS tables), plus the total number of distinct weight
groups for that mode/kernel_size.

Originally derived and verified by hand in pytorch-hexagdly (pointy-top ASCII
diagrams, see comments below); both keras-hexagdly and pytorch-hexagdly now
validate their own `weight_maps_2d`-equivalent derivation against this single
shared table instead of one project reaching into the other's private
attributes.

kernel_size 1 and 2 only -- share_neighbors is not implemented for larger
kernels in either framework.
"""

import numpy as np

HARDCODED_MAPS = {
    # --- ring n=1 (identical for both parities: center=0, all 6 neighbors=1) ---
    ("ring", 1, 0): (
        [np.array([[1], [0], [1]]), np.array([[1, 1], [1, 1]])],
        2,
    ),
    ("ring", 1, 1): (
        [np.array([[1], [0], [1]]), np.array([[1, 1], [1, 1]])],
        2,
    ),
    # --- ring n=2 ---
    # From user-validated pointy-top ASCII:
    #           [ B ]         A=1 (ring-1, 6 cells): (+1,0),(-1,0),(0,-1),(0,+1),(-1,-1),(-1,+1)
    #        [ B ] [ B ]      B=2 (ring-2, 12 cells): all others
    #     [ B ] [ A ] [ B ]
    #        [ A ] [ A ]
    #     [ B ] [ X ] [ B ]
    #        [ A ] [ A ]
    #     [ B ] [ A ] [ B ]
    #        [ B ] [ B ]
    #           [ B ]
    ("ring", 2, 0): (
        [
            np.array([[2], [1], [0], [1], [2]]),  # sub0
            np.array([[2, 2], [1, 1], [1, 1], [2, 2]]),  # sub1 even
            np.array([[2, 2], [2, 2], [2, 2]]),  # sub2
        ],
        3,
    ),
    ("ring", 2, 1): (
        [
            np.array([[2], [1], [0], [1], [2]]),  # sub0
            np.array([[2, 2], [1, 1], [1, 1], [2, 2]]),  # sub1 odd
            np.array([[2, 2], [2, 2], [2, 2]]),  # sub2
        ],
        3,
    ),
    # --- diag n=1 (identical for both parities) ---
    ("diag", 1, 0): (
        [np.array([[3], [0], [3]]), np.array([[1, 2], [2, 1]])],
        4,
    ),
    ("diag", 1, 1): (
        [np.array([[3], [0], [3]]), np.array([[1, 2], [2, 1]])],
        4,
    ),
    # --- diag n=2 ---
    # 9 visual-antipodal pairs from the flat-top hex kernel (read from pointy-top ASCII):
    #   E=1: (+2,0)<->(-2,0)     C=5: (+1,0)<->(-1,0)     D=9: (0,-2)<->(0,+2)
    #   F=2: (+1,-1)<->(-2,+1)   H=4: (+1,-2)<->(-1,+2)
    #   I=3: (+1,+1)<->(-2,-1)   G=6: (+1,+2)<->(-1,-2)
    #   B=7: (0,-1)<->(-1,+1)    A=8: (0,+1)<->(-1,-1)
    # Orphans (-2,+-1) [even] or (+2,+-1) [odd] pair with (+1,-+1) or (-1,+-1) respectively.
    ("diag", 2, 0): (
        [
            np.array([[1], [5], [0], [5], [1]]),  # sub0
            np.array([[3, 2], [8, 7], [7, 8], [2, 3]]),  # sub1 even
            np.array([[6, 4], [9, 9], [4, 6]]),  # sub2
        ],
        10,
    ),
    ("diag", 2, 1): (
        [
            np.array([[1], [5], [0], [5], [1]]),  # sub0
            np.array([[3, 2], [8, 7], [7, 8], [2, 3]]),  # sub1 odd
            np.array([[6, 4], [9, 9], [4, 6]]),  # sub2
        ],
        10,
    ),
    # --- sym n=1 (identical for both parities) ---
    ("sym", 1, 0): (
        [np.array([[2], [0], [3]]), np.array([[2, 1], [3, 1]])],
        4,
    ),
    ("sym", 1, 1): (
        [np.array([[2], [0], [3]]), np.array([[2, 1], [3, 1]])],
        4,
    ),
    # --- sym n=2 ---
    # 9 adjacent pairs from user-validated pointy-top ASCII:
    #           [ G ]         G=7: (+2,0)<->(+1,-1)
    #        [ G ] [ F ]      F=6: (+1,+1)<->(+1,+2)
    #     [ H ] [ A ] [ F ]   H=8: (+1,-2)<->(0,-2)    A=1: (+1,0)<->(0,-1)
    #        [ A ] [ B ]      B=2: (0,+1)<->(-1,+1)
    #     [ H ] [ X ] [ E ]   E=5: (0,+2)<->(-1,+2)
    #        [ C ] [ B ]      C=3: (-1,-1)<->(-1,0)
    #     [ I ] [ C ] [ E ]   I=9: (-1,-2)<->(-2,-1)[even] or (-1,-2)<->(+2,-1)[odd]
    #        [ I ] [ D ]      D=4: (-2,+1)<->(-2,0)[even] or (+2,+1)<->(-2,0)[odd]
    #           [ D ]
    ("sym", 2, 0): (
        [
            np.array([[4], [3], [0], [1], [7]]),  # sub0
            np.array([[9, 4], [3, 2], [1, 2], [7, 6]]),  # sub1 even
            np.array([[9, 5], [8, 5], [8, 6]]),  # sub2
        ],
        10,
    ),
    ("sym", 2, 1): (
        [
            np.array([[4], [3], [0], [1], [7]]),  # sub0
            np.array([[9, 4], [3, 2], [1, 2], [7, 6]]),  # sub1 odd
            np.array([[9, 5], [8, 5], [8, 6]]),  # sub2
        ],
        10,
    ),
}

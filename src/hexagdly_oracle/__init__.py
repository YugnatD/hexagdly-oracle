from hexagdly_oracle.geometry import (
    DIAG2_NEIGHBORS,
    DIAG_NEIGHBORS,
    NOSHARE2_OFFSETS,
    NOSHARE_OFFSETS,
    RING2_NEIGHBORS,
    RING_NEIGHBORS,
    SYM2_NEIGHBORS,
    SYM_NEIGHBORS,
)
from hexagdly_oracle.hex_reference import (
    conv2d_expected,
    conv2d_input_nhwc,
    conv3d_expected,
    conv3d_input_ndhwc,
    custom_kernel_expected,
    custom_kernel_input_nhwc,
    maxpool2d_expected,
    maxpool2d_input_nhwc,
    maxpool3d_expected,
)
from hexagdly_oracle.oracle import oracle, oracle_k2, oracle_noshare
from hexagdly_oracle.weight_maps import HARDCODED_MAPS

__all__ = [
    "DIAG2_NEIGHBORS",
    "DIAG_NEIGHBORS",
    "HARDCODED_MAPS",
    "NOSHARE2_OFFSETS",
    "NOSHARE_OFFSETS",
    "RING2_NEIGHBORS",
    "RING_NEIGHBORS",
    "SYM2_NEIGHBORS",
    "SYM_NEIGHBORS",
    "conv2d_expected",
    "conv2d_input_nhwc",
    "conv3d_expected",
    "conv3d_input_ndhwc",
    "custom_kernel_expected",
    "custom_kernel_input_nhwc",
    "maxpool2d_expected",
    "maxpool2d_input_nhwc",
    "maxpool3d_expected",
    "oracle",
    "oracle_k2",
    "oracle_noshare",
]

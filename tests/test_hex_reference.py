"""Self-consistency tests for hex_reference.py: shapes and a couple of
hand-checked values. No framework dependency -- actually comparing a real
Conv2d/Conv3d/MaxPool2d/MaxPool3d/Conv2d_CustomKernel layer's output against
these arrays is each consuming repo's job.
"""

import numpy as np

from hexagdly_oracle import (
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


def test_conv2d_shapes_and_bias():
    x = conv2d_input_nhwc(in_channels=3)
    assert x.shape == (1, 5, 8, 3)
    e1 = conv2d_expected(3, kernel_size=1, stride=1, bias_value=0.0)
    e2 = conv2d_expected(3, kernel_size=1, stride=1, bias_value=1.0)
    assert e1.shape == (5, 8)
    np.testing.assert_allclose(e2, e1 + 1.0)


def test_conv2d_stride_shrinks_output():
    e1 = conv2d_expected(1, kernel_size=1, stride=1, bias_value=0.0)
    e2 = conv2d_expected(1, kernel_size=1, stride=2, bias_value=0.0)
    e3 = conv2d_expected(1, kernel_size=1, stride=3, bias_value=0.0)
    assert e1.shape == (5, 8)
    assert e2.shape == (2, 4)
    assert e3.shape == (2, 3)


def test_conv3d_shapes():
    x = conv3d_input_ndhwc(in_channels=2, depth=4)
    assert x.shape == (1, 4, 5, 8, 2)
    e = conv3d_expected(
        in_channels=2,
        depth=4,
        kernel_size_depth=1,
        kernel_size_hex=1,
        stride_depth=1,
        stride_hex=1,
        bias_value=0.0,
    )
    assert e.shape == (4, 5, 8)


def test_maxpool2d_channel_independence():
    x = maxpool2d_input_nhwc(in_channels=2)
    assert x.shape == (1, 5, 8, 2)
    e = maxpool2d_expected(2, kernel_size=1, stride=1)
    assert e.shape == (5, 8, 2)
    # channel 1 is channel 0 shifted by a constant (see hex_reference's
    # channel_dist convention), so their difference must be uniform.
    diff = e[..., 1] - e[..., 0]
    assert np.allclose(diff, diff.flat[0])


def test_maxpool3d_shapes():
    e = maxpool3d_expected(
        in_channels=1,
        depth=4,
        kernel_size_depth=1,
        kernel_size_hex=1,
        stride_depth=1,
        stride_hex=1,
    )
    assert e.shape == (4, 5, 8, 1)


def test_custom_kernel_shapes_and_bias():
    x = custom_kernel_input_nhwc(in_channels=2)
    assert x.shape == (1, 4, 6, 2)
    e1 = custom_kernel_expected(2, kernel_size=1, stride=1, bias_value=0.0)
    e2 = custom_kernel_expected(2, kernel_size=1, stride=1, bias_value=1.0)
    assert e1.shape == (4, 6)
    np.testing.assert_allclose(e2, e1 + 1.0)

"""Keras-only edge cases: everything else in this family (numerical
robustness, dtype handling, debug weights) is shared and dual-backed -- see
test_layers.py. What's left here has no pytorch-hexagdly equivalent:

- in_channels-mismatch / dynamic-shape checks rely on Keras's symbolic
  Input/build() mechanism, which pytorch has no equivalent of at all.
- construction-time ValueError for bad kernel_size/stride is a
  keras-hexagdly design decision pytorch-hexagdly does not replicate
  (confirmed: its Conv2d/Conv3d.__init__ assigns kernel_size/stride with no
  validation, and fails later with an obscure, backend/location-dependent
  error instead -- exactly the failure mode this port's validation exists to
  avoid).
- the out_channels-only in_channels-inference call form for Conv3d is a
  keras-hexagdly-only convenience API; pytorch-hexagdly always requires
  in_channels explicitly.
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
hgly = pytest.importorskip("keras_hexagdly")


def test_in_channels_mismatch_raises():
    """An explicit in_channels that disagrees with the actual input must fail
    loudly, not silently use the wrong value."""
    layer = hgly.Conv2d(in_channels=3, out_channels=2, kernel_size=1)
    with pytest.raises(ValueError):
        layer(keras.ops.zeros((1, 9, 8, 5)))  # 5 channels, declared 3


def test_dynamic_spatial_dims_raise():
    """The hex addressing arithmetic needs static H/W; a None spatial dim
    must fail with a clear error, not crash deep inside the conv."""
    layer = hgly.Conv2d(out_channels=2, kernel_size=1)
    with pytest.raises(ValueError):
        layer.build((None, None, None, 1))


def test_dynamic_batch_dim_is_fine():
    """Only the spatial/channel dims need to be static; batch can be None,
    as in a normal Keras functional Input."""
    x = keras.Input(shape=(9, 8, 2))
    layer = hgly.Conv2d(out_channels=3, kernel_size=1)
    y = layer(x)
    assert y.shape == (None, 9, 8, 3)


def test_out_channels_only_form_3d():
    """The out_channels-only Conv3d(...) call form infers in_channels too."""
    x = np.zeros((1, 5, 9, 8, 4), dtype=np.float32)
    layer = hgly.Conv3d(out_channels=2, kernel_size=1)
    out = layer(keras.ops.convert_to_tensor(x))
    assert out.shape == (1, 5, 9, 8, 2)


# ----------------------------------------------------------------------------
# kernel_size/stride validation: upstream PyTorch HexagDLy AND pytorch-hexagdly
# both accept these and fail with an obscure, backend/location-dependent error
# deep inside the first call (ZeroDivisionError, a stray AttributeError, a
# backend-specific "stride must be > 0"...). This port validates at
# construction time instead.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("bad_kernel_size", [0, -1, 1.5, "1", None])
def test_conv2d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.Conv2d(out_channels=1, kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [0, -1, 1.5, "1", None])
def test_conv2d_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv2d(out_channels=1, kernel_size=1, stride=bad_stride)


@pytest.mark.parametrize("bad_kernel_size", [0, -1, 1.5])
def test_maxpool2d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.MaxPool2d(kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [0, -1, 1.5])
def test_conv2d_custom_kernel_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv2d_CustomKernel(stride=bad_stride)


@pytest.mark.parametrize("bad_kernel_size", [(1, 0), (0, 1), (1, 2, 3), "x", -1])
def test_conv3d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.Conv3d(out_channels=1, kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [(1, 0), (0, 1), (1, 2, 3), "x", -1])
def test_conv3d_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv3d(out_channels=1, kernel_size=1, stride=bad_stride)


def test_maxpool3d_bad_kernel_size_raises_at_construction():
    with pytest.raises(ValueError):
        hgly.MaxPool3d(kernel_size=(1, 0))

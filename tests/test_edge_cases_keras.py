"""Keras-only edge cases: everything else in this family (numerical
robustness, dtype handling, debug weights) is shared and dual-backed -- see
test_layers.py. What's left here has no pytorch-hexagdly equivalent:

- dynamic-shape checks rely on Keras's symbolic Input/build() mechanism,
  which pytorch has no equivalent of at all.
- construction-time ValueError for bad kernel_size/stride started as a
  keras-hexagdly-only design decision; pytorch-hexagdly 0.3.0+ validates the
  same way (see its own test suite), so only the exact wording/parametrize
  values below (and in_channels, dynamic shapes) are genuinely Keras-only.
- in_channels is never a constructor argument at all (always inferred in
  build(), like every standard Keras layer) -- this is a deliberate
  departure from pytorch-hexagdly, which always requires it explicitly.
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
hgly = pytest.importorskip("keras_hexagdly")


def test_in_channels_is_not_a_constructor_argument():
    """Unlike pytorch-hexagdly (and unlike this package before 0.4.0),
    in_channels cannot be passed explicitly at all -- it's always inferred
    from the input in build(). Passing it raises loudly rather than being
    silently accepted and ignored or misread as something else."""
    with pytest.raises((TypeError, ValueError)):
        hgly.Conv2d(in_channels=3, filters=2, kernel_size=1)


def test_dynamic_spatial_dims_raise():
    """The hex addressing arithmetic needs static H/W; a None spatial dim
    must fail with a clear error, not crash deep inside the conv."""
    layer = hgly.Conv2d(filters=2, kernel_size=1)
    with pytest.raises(ValueError):
        layer.build((None, None, None, 1))


def test_dynamic_batch_dim_is_fine():
    """Only the spatial/channel dims need to be static; batch can be None,
    as in a normal Keras functional Input."""
    x = keras.Input(shape=(9, 8, 2))
    layer = hgly.Conv2d(filters=3, kernel_size=1)
    y = layer(x)
    assert y.shape == (None, 9, 8, 3)


def test_in_channels_always_inferred_3d():
    """in_channels is inferred from the input for Conv3d too."""
    x = np.zeros((1, 5, 9, 8, 4), dtype=np.float32)
    layer = hgly.Conv3d(filters=2, kernel_size=1)
    out = layer(keras.ops.convert_to_tensor(x))
    assert out.shape == (1, 5, 9, 8, 2)


# ----------------------------------------------------------------------------
# kernel_size/stride validation: upstream PyTorch HexagDLy accepts these and
# fails with an obscure, backend/location-dependent error deep inside the
# first call (ZeroDivisionError, a stray AttributeError, a backend-specific
# "stride must be > 0"...). This port validates at construction time instead.
#
# kernel_size/pool_size=0 (pointwise: a plain per-cell projection, no
# neighbours -- 0 hexagonal rings) is a legitimate value, not a bad one, so
# it is exercised separately below rather than parametrized as "bad". strides
# must stay strictly positive either way.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("bad_kernel_size", [-1, 1.5, "1", None])
def test_conv2d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.Conv2d(filters=1, kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [0, -1, 1.5, "1", None])
def test_conv2d_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv2d(filters=1, kernel_size=1, strides=bad_stride)


@pytest.mark.parametrize("bad_kernel_size", [-1, 1.5])
def test_maxpool2d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    with pytest.raises(ValueError):
        hgly.MaxPool2d(kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [0, -1, 1.5])
def test_conv2d_custom_kernel_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv2d_CustomKernel(strides=bad_stride)


@pytest.mark.parametrize("bad_kernel_size", [(0, 1), (1, 2, 3), "x", -1, 0])
def test_conv3d_bad_kernel_size_raises_at_construction(bad_kernel_size):
    """A bare int must stay positive -- it sets the depth axis too, and a
    0-depth kernel (no time frames at all) is degenerate. (0, 1) is bad for
    the same reason: depth=0. Only the hex element, in the 2-tuple form, can
    be 0 -- see test_conv3d_pointwise_hex_element_is_accepted."""
    with pytest.raises(ValueError):
        hgly.Conv3d(filters=1, kernel_size=bad_kernel_size)


@pytest.mark.parametrize("bad_stride", [(1, 0), (0, 1), (1, 2, 3), "x", -1])
def test_conv3d_bad_stride_raises_at_construction(bad_stride):
    with pytest.raises(ValueError):
        hgly.Conv3d(filters=1, kernel_size=1, strides=bad_stride)


def test_maxpool3d_bad_kernel_size_raises_at_construction():
    """(0, 1): depth=0 is degenerate, same reasoning as Conv3d above."""
    with pytest.raises(ValueError):
        hgly.MaxPool3d(kernel_size=(0, 1))


# ----------------------------------------------------------------------------
# kernel_size/pool_size=0 (pointwise): now accepted, not rejected.
# ----------------------------------------------------------------------------


def test_conv2d_pointwise_kernel_size_is_accepted():
    x = np.zeros((1, 9, 8, 2), dtype=np.float32)
    layer = hgly.Conv2d(filters=3, kernel_size=0)
    assert layer(keras.ops.convert_to_tensor(x)).shape == (1, 9, 8, 3)


def test_maxpool2d_pointwise_pool_size_is_accepted():
    x = np.zeros((1, 9, 8, 2), dtype=np.float32)
    layer = hgly.MaxPool2d(pool_size=0)
    assert layer(keras.ops.convert_to_tensor(x)).shape == (1, 9, 8, 2)


def test_conv3d_pointwise_hex_element_is_accepted():
    """The hex element of the (depth, hex) tuple can be 0; only a bare int
    (which would also zero the depth axis) is rejected."""
    x = np.zeros((1, 5, 9, 8, 4), dtype=np.float32)
    layer = hgly.Conv3d(filters=2, kernel_size=(5, 0))
    assert layer(keras.ops.convert_to_tensor(x)).shape == (1, 1, 9, 8, 2)


def test_maxpool3d_pointwise_hex_element_is_accepted():
    x = np.zeros((1, 5, 9, 8, 4), dtype=np.float32)
    layer = hgly.MaxPool3d(pool_size=(5, 0))
    assert layer(keras.ops.convert_to_tensor(x)).shape == (1, 1, 9, 8, 4)

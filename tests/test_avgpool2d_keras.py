"""Keras-only tests for keras_hexagdly.AvgPool2d against the oracle.

AvgPool2d has no pytorch-hexagdly counterpart (upstream HexagDLy only
provides max pooling), so these are single-backed. The ground truth is this
package's own first-principles oracle: with every weight set to 1,
oracle()/oracle_k2() sum each window over its in-grid cells only, so
oracle(image) / oracle(ones) is exactly the in-grid mean AvgPool2d is
documented to compute -- derived from the measured geometry tables, not from
AvgPool2d's own code path.

The oracle is stride=1 only. Strided outputs are checked by sampling the
stride-1 oracle mean at the strided output positions, using a
Conv2d_CustomKernel whose only non-zero tap is the window centre.
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
hgly = pytest.importorskip("keras_hexagdly")

from hexagdly_oracle import (
    RING2_NEIGHBORS,
    RING_NEIGHBORS,
    oracle,
    oracle_k2,
)

RNG = np.random.default_rng(11)
ATOL = 1e-5

# Every group weight set to 1 turns the oracles into in-grid window sums.
_WINDOW_SUM = {
    1: (oracle, RING_NEIGHBORS, {0: 1.0, 1: 1.0}),
    2: (oracle_k2, RING2_NEIGHBORS, {0: 1.0, 1: 1.0, 2: 1.0}),
}

# Odd and even widths, plus grids narrower than a kernel_size=2 window and a
# single column (see keras-hexagdly 0.5.0's narrow-grid fix).
SHAPES = [(7, 8), (9, 9), (6, 5), (8, 3), (4, 2), (5, 1)]


def _oracle_mean(image, pool_size):
    fn, table, weights = _WINDOW_SUM[pool_size]
    total = np.array(fn(image.tolist(), weights, table))
    count = np.array(fn(np.ones_like(image).tolist(), weights, table))
    return total / count


def _avgpool(image_hwc, pool_size, strides=1):
    out = hgly.AvgPool2d(pool_size, strides=strides)(image_hwc[None])
    return keras.ops.convert_to_numpy(out)[0]


def _centre_sampler(strides):
    """Picks each strided output position's centre cell from its input."""
    centre = np.zeros((1, 1, 3, 1), "float32")
    centre[0, 0, 1, 0] = 1.0
    return hgly.Conv2d_CustomKernel(
        sub_kernels=[centre, np.zeros((1, 1, 2, 2), "float32")], strides=strides
    )


@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_the_oracle_mean(pool_size, shape):
    image = RNG.standard_normal(shape).astype("float32")
    got = _avgpool(image[:, :, None], pool_size)[:, :, 0]
    np.testing.assert_allclose(got, _oracle_mean(image, pool_size), atol=ATOL)


@pytest.mark.parametrize("pool_size", [1, 2])
def test_channels_are_pooled_independently(pool_size):
    image = RNG.standard_normal((7, 8, 3)).astype("float32")
    got = _avgpool(image, pool_size)
    for c in range(3):
        np.testing.assert_allclose(
            got[:, :, c], _oracle_mean(image[:, :, c], pool_size), atol=ATOL
        )


@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strides", [2, 3])
@pytest.mark.parametrize("shape", [(7, 8), (9, 9), (6, 5), (8, 11)])
def test_strided_matches_the_oracle_mean_at_the_strided_positions(
    pool_size, strides, shape
):
    image = RNG.standard_normal(shape).astype("float32")
    stride1_mean = _oracle_mean(image, pool_size).astype("float32")
    expected = keras.ops.convert_to_numpy(
        _centre_sampler(strides)(stride1_mean[None, :, :, None])
    )
    got = _avgpool(image[:, :, None], pool_size, strides)
    assert got.shape == expected.shape[1:]
    np.testing.assert_allclose(got, expected[0], atol=ATOL)


@pytest.mark.parametrize("pool_size", [1, 2])
@pytest.mark.parametrize("strides", [1, 2, 3])
@pytest.mark.parametrize("shape", [(7, 8), (6, 5), (5, 3), (4, 2), (5, 1)])
def test_constant_image_stays_constant(pool_size, strides, shape):
    """Holds at the border only if the divisor counts exactly the in-grid
    cells -- including grids no wider than the stride."""
    got = _avgpool(np.full(shape + (1,), 2.5, "float32"), pool_size, strides)
    np.testing.assert_allclose(got, 2.5, rtol=1e-6)


@pytest.mark.parametrize("strides", [1, 2, 3])
def test_output_grid_matches_maxpool2d(strides):
    """Same window walk as MaxPool2d, so the same output grid."""
    x = RNG.standard_normal((1, 9, 8, 2)).astype("float32")
    avg = hgly.AvgPool2d(2, strides=strides)(x)
    mx = hgly.MaxPool2d(2, strides=strides)(x)
    assert tuple(avg.shape) == tuple(mx.shape)

"""Single source of test logic for hand-verified layer outputs and the
share_neighbors oracle, run against BOTH keras_hexagdly and pytorch_hexagdly
through a small per-framework adapter (`_KerasBackend` / `_PytorchBackend`).

CASES tables, expected values, and assertions are defined exactly once here
-- not copy-pasted into a "test_layers_keras.py" and a "test_layers_pytorch.py"
that differ only by s/hgly/hex/ and NCHW<->NHWC transposes. Each backend is
only instantiated if its framework is importable, so this file works with
either framework installed, both, or neither (in which case BACKENDS is
empty and every parametrized test is trivially, harmlessly empty -- not an
error).

The one place the two ports genuinely diverge in capability is hls4ml
export, which only exists on the keras side (planned for pytorch-hexagdly,
not there yet) -- that lives in this package's own
tests/test_hls4ml_ext_keras.py, single-backed.
"""

import numpy as np
import pytest

from hexagdly_oracle import (
    DIAG_NEIGHBORS as _DIAG_NEIGHBORS,
    DIAG2_NEIGHBORS as _DIAG2_NEIGHBORS,
    NOSHARE_OFFSETS as _NOSHARE_OFFSETS,
    NOSHARE2_OFFSETS as _NOSHARE2_OFFSETS,
    RING_NEIGHBORS as _RING_NEIGHBORS,
    RING2_NEIGHBORS as _RING2_NEIGHBORS,
    SYM_NEIGHBORS as _SYM_NEIGHBORS,
    SYM2_NEIGHBORS as _SYM2_NEIGHBORS,
    conv2d_expected,
    conv2d_input_nhwc,
    conv3d_expected,
    conv3d_input_ndhwc,
    custom_kernel_expected,
    custom_kernel_input_nhwc,
    maxpool2d_expected,
    maxpool2d_input_nhwc,
    maxpool3d_expected,
    oracle as _oracle,
    oracle_k2 as _oracle_k2,
    oracle_noshare as _oracle_noshare,
)
from hexagdly_oracle.fixtures import (
    GRID_K2,
    GRID_LARGE,
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
from hexagdly_oracle.testing import check_grid as _check_grid


# =============================================================================
# Per-framework adapters. Every test below is written once against this
# interface; only these two classes know about keras/torch shapes and APIs.
# =============================================================================


class _KerasBackend:
    name = "keras"

    def __init__(self):
        import keras
        import keras_hexagdly as hgly

        self.keras = keras
        self.hgly = hgly

    # ---- Conv2d / Conv2d_CustomKernel -------------------------------------
    def build_conv2d(
        self, in_channels, out_channels, kernel_size, stride, bias, debug=True
    ):
        return self.hgly.Conv2d(
            out_channels, kernel_size, stride, bias, debug=debug
        )

    def build_conv2d_custom_kernel(self, sub_kernels, stride, bias_arg):
        return self.hgly.Conv2d_CustomKernel(sub_kernels, strides=stride, bias=bias_arg)

    def build_maxpool2d(self, kernel_size, stride):
        return self.hgly.MaxPool2d(kernel_size, stride)

    def run_nhwc(self, layer, x_nhwc):
        out = layer(self.keras.ops.convert_to_tensor(x_nhwc))
        return self.keras.ops.convert_to_numpy(out)

    def debug_kernels_all_ones(self, layer, n):
        return all(
            np.all(np.asarray(layer._base_kernels[i]) == 1.0) for i in range(n + 1)
        )

    # ---- Conv3d / Conv3d_CustomKernel / MaxPool3d -------------------------
    def build_conv3d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias,
        debug=True,
        share_neighbors=False,
        depth_padding="valid",
    ):
        return self.hgly.Conv3d(
            out_channels,
            kernel_size,
            stride,
            bias,
            debug=debug,
            share_neighbors=share_neighbors,
            depth_padding=depth_padding,
        )

    def build_conv3d_custom_kernel(self, sub_kernels, stride, bias_arg):
        return self.hgly.Conv3d_CustomKernel(sub_kernels, strides=stride, bias=bias_arg)

    def build_maxpool3d(self, kernel_size, stride):
        return self.hgly.MaxPool3d(kernel_size, stride)

    def run_ndhwc(self, layer, x_ndhwc):
        out = layer(self.keras.ops.convert_to_tensor(x_ndhwc))
        return self.keras.ops.convert_to_numpy(out)

    def copy_conv3d_kernels(self, src, dst, n):
        for i in range(n + 1):
            dst._base_kernels[i].assign(src._base_kernels[i].numpy())

    # ---- share_neighbors ---------------------------------------------------
    def build_shared_conv2d(self, mode, kernel_size=1, stride=1):
        conv = self.hgly.Conv2d(
            1,
            kernel_size=kernel_size,
            strides=stride,
            use_bias=False,
            share_neighbors=mode,
        )
        _ = conv(self.keras.ops.zeros((1, 10, 10, 1)))  # trigger build()
        return conv

    def build_noshare_conv2d(self, kernel_size=1, stride=1):
        conv = self.hgly.Conv2d(
            1, kernel_size=kernel_size, strides=stride, use_bias=False
        )
        _ = conv(self.keras.ops.zeros((1, 10, 10, 1)))
        return conv

    def set_shared_weights(self, conv, weights):
        num = int(conv.ring_weights.shape[0])
        arr = np.zeros((num, 1, 1), dtype=np.float32)
        for g, w in weights.items():
            arr[g, 0, 0] = w
        conv.ring_weights.assign(arr)

    def set_noshare_weights(self, conv, n, weights):
        for i in range(n + 1):
            kh = 2 * n + 1 - i
            kw = 1 if i == 0 else 2
            arr = np.zeros((kh, kw, 1, 1), dtype=np.float32)
            for (sub_i, r, c), w in weights.items():
                if sub_i == i:
                    arr[r, c, 0, 0] = w
            conv._base_kernels[i].assign(arr)

    def run_forward_grid(self, conv, grid):
        arr = np.array(grid, dtype=np.float32)[
            None, :, :, None
        ]  # (1,H,W,1) channels-last
        out = self.run_nhwc(conv, arr)
        return out[0, :, :, 0].tolist()

    def shared_weight_count(self, conv):
        return int(conv.ring_weights.shape[0])

    def forward_finite(self, conv, height, width, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((1, height, width, 1)).astype(np.float32)
        out = self.run_nhwc(conv, x)
        return np.all(np.isfinite(out)) and out.shape == (1, height, width, 1)

    def check_shared_gradients_nonzero(self, conv, grid):
        if self.keras.backend.backend() != "tensorflow":
            pytest.skip(
                "uses tf.GradientTape; only meaningful on the tensorflow backend"
            )
        import tensorflow as tf

        arr = np.array(grid, dtype=np.float32)[None, :, :, None]
        x = tf.constant(arr)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(conv(x))
        grad = tape.gradient(loss, conv.ring_weights)
        assert grad is not None
        if isinstance(grad, tf.IndexedSlices):
            grad = tf.convert_to_tensor(grad)
        assert (np.abs(grad.numpy()) > 0).all()

    def check_noshare_gradients_nonzero(self, conv, grid, n):
        if self.keras.backend.backend() != "tensorflow":
            pytest.skip(
                "uses tf.GradientTape; only meaningful on the tensorflow backend"
            )
        import tensorflow as tf

        arr = np.array(grid, dtype=np.float32)[None, :, :, None]
        x = tf.constant(arr)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(conv(x))
        grads = tape.gradient(loss, [conv._base_kernels[i] for i in range(n + 1)])
        for i, g in enumerate(grads):
            assert g is not None
            assert (np.abs(g.numpy()) > 0).all(), (
                f"kernel{i} should have non-zero gradients"
            )


class _PytorchBackend:
    name = "pytorch"

    def __init__(self):
        import torch
        import pytorch_hexagdly as hex

        self.torch = torch
        self.hex = hex

    @staticmethod
    def _to_nchw(x_nhwc):
        return np.transpose(x_nhwc, (0, 3, 1, 2))

    @staticmethod
    def _to_nhwc(x_nchw):
        return np.transpose(x_nchw, (0, 2, 3, 1))

    @staticmethod
    def _to_ncdhw(x_ndhwc):
        return np.transpose(x_ndhwc, (0, 4, 1, 2, 3))

    @staticmethod
    def _to_ndhwc(x_ncdhw):
        return np.transpose(x_ncdhw, (0, 2, 3, 4, 1))

    # ---- Conv2d / Conv2d_CustomKernel -------------------------------------
    def build_conv2d(
        self, in_channels, out_channels, kernel_size, stride, bias, debug=True
    ):
        return self.hex.Conv2d(
            in_channels, out_channels, kernel_size, stride, bias, debug
        )

    def build_conv2d_custom_kernel(self, sub_kernels, stride, bias_arg):
        return self.hex.Conv2d_CustomKernel(sub_kernels, stride, bias_arg)

    def build_maxpool2d(self, kernel_size, stride):
        return self.hex.MaxPool2d(kernel_size, stride)

    def run_nhwc(self, layer, x_nhwc):
        out = layer(self.torch.FloatTensor(self._to_nchw(x_nhwc))).detach().numpy()
        return self._to_nhwc(out)

    def debug_kernels_all_ones(self, layer, n):
        return all(
            bool((getattr(layer, f"kernel{i}") == 1.0).all()) for i in range(n + 1)
        )

    # ---- Conv3d / Conv3d_CustomKernel / MaxPool3d -------------------------
    def build_conv3d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias,
        debug=True,
        share_neighbors=False,
        depth_padding="valid",
    ):
        return self.hex.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias,
            debug,
            share_neighbors=(share_neighbors or None),
            depth_padding=depth_padding,
        )

    def build_conv3d_custom_kernel(self, sub_kernels, stride, bias_arg):
        return self.hex.Conv3d_CustomKernel(sub_kernels, stride, bias_arg)

    def build_maxpool3d(self, kernel_size, stride):
        return self.hex.MaxPool3d(kernel_size, stride)

    def run_ndhwc(self, layer, x_ndhwc):
        out = layer(self.torch.FloatTensor(self._to_ncdhw(x_ndhwc))).detach().numpy()
        return self._to_ndhwc(out)

    def copy_conv3d_kernels(self, src, dst, n):
        with self.torch.no_grad():
            for i in range(n + 1):
                getattr(dst, f"kernel{i}").copy_(getattr(src, f"kernel{i}"))

    # ---- share_neighbors ---------------------------------------------------
    def build_shared_conv2d(self, mode, kernel_size=1, stride=1):
        return self.hex.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            share_neighbors=mode,
        )

    def build_noshare_conv2d(self, kernel_size=1, stride=1):
        return self.hex.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, bias=False)

    def set_shared_weights(self, conv, weights):
        with self.torch.no_grad():
            for g, w in weights.items():
                conv.shared_weights[0, 0, g] = w

    def set_noshare_weights(self, conv, n, weights):
        with self.torch.no_grad():
            for (sub_i, r, c), w in weights.items():
                getattr(conv, f"kernel{sub_i}")[0, 0, r, c] = w

    def run_forward_grid(self, conv, grid):
        x = self.torch.tensor([grid], dtype=self.torch.float32).unsqueeze(
            0
        )  # (1,1,H,W)
        return conv(x)[0, 0].detach().tolist()

    def shared_weight_count(self, conv):
        return int(conv.shared_weights.shape[-1])

    def forward_finite(self, conv, height, width, seed=0):
        x = self.torch.randn(1, 1, height, width)
        out = conv(x)
        return bool(out.isfinite().all()) and tuple(out.shape) == (1, 1, height, width)

    def check_shared_gradients_nonzero(self, conv, grid):
        x = self.torch.tensor([grid], dtype=self.torch.float32).unsqueeze(0)
        conv(x).sum().backward()
        assert conv.shared_weights.grad is not None
        assert (conv.shared_weights.grad.abs() > 0).all()

    def check_noshare_gradients_nonzero(self, conv, grid, n):
        x = self.torch.tensor([grid], dtype=self.torch.float32).unsqueeze(0)
        conv(x).sum().backward()
        for i in range(n + 1):
            k = getattr(conv, f"kernel{i}")
            assert k.grad is not None
            assert (k.grad.abs() > 0).all(), f"kernel{i} should have non-zero gradients"


def _make_backends():
    backends = []
    for cls in (_KerasBackend, _PytorchBackend):
        try:
            backends.append(cls())
        except ImportError:
            pass
    return backends


BACKENDS = _make_backends()
_BACKEND_IDS = [b.name for b in BACKENDS]
backend_param = pytest.mark.parametrize("backend", BACKENDS, ids=_BACKEND_IDS)


def _ones_sub_kernels_2d(in_channels, kernel_size):
    if kernel_size == 1:
        return [
            np.ones((1, in_channels, 3, 1), np.float32),
            np.ones((1, in_channels, 2, 2), np.float32),
        ]
    return [
        np.ones((1, in_channels, 5, 1), np.float32),
        np.ones((1, in_channels, 4, 2), np.float32),
        np.ones((1, in_channels, 3, 2), np.float32),
    ]


def _ones_sub_kernels_3d(in_channels, kernel_size_depth, kernel_size_hex):
    if kernel_size_hex == 1:
        return [
            np.ones((1, in_channels, kernel_size_depth, 3, 1), np.float32),
            np.ones((1, in_channels, kernel_size_depth, 2, 2), np.float32),
        ]
    return [
        np.ones((1, in_channels, kernel_size_depth, 5, 1), np.float32),
        np.ones((1, in_channels, kernel_size_depth, 4, 2), np.float32),
        np.ones((1, in_channels, kernel_size_depth, 3, 2), np.float32),
    ]


# =============================================================================
# Conv2d / Conv2d_CustomKernel hand-verified
# =============================================================================


@backend_param
@pytest.mark.parametrize("in_channels", [1, 5])
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_hand_verified(backend, in_channels, kernel_size, stride, bias):
    bias_value = 1.0 if bias else 0.0
    x = conv2d_input_nhwc(in_channels)
    expected = conv2d_expected(in_channels, kernel_size, stride, bias_value)[
        None, ..., None
    ]

    layer = backend.build_conv2d(in_channels, 1, kernel_size, stride, bias, debug=True)
    out = backend.run_nhwc(layer, x)

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


@backend_param
@pytest.mark.parametrize("in_channels", [1, 5])
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_custom_kernel_hand_verified(
    backend, in_channels, kernel_size, stride, bias
):
    bias_value = 1.0 if bias else 0.0
    bias_arg = np.array([1.0]) if bias else None
    x = conv2d_input_nhwc(in_channels)
    expected = conv2d_expected(in_channels, kernel_size, stride, bias_value)[
        None, ..., None
    ]

    layer = backend.build_conv2d_custom_kernel(
        _ones_sub_kernels_2d(in_channels, kernel_size), stride, bias_arg
    )
    out = backend.run_nhwc(layer, x)

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


@backend_param
@pytest.mark.parametrize("in_channels", [1, 5])
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_custom_kernel_impulse_hand_verified(
    backend, in_channels, kernel_size, stride, bias
):
    """Second, independent hand-verified oracle: a sparse impulse input
    (rather than the sequential grid used above)."""
    bias_value = 1.0 if bias else 0.0
    bias_arg = np.array([1.0]) if bias else None
    x = custom_kernel_input_nhwc(in_channels)
    expected = custom_kernel_expected(in_channels, kernel_size, stride, bias_value)[
        None, ..., None
    ]

    layer = backend.build_conv2d_custom_kernel(
        _ones_sub_kernels_2d(in_channels, kernel_size), stride, bias_arg
    )
    out = backend.run_nhwc(layer, x)

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


# =============================================================================
# Conv3d / Conv3d_CustomKernel hand-verified
# =============================================================================

# (in_channels, depth, kernel_size_depth, kernel_size_hex, stride_depth, stride_hex, bias)
_CONV3D_CASES = [
    (1, 1, 1, 1, 1, 1, False),
    (1, 1, 1, 1, 1, 2, False),
    (1, 1, 1, 1, 1, 3, False),
    (1, 1, 1, 2, 1, 1, False),
    (1, 1, 1, 2, 1, 2, False),
    (1, 1, 1, 2, 1, 3, False),
    (1, 9, 1, 1, 1, 1, False),
    (1, 9, 1, 1, 2, 1, False),
    (1, 9, 1, 1, 3, 1, False),
    (1, 9, 2, 1, 1, 1, False),
    (1, 9, 2, 1, 2, 1, False),
    (1, 9, 2, 1, 2, 2, False),
    (1, 9, 7, 2, 1, 1, False),
    (1, 9, 7, 2, 2, 1, False),
    (5, 9, 7, 2, 1, 1, False),
    (5, 9, 7, 2, 1, 1, True),
]


@backend_param
@pytest.mark.parametrize("in_channels,depth,kd,kh,sd,sh,bias", _CONV3D_CASES)
def test_conv3d_hand_verified(backend, in_channels, depth, kd, kh, sd, sh, bias):
    bias_value = 1.0 if bias else 0.0
    x = conv3d_input_ndhwc(in_channels, depth)
    expected = conv3d_expected(in_channels, depth, kd, kh, sd, sh, bias_value)[
        None, ..., None
    ]

    layer = backend.build_conv3d(in_channels, 1, (kd, kh), (sd, sh), bias, debug=True)
    out = backend.run_ndhwc(layer, x)

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


_CONV3D_CUSTOM_KERNEL_CASES = [
    (1, 1, 1, 1, 1, 1, False),
    (1, 1, 1, 1, 1, 2, False),
    (1, 1, 1, 1, 1, 3, False),
    (1, 1, 1, 2, 1, 1, False),
    (1, 1, 1, 2, 1, 2, False),
    (1, 1, 1, 2, 1, 3, False),
    (1, 9, 1, 1, 1, 1, False),
    (1, 9, 1, 1, 2, 1, False),
    (1, 9, 1, 1, 3, 1, False),
    (1, 9, 2, 1, 1, 1, False),
    (1, 9, 2, 1, 2, 1, False),
    (1, 9, 2, 1, 2, 2, False),
    (1, 9, 7, 2, 1, 1, False),
    (1, 9, 7, 2, 1, 2, False),
    (1, 9, 7, 2, 2, 2, False),
    (5, 9, 3, 2, 1, 1, False),
    (5, 9, 3, 2, 1, 1, True),
]


@backend_param
@pytest.mark.parametrize(
    "in_channels,depth,kd,kh,sd,sh,bias", _CONV3D_CUSTOM_KERNEL_CASES
)
def test_conv3d_custom_kernel_hand_verified(
    backend, in_channels, depth, kd, kh, sd, sh, bias
):
    bias_value = 1.0 if bias else 0.0
    bias_arg = np.array([1.0]) if bias else None
    x = conv3d_input_ndhwc(in_channels, depth)
    expected = conv3d_expected(in_channels, depth, kd, kh, sd, sh, bias_value)[
        None, ..., None
    ]

    kernel = _ones_sub_kernels_3d(in_channels, kd, kh)
    layer = backend.build_conv3d_custom_kernel(kernel, (sd, sh), bias_arg)
    out = backend.run_ndhwc(layer, x)

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


# =============================================================================
# MaxPool2d / MaxPool3d hand-verified
# =============================================================================


@backend_param
@pytest.mark.parametrize("in_channels", [1, 5])
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_maxpool2d_hand_verified(backend, in_channels, kernel_size, stride):
    x = maxpool2d_input_nhwc(in_channels)
    expected = maxpool2d_expected(in_channels, kernel_size, stride)[None, ...]

    layer = backend.build_maxpool2d(kernel_size, stride)
    out = backend.run_nhwc(layer, x)

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


_MAXPOOL3D_CASES = [
    (1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 2),
    (1, 1, 1, 1, 1, 3),
    (1, 1, 1, 2, 1, 1),
    (1, 9, 1, 1, 1, 1),
    (1, 9, 1, 1, 2, 1),
    (1, 9, 1, 1, 3, 1),
    (1, 9, 2, 1, 1, 1),
    (1, 9, 2, 1, 2, 1),
    (1, 9, 2, 1, 2, 2),
    (1, 9, 7, 2, 1, 1),
    (1, 9, 7, 2, 1, 2),
    (1, 9, 7, 2, 2, 2),
    (5, 9, 3, 2, 1, 1),
]


@backend_param
@pytest.mark.parametrize("in_channels,depth,kd,kh,sd,sh", _MAXPOOL3D_CASES)
def test_maxpool3d_hand_verified(backend, in_channels, depth, kd, kh, sd, sh):
    x = conv3d_input_ndhwc(in_channels, depth)
    expected = maxpool3d_expected(in_channels, depth, kd, kh, sd, sh)[None, ...]

    layer = backend.build_maxpool3d((kd, kh), (sd, sh))
    out = backend.run_ndhwc(layer, x)

    np.testing.assert_allclose(out, expected, rtol=5e-4, atol=1e-2)


# =============================================================================
# share_neighbors: pure-Python first-principles oracle against real layers.
# See geometry.py for the parity-probe bug this pins (ring_maps_2d(2) used to
# classify 4 of 19 taps into the wrong ring).
# =============================================================================


@backend_param
class TestRingOracleFirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle(grid, weights, _RING_NEIGHBORS)
        conv = backend.build_shared_conv2d("ring")
        backend.set_shared_weights(conv, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_small_grid(self, backend):
        _check_grid(
            *self._run(backend, GRID_SMALL, WEIGHTS_RING), "ring small", tol=1e-3
        )

    def test_large_grid(self, backend):
        _check_grid(
            *self._run(backend, GRID_LARGE, WEIGHTS_RING), "ring large", tol=1e-3
        )

    def test_weight_count(self, backend):
        assert backend.shared_weight_count(backend.build_shared_conv2d("ring")) == 2

    def test_all_neighbors_same_weight(self, backend):
        # w0=0, w1=1: output[r,c] = number of valid neighbors of (r,c).
        _check_grid(
            *self._run(backend, GRID_LARGE, {0: 0.0, 1: 1.0}), "ring uniform", tol=1e-3
        )


@backend_param
class TestDiagOracleFirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle(grid, weights, _DIAG_NEIGHBORS)
        conv = backend.build_shared_conv2d("diag")
        backend.set_shared_weights(conv, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_small_grid(self, backend):
        _check_grid(
            *self._run(backend, GRID_SMALL, WEIGHTS_DIAG), "diag small", tol=1e-3
        )

    def test_large_grid(self, backend):
        _check_grid(
            *self._run(backend, GRID_LARGE, WEIGHTS_DIAG), "diag large", tol=1e-3
        )

    def test_weight_count(self, backend):
        assert backend.shared_weight_count(backend.build_shared_conv2d("diag")) == 4

    def test_antipodal_symmetry(self, backend):
        # Uniform input, w0=0, all others=1: every interior pixel sees 6 neighbors.
        grid = [[1.0] * 6 for _ in range(7)]
        _, got = self._run(backend, grid, {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0})
        for r in range(1, 6):
            for c in range(1, 5):
                assert abs(got[r][c] - 6.0) < 1e-3, (
                    f"diag uniform [{r},{c}]: got {got[r][c]:.4f}, expected 6.0"
                )

    def test_gradients_flow_to_all_weights(self, backend):
        conv = backend.build_shared_conv2d("diag")
        backend.set_shared_weights(conv, WEIGHTS_DIAG)
        backend.check_shared_gradients_nonzero(conv, GRID_LARGE)


@backend_param
class TestSymOracleFirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle(grid, weights, _SYM_NEIGHBORS)
        conv = backend.build_shared_conv2d("sym")
        backend.set_shared_weights(conv, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_small_grid(self, backend):
        _check_grid(*self._run(backend, GRID_SMALL, WEIGHTS_SYM), "sym small", tol=1e-3)

    def test_large_grid(self, backend):
        _check_grid(*self._run(backend, GRID_LARGE, WEIGHTS_SYM), "sym large", tol=1e-3)

    def test_weight_count(self, backend):
        assert backend.shared_weight_count(backend.build_shared_conv2d("sym")) == 4

    def test_gradients_flow_to_all_weights(self, backend):
        conv = backend.build_shared_conv2d("sym")
        backend.set_shared_weights(conv, WEIGHTS_SYM)
        backend.check_shared_gradients_nonzero(conv, GRID_LARGE)


@backend_param
class TestNoShareOracleFirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle_noshare(grid, weights, _NOSHARE_OFFSETS)
        conv = backend.build_noshare_conv2d()
        backend.set_noshare_weights(conv, 1, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_small_grid(self, backend):
        _check_grid(
            *self._run(backend, GRID_SMALL, WEIGHTS_NOSHARE), "noshare small", tol=1e-3
        )

    def test_large_grid(self, backend):
        _check_grid(
            *self._run(backend, GRID_LARGE, WEIGHTS_NOSHARE), "noshare large", tol=1e-3
        )

    def test_center_only(self, backend):
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE}
        weights[(0, 1, 0)] = 5.0
        _, got = self._run(backend, GRID_LARGE, weights)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - 5.0 * GRID_LARGE[r][c]) < 1e-3

    def test_parity_routing_sub1(self, backend):
        """Cell (1,0,0) reads (-1,-1) for even-column outputs but (0,-1) for
        odd-column outputs -- different input pixels. A wrong parity routing
        (e.g. always using the even-column offset) would fail this."""
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE}
        weights[(1, 0, 0)] = 1.0
        _check_grid(
            *self._run(backend, GRID_LARGE, weights), "parity sub1[r=0,c=0]", tol=1e-3
        )

    def test_gradients_flow(self, backend):
        conv = backend.build_noshare_conv2d()
        backend.set_noshare_weights(conv, 1, WEIGHTS_NOSHARE)
        backend.check_noshare_gradients_nonzero(conv, GRID_LARGE, 1)


@backend_param
class TestRingOracleK2FirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle_k2(grid, weights, _RING2_NEIGHBORS)
        conv = backend.build_shared_conv2d("ring", kernel_size=2)
        backend.set_shared_weights(conv, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_grid(self, backend):
        _check_grid(*self._run(backend, GRID_K2, WEIGHTS_RING2), "ring k=2")

    def test_weight_count(self, backend):
        assert (
            backend.shared_weight_count(
                backend.build_shared_conv2d("ring", kernel_size=2)
            )
            == 3
        )

    def test_gradients_flow(self, backend):
        conv = backend.build_shared_conv2d("ring", kernel_size=2)
        backend.set_shared_weights(conv, WEIGHTS_RING2)
        backend.check_shared_gradients_nonzero(conv, GRID_K2)


@backend_param
class TestDiagOracleK2FirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle_k2(grid, weights, _DIAG2_NEIGHBORS)
        conv = backend.build_shared_conv2d("diag", kernel_size=2)
        backend.set_shared_weights(conv, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_grid(self, backend):
        _check_grid(*self._run(backend, GRID_K2, WEIGHTS_DIAG2), "diag k=2")

    def test_weight_count(self, backend):
        assert (
            backend.shared_weight_count(
                backend.build_shared_conv2d("diag", kernel_size=2)
            )
            == 10
        )

    def test_gradients_flow(self, backend):
        conv = backend.build_shared_conv2d("diag", kernel_size=2)
        backend.set_shared_weights(conv, WEIGHTS_DIAG2)
        backend.check_shared_gradients_nonzero(conv, GRID_K2)


@backend_param
class TestSymOracleK2FirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle_k2(grid, weights, _SYM2_NEIGHBORS)
        conv = backend.build_shared_conv2d("sym", kernel_size=2)
        backend.set_shared_weights(conv, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_grid(self, backend):
        _check_grid(*self._run(backend, GRID_K2, WEIGHTS_SYM2), "sym k=2")

    def test_weight_count(self, backend):
        assert (
            backend.shared_weight_count(
                backend.build_shared_conv2d("sym", kernel_size=2)
            )
            == 10
        )

    def test_gradients_flow(self, backend):
        conv = backend.build_shared_conv2d("sym", kernel_size=2)
        backend.set_shared_weights(conv, WEIGHTS_SYM2)
        backend.check_shared_gradients_nonzero(conv, GRID_K2)


@backend_param
class TestNoShareOracleK2FirstPrinciples:
    def _run(self, backend, grid, weights):
        expected = _oracle_noshare(grid, weights, _NOSHARE2_OFFSETS)
        conv = backend.build_noshare_conv2d(kernel_size=2)
        backend.set_noshare_weights(conv, 2, weights)
        return expected, backend.run_forward_grid(conv, grid)

    def test_grid(self, backend):
        _check_grid(*self._run(backend, GRID_K2, WEIGHTS_NOSHARE2), "noshare k=2")

    def test_center_only(self, backend):
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE2}
        weights[(0, 2, 0)] = 5.0  # sub0 row 2 = the center cell
        _, got = self._run(backend, GRID_K2, weights)
        for r in range(len(GRID_K2)):
            for c in range(len(GRID_K2[0])):
                assert abs(got[r][c] - 5.0 * GRID_K2[r][c]) < 1e-2

    def test_parity_routing_sub1(self, backend):
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE2}
        weights[(1, 1, 0)] = 1.0
        _check_grid(
            *self._run(backend, GRID_K2, weights), "noshare k=2 parity sub1[1,0]"
        )

    def test_gradients_flow(self, backend):
        conv = backend.build_noshare_conv2d(kernel_size=2)
        backend.set_noshare_weights(conv, 2, WEIGHTS_NOSHARE2)
        backend.check_noshare_gradients_nonzero(conv, GRID_K2, 2)


@backend_param
@pytest.mark.parametrize("width", [16, 17])
def test_forward_finite_diag_k2(backend, width):
    conv = backend.build_shared_conv2d("diag", kernel_size=2)
    assert backend.forward_finite(conv, 21, width)


# =============================================================================
# Independent geometry check (no external oracle, no hand-built arrays): a
# debug kernel (all weights = 1, no bias) computes, at each output pixel, the
# SUM of its hex neighbourhood. Feed a single impulse: the output then equals
# 1 exactly at the impulse and at each of its hex neighbours, and the COUNT
# of ones must be the hex neighbourhood size 1 + 3n(n+1). This validates that
# the conv really sums the right hexagonal support, independent of any
# reference implementation -- and independent of framework, hence shared.
# =============================================================================


@backend_param
@pytest.mark.parametrize("n", [1, 2, 3])
def test_impulse_response_hex_neighbourhood_size(backend, n):
    H, W = 15, 15
    x = np.zeros((1, H, W, 1), dtype=np.float32)
    x[0, H // 2, W // 2, 0] = 1.0

    layer = backend.build_conv2d(1, 1, kernel_size=n, stride=1, bias=False, debug=True)
    out = backend.run_nhwc(layer, x)[0, :, :, 0]

    n_hits = int(np.sum(np.isclose(out, 1.0)))
    expected = 1 + 3 * n * (n + 1)  # hex cells within radius n
    clean = np.all(np.isclose(out, 0.0) | np.isclose(out, 1.0))

    assert clean
    assert n_hits == expected


@backend_param
def test_determinism(backend):
    """Same input twice -> identical output (no hidden state / nondeterminism)."""
    rng = np.random.default_rng(23)
    x = rng.standard_normal((2, 12, 9, 2)).astype(np.float32)
    layer = backend.build_conv2d(2, 3, kernel_size=3, stride=2, bias=True, debug=False)
    a = backend.run_nhwc(layer, x)
    b = backend.run_nhwc(layer, x)
    np.testing.assert_array_equal(a, b)


# =============================================================================
# depth_padding="same" on Conv3d: both ports expose the identical API
# (in_channels, out_channels, kernel_size, stride, bias, debug, share_neighbors,
# depth_padding), so its defining property -- output depth == input depth, and
# the result equals manually zero-padding the depth axis then running the
# default "valid" Conv3d -- is shared. Keras-specific graph-mode/eager
# regression tests for this feature stay in test_depth_padding_keras.py: they
# test a keras-only tracing bug, not the depth_padding semantic itself.
# =============================================================================


@backend_param
@pytest.mark.parametrize(
    "share", [False, "ring"]
)  # "ring", not True: pytorch has no bool alias
@pytest.mark.parametrize("D,kd,n", [(9, 5, 1), (9, 3, 2), (7, 5, 1)])
def test_depth_padding_same_preserves_depth(backend, share, D, kd, n):
    rng = np.random.default_rng(40)
    Cin, Cout, H, W = 2, 3, 9, 8
    x = rng.standard_normal((2, D, H, W, Cin)).astype(np.float32)

    layer = backend.build_conv3d(
        Cin,
        Cout,
        (kd, n),
        1,
        True,
        debug=False,
        share_neighbors=share,
        depth_padding="same",
    )
    out = backend.run_ndhwc(layer, x)
    assert out.shape[1] == D


@backend_param
@pytest.mark.parametrize("D,kd,n", [(9, 5, 1), (7, 3, 2)])
def test_depth_padding_same_equals_manual_pad_then_valid(backend, D, kd, n):
    """depth_padding="same" must equal: zero-pad depth by (kd-1)//2 each side
    (centred kernel), then run the default "valid" Conv3d."""
    rng = np.random.default_rng(41)
    Cin, Cout, H, W = 2, 3, 9, 8
    x = rng.standard_normal((1, D, H, W, Cin)).astype(np.float32)

    same_layer = backend.build_conv3d(
        Cin, Cout, (kd, n), 1, False, depth_padding="same"
    )
    out_same = backend.run_ndhwc(same_layer, x)

    valid_layer = backend.build_conv3d(Cin, Cout, (kd, n), 1, False)
    _ = backend.run_ndhwc(
        valid_layer, np.zeros((1, D + kd - 1, H, W, Cin), dtype=np.float32)
    )
    backend.copy_conv3d_kernels(same_layer, valid_layer, n)

    pad = (kd - 1) // 2
    top, bot = pad, kd - 1 - pad
    x_padded = np.pad(x, [(0, 0), (top, bot), (0, 0), (0, 0), (0, 0)])
    out_valid = backend.run_ndhwc(valid_layer, x_padded)

    np.testing.assert_allclose(out_same, out_valid, atol=1e-5)


@backend_param
def test_depth_padding_invalid_value_raises(backend):
    with pytest.raises(ValueError):
        backend.build_conv3d(1, 1, (3, 1), 1, False, depth_padding="bogus")


# =============================================================================
# Edge cases and robustness: numerical behavior that is not framework-API-
# specific (finite output at minimal sizes, dtype handling, debug weights).
# Keras-only edge cases live in test_edge_cases_keras.py instead: construction
# -time ValueError validation for bad kernel_size/stride is a keras-hexagdly
# design decision pytorch-hexagdly does not replicate (confirmed: its Conv2d/
# Conv3d.__init__ assigns kernel_size/stride with no validation at all), and
# the dynamic-shape / in_channels-mismatch tests rely on Keras's symbolic
# Input/build() mechanism, which pytorch has no equivalent of.
# =============================================================================


@backend_param
@pytest.mark.parametrize("kernel_size", [1, 2, 3])
def test_minimum_viable_input_size_stride1(backend, kernel_size):
    """strides=1 only needs H big enough to contain the kernel once."""
    H = 2 * kernel_size + 1
    W = 4
    x = np.random.randn(1, H, W, 1).astype(np.float32)
    layer = backend.build_conv2d(1, 1, kernel_size, 1, False, debug=False)
    out = backend.run_nhwc(layer, x)
    assert np.all(np.isfinite(out))


@backend_param
def test_too_few_rows_for_stride_raises(backend):
    """The library asserts on inputs too small for the requested stride --
    verify it fails loudly rather than returning garbage."""
    layer = backend.build_conv2d(1, 1, 1, 4, False, debug=False)
    x = np.random.randn(1, 1, 4, 1).astype(np.float32)  # H=1, way too small
    with pytest.raises(AssertionError):
        backend.run_nhwc(layer, x)


@backend_param
def test_float64_input_does_not_crash(backend):
    x = np.random.randn(1, 9, 8, 2).astype(np.float64)
    layer = backend.build_conv2d(2, 3, 2, 1, False, debug=False)
    out = backend.run_nhwc(layer, x)
    assert np.all(np.isfinite(out))


@backend_param
@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 3, 5])
def test_no_nan_or_inf_random_inputs(backend, kernel_size, stride):
    rng = np.random.default_rng(99)
    x = (rng.standard_normal((2, 17, 14, 3)) * 1000).astype(np.float32)
    layer = backend.build_conv2d(3, 2, kernel_size, stride, True, debug=False)
    out = backend.run_nhwc(layer, x)
    assert np.all(np.isfinite(out))


@backend_param
def test_conv2d_custom_kernel_bad_subkernel_shape_raises(backend):
    """The first sub-kernel must have exactly 1 column; a malformed one must
    be rejected at construction, not produce a silently wrong convolution."""
    bad = [np.ones((1, 1, 3, 2), np.float32)]  # should have 1 column, not 2
    with pytest.raises(AssertionError):
        backend.build_conv2d_custom_kernel(bad, 1, None)


@backend_param
def test_debug_weights_are_exactly_one(backend):
    layer = backend.build_conv2d(1, 1, 2, 1, False, debug=True)
    _ = backend.run_nhwc(layer, np.zeros((1, 9, 8, 1), dtype=np.float32))
    assert backend.debug_kernels_all_ones(layer, 2)


@backend_param
def test_conv2d_valid_kernel_size_and_stride_still_work(backend):
    """The validation itself must not reject legitimate values."""
    layer = backend.build_conv2d(1, 2, 3, 2, False, debug=False)
    out = backend.run_nhwc(layer, np.zeros((1, 15, 14, 1), dtype=np.float32))
    assert out is not None


@backend_param
def test_conv3d_valid_tuple_kernel_size_and_stride_still_work(backend):
    layer = backend.build_conv3d(1, 2, (3, 1), (2, 1), False, debug=False)
    out = backend.run_ndhwc(layer, np.zeros((1, 5, 9, 8, 1), dtype=np.float32))
    assert out is not None

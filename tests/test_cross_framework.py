"""Cross-checks that keras-hexagdly and pytorch-hexagdly (the two consumers
of this package) agree with each other on share_neighbors ("ring"/"diag"/
"sym"/no-share), bit-for-bit: same random weights copied across frameworks,
same input, outputs compared.

This is the single place that owns "do the two ports agree" -- it used to be
duplicated as `TestVsPytorchHexagdly` inside keras-hexagdly's own
test_share_neighbors.py, reaching into pytorch_hexagdly's private
`_HARDCODED_MAPS` attribute directly. It now checks both ports against this
package's own public `HARDCODED_MAPS` instead.

Requires `keras`, `torch`, `keras_hexagdly`, and `pytorch_hexagdly` all
installed -- see the `cross-framework` extra. The whole module is skipped if
any of them is missing, since there is nothing else in this file to protect
(unlike each consuming repo's own test suite, which has plenty of
framework-only content that must not be skipped just because the other
framework isn't installed).
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
torch = pytest.importorskip("torch")
hgly = pytest.importorskip("keras_hexagdly")
ph = pytest.importorskip("pytorch_hexagdly")

from keras_hexagdly.layers import SHARE_NEIGHBORS_MODES, weight_maps_2d  # noqa: E402

from hexagdly_oracle import HARDCODED_MAPS  # noqa: E402

_RTOL, _ATOL = 1e-4, 1e-4


@pytest.fixture(autouse=True)
def _force_cpu():
    """Keeps this file testing algorithm correctness, not GPU-vs-CPU float
    reduction-order noise (see keras-hexagdly's test_vs_pytorch_hexagdly.py
    for the same rationale)."""
    with keras.device("cpu"):
        yield


def _to_nhwc(x_nchw):
    return np.transpose(x_nchw, (0, 2, 3, 1))


def _to_ndhwc(x_ncdhw):
    return np.transpose(x_ncdhw, (0, 2, 3, 4, 1))


def _copy_shared_weights_2d(torch_layer, keras_layer):
    # pytorch shared_weights: (out, in, num_shared) -> keras ring_weights: (num_shared, in, out)
    w = torch_layer.shared_weights.detach().numpy()
    keras_layer.ring_weights.assign(np.transpose(w, (2, 1, 0)))
    if getattr(torch_layer, "bias", False):
        keras_layer.bias_tensor.assign(torch_layer.kwargs["bias"].detach().numpy())


def _copy_shared_weights_3d(torch_layer, keras_layer):
    # pytorch: (out, in, depth, num_shared) -> keras: (depth, num_shared, in, out)
    w = torch_layer.shared_weights.detach().numpy()
    keras_layer.ring_weights.assign(np.transpose(w, (2, 3, 1, 0)))
    if getattr(torch_layer, "bias", False):
        keras_layer.bias_tensor.assign(torch_layer.kwargs["bias"].detach().numpy())


def _copy_noshare_weights_2d(torch_layer, keras_layer):
    # pytorch kernel{i}: (out, in, kh, kw) -> keras base_kernel{i}: (kh, kw, in, out)
    for i in range(torch_layer.hexbase_size + 1):
        w = getattr(torch_layer, "kernel" + str(i)).detach().numpy()
        keras_layer._base_kernels[i].assign(np.transpose(w, (2, 3, 1, 0)))
    if getattr(torch_layer, "bias", False):
        keras_layer.bias_tensor.assign(torch_layer.kwargs["bias"].detach().numpy())


def _copy_noshare_weights_3d(torch_layer, keras_layer):
    # pytorch kernel{i}: (out, in, depth, kh, kw) -> keras base_kernel{i}: (depth, kh, kw, in, out)
    for i in range(torch_layer.hexbase_size + 1):
        w = getattr(torch_layer, "kernel" + str(i)).detach().numpy()
        keras_layer._base_kernels[i].assign(np.transpose(w, (2, 3, 4, 1, 0)))
    if getattr(torch_layer, "bias", False):
        keras_layer.bias_tensor.assign(torch_layer.kwargs["bias"].detach().numpy())


def _assert_close(torch_out_nchw, keras_out_nhwc, dims):
    t = torch_out_nchw.detach().cpu().numpy()
    k = keras.ops.convert_to_numpy(keras_out_nhwc)
    k = np.transpose(k, (0, 3, 1, 2)) if dims == 2 else np.transpose(k, (0, 4, 1, 2, 3))
    assert t.shape == k.shape
    np.testing.assert_allclose(t, k, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("mode", SHARE_NEIGHBORS_MODES)
@pytest.mark.parametrize("kernel_size", [1, 2])
def test_weight_map_matches_shared_hardcoded_table(mode, kernel_size):
    """Structural check, no forward pass: pins keras_hexagdly's weight_maps_2d
    directly against this package's HARDCODED_MAPS (previously checked
    against pytorch_hexagdly's private _HARDCODED_MAPS attribute instead)."""
    keras_maps, keras_num = weight_maps_2d(kernel_size, mode)
    torch_maps, torch_num = HARDCODED_MAPS[(mode, kernel_size, 0)]
    assert keras_num == torch_num
    assert len(keras_maps) == len(torch_maps)
    for i, (km, tm) in enumerate(zip(keras_maps, torch_maps)):
        np.testing.assert_array_equal(
            km, tm, err_msg=f"sub-kernel {i} mismatch for mode={mode} n={kernel_size}"
        )


def test_weight_map_both_parities_identical_upstream():
    for mode in SHARE_NEIGHBORS_MODES:
        for n in (1, 2):
            even, _ = HARDCODED_MAPS[(mode, n, 0)]
            odd, _ = HARDCODED_MAPS[(mode, n, 1)]
            for e, o in zip(even, odd):
                np.testing.assert_array_equal(
                    e, o, err_msg=f"mode={mode} n={n}: parity maps differ upstream"
                )


@pytest.mark.parametrize("mode", SHARE_NEIGHBORS_MODES)
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("H,W,Cin,Cout", [(9, 8, 2, 3), (12, 7, 1, 4)])
def test_conv2d_share_neighbors(mode, kernel_size, stride, H, W, Cin, Cout):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)
    tl = ph.Conv2d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=mode,
    )
    kl = hgly.Conv2d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=mode,
    )
    _ = kl(keras.ops.zeros((1, H, W, Cin)))

    with torch.no_grad():
        tl.shared_weights.copy_(
            torch.from_numpy(
                rng.standard_normal(tuple(tl.shared_weights.shape)).astype(np.float32)
            )
        )
        tl.kwargs["bias"].copy_(
            torch.from_numpy(rng.standard_normal(Cout).astype(np.float32))
        )
    _copy_shared_weights_2d(tl, kl)

    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("H,W,Cin,Cout", [(9, 8, 2, 3), (12, 7, 1, 4)])
def test_conv2d_noshare(kernel_size, stride, H, W, Cin, Cout):
    """share_neighbors=False, cross-checked via the real forward pass -- needs
    no assumption about how the offset table behaves across strides (unlike a
    stride>1 first-principles oracle built from stride=1-derived tables would;
    see hexagdly-oracle's README "Scope: stride=1 only")."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, Cin, H, W)).astype(np.float32)
    tl = ph.Conv2d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=None,
    )
    kl = hgly.Conv2d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=False,
    )
    _ = kl(keras.ops.zeros((1, H, W, Cin)))

    with torch.no_grad():
        for i in range(kernel_size + 1):
            k = getattr(tl, "kernel" + str(i))
            k.copy_(
                torch.from_numpy(rng.standard_normal(tuple(k.shape)).astype(np.float32))
            )
        tl.kwargs["bias"].copy_(
            torch.from_numpy(rng.standard_normal(Cout).astype(np.float32))
        )
    _copy_noshare_weights_2d(tl, kl)

    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_nhwc(x)))
    _assert_close(to, ko, 2)


@pytest.mark.parametrize("mode", SHARE_NEIGHBORS_MODES)
@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_conv3d_share_neighbors(mode, kernel_size, stride):
    rng = np.random.default_rng(1)
    D, H, W, Cin, Cout = 5, 9, 8, 2, 3
    x = rng.standard_normal((2, Cin, D, H, W)).astype(np.float32)
    tl = ph.Conv3d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=mode,
    )
    kl = hgly.Conv3d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=mode,
    )
    _ = kl(keras.ops.zeros((1, D, H, W, Cin)))

    with torch.no_grad():
        tl.shared_weights.copy_(
            torch.from_numpy(
                rng.standard_normal(tuple(tl.shared_weights.shape)).astype(np.float32)
            )
        )
        tl.kwargs["bias"].copy_(
            torch.from_numpy(rng.standard_normal(Cout).astype(np.float32))
        )
    _copy_shared_weights_3d(tl, kl)

    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    _assert_close(to, ko, 3)


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_conv3d_noshare(kernel_size, stride):
    rng = np.random.default_rng(3)
    D, H, W, Cin, Cout = 5, 9, 8, 2, 3
    x = rng.standard_normal((2, Cin, D, H, W)).astype(np.float32)
    tl = ph.Conv3d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=None,
    )
    kl = hgly.Conv3d(
        Cin,
        Cout,
        kernel_size=kernel_size,
        stride=stride,
        bias=True,
        share_neighbors=False,
    )
    _ = kl(keras.ops.zeros((1, D, H, W, Cin)))

    with torch.no_grad():
        for i in range(kernel_size + 1):
            k = getattr(tl, "kernel" + str(i))
            k.copy_(
                torch.from_numpy(rng.standard_normal(tuple(k.shape)).astype(np.float32))
            )
        tl.kwargs["bias"].copy_(
            torch.from_numpy(rng.standard_normal(Cout).astype(np.float32))
        )
    _copy_noshare_weights_3d(tl, kl)

    to = tl(torch.from_numpy(x))
    ko = kl(keras.ops.convert_to_tensor(_to_ndhwc(x)))
    _assert_close(to, ko, 3)

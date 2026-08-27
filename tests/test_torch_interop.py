"""Loading PyTorch hexagonal-conv weights into the Keras layers.

Covers both PyTorch implementations -- upstream `hexagdly` (no weight sharing)
and this project's `pytorch-hexagdly` fork (ring/diag/sym) -- and both input
forms: live torch tensors and plain numpy restored from an .npz, which is what
lets an export machine avoid depending on torch at all.

A mis-transposed weight loads without complaint and only shows up as degraded
accuracy, so every test here compares against the PyTorch forward pass rather
than just checking shapes.
"""

import numpy as np
import pytest

import keras
import keras_hexagdly as hgly
from keras_hexagdly.torch_interop import (
    load_hex_conv2d_weights,
    load_torch_state_dict,
    to_numpy,
    verify_against,
)

torch = pytest.importorskip("torch")
pth = pytest.importorskip("pytorch_hexagdly")

CIN, COUT, H, W = 3, 4, 9, 9


def _keras_model(share=False, kernel_size=1, bias=True):
    inp = keras.Input(shape=(H, W, CIN), name="image")
    out = hgly.Conv2d(
        COUT,
        kernel_size=kernel_size,
        strides=1,
        use_bias=bias,
        share_neighbors=share,
        name="conv",
    )(inp)
    return keras.Model(inp, out)


def _torch_layer(share=None, kernel_size=1, bias=True):
    layer = pth.Conv2d(
        in_channels=CIN,
        out_channels=COUT,
        kernel_size=kernel_size,
        stride=1,
        bias=bias,
        share_neighbors=share,
    )
    g = torch.Generator().manual_seed(0)
    for p in layer.parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return layer


def _agree(torch_layer, keras_model, x_nchw, atol=1e-5):
    y_t = torch_layer(torch.from_numpy(x_nchw)).detach().numpy()
    y_k = np.asarray(keras_model.predict(np.transpose(x_nchw, (0, 2, 3, 1)), verbose=0))
    return float(np.max(np.abs(np.transpose(y_t, (0, 2, 3, 1)) - y_k)))


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_unshared_roundtrip_matches_pytorch(kernel_size, bias):
    tl = _torch_layer(None, kernel_size, bias)
    km = _keras_model(False, kernel_size, bias)
    load_hex_conv2d_weights(km.get_layer("conv"), tl.state_dict(), prefix="")
    x = np.random.default_rng(0).standard_normal((2, CIN, H, W)).astype(np.float32)
    assert _agree(tl, km, x) < 1e-5


@pytest.mark.parametrize("mode", ["ring", "diag", "sym"])
def test_shared_modes_match_pytorch(mode):
    """The fork's shared_weights (out, in, num_shared) must land in Keras'
    ring_weights (num_shared, in, out) with the groups in the same order."""
    tl = _torch_layer(mode, kernel_size=1)
    km = _keras_model(mode, kernel_size=1)
    load_hex_conv2d_weights(km.get_layer("conv"), tl.state_dict(), prefix="")
    x = np.random.default_rng(1).standard_normal((2, CIN, H, W)).astype(np.float32)
    assert _agree(tl, km, x) < 1e-5


def test_numpy_path_is_identical_to_torch_path(tmp_path):
    """An .npz restored with numpy must load exactly like the live tensors --
    this is what lets the export side drop torch entirely."""
    tl = _torch_layer(None, kernel_size=2)
    sd = tl.state_dict()

    km_torch = _keras_model(False, kernel_size=2)
    load_hex_conv2d_weights(km_torch.get_layer("conv"), sd, prefix="")

    npz = tmp_path / "w.npz"
    np.savez(npz, **{k: v.detach().cpu().numpy() for k, v in sd.items()})
    sd_np = dict(np.load(npz))
    assert all(isinstance(v, np.ndarray) for v in sd_np.values())

    km_numpy = _keras_model(False, kernel_size=2)
    load_hex_conv2d_weights(km_numpy.get_layer("conv"), sd_np, prefix="")

    x = np.random.default_rng(2).standard_normal((2, H, W, CIN)).astype(np.float32)
    assert np.array_equal(
        km_torch.predict(x, verbose=0), km_numpy.predict(x, verbose=0)
    )


def test_upstream_hexagdly_checkpoint_loads():
    """Upstream `hexagdly` names its parameters exactly like the fork does in
    the unshared case, so its checkpoints must load unchanged."""
    up = pytest.importorskip("hexagdly")
    tl = up.Conv2d(
        in_channels=CIN, out_channels=COUT, kernel_size=1, stride=1, bias=True
    )
    g = torch.Generator().manual_seed(3)
    for p in tl.parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    km = _keras_model(False, 1, True)
    load_hex_conv2d_weights(km.get_layer("conv"), tl.state_dict(), prefix="")
    x = np.random.default_rng(3).standard_normal((2, CIN, H, W)).astype(np.float32)
    assert _agree(tl, km, x) < 1e-5


def test_shape_mismatch_is_reported_not_silent():
    tl = _torch_layer(None, kernel_size=1)
    km = _keras_model(False, kernel_size=2)  # deliberately different geometry
    with pytest.raises((ValueError, KeyError)):
        load_hex_conv2d_weights(km.get_layer("conv"), tl.state_dict(), prefix="")


def test_unmapped_weighted_layer_raises():
    """A layer left out of `mapping` would silently keep its random init."""
    inp = keras.Input(shape=(H, W, CIN))
    x = hgly.Conv2d(COUT, kernel_size=1, strides=1, use_bias=True, name="conv")(inp)
    out = keras.layers.Dense(2, name="head")(x)
    model = keras.Model(inp, out)
    tl = _torch_layer(None, 1)
    with pytest.raises(ValueError, match="random initialisation"):
        load_torch_state_dict(model, tl.state_dict(), mapping={"conv": ""})


def test_verify_against_catches_a_wrong_layout():
    """verify_against is the guard the docs call non-optional -- it must FAIL on
    a model whose weights were left in PyTorch's layout, and pass once they are
    transposed correctly."""
    tl = _torch_layer(None, kernel_size=1)
    km = _keras_model(False, kernel_size=1)
    load_hex_conv2d_weights(km.get_layer("conv"), tl.state_dict(), prefix="")
    x = np.random.default_rng(4).standard_normal((2, CIN, H, W)).astype(np.float32)
    assert verify_against(km, tl, x) < 1e-4

    # Now make the realistic mistake: reshape the PyTorch weight into the Keras
    # shape without transposing it. Same element count, scrambled meaning --
    # exactly what verify_against exists to catch.
    raw = tl.state_dict()["kernel0"].detach().numpy()
    target = km.get_layer("conv")._base_kernels[0]
    km.get_layer("conv")._base_kernels[0].assign(raw.reshape(target.shape))
    with pytest.raises(AssertionError, match="disagree"):
        verify_against(km, tl, x)


def test_to_numpy_accepts_both_forms():
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert np.array_equal(to_numpy(t), np.arange(6, dtype=np.float32).reshape(2, 3))
    a = np.ones((2, 2), dtype=np.float32)
    assert to_numpy(a) is a or np.array_equal(to_numpy(a), a)

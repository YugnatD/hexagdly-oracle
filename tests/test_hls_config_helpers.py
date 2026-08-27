"""hls4ml configuration helpers: reuse-factor sizing and misconfiguration checks.

Both guard against traps that produced silently wrong measurements before they
existed -- see docs/rom-investigation/NOTES.md and docs/pytorch/NOTES_LATENCY.md
in keras-hexagdly:

  * config_from_keras_model(granularity='name') writes ReuseFactor=1 into every
    layer it recognises, and that overrides Model.ReuseFactor. Hex layers get no
    entry, so they inherit the model-level value instead -- two different reuse
    factors in one model, with nothing saying so.
  * Strategy='Latency' + ReuseFactor=1 + Flatten downstream mis-synthesises:
    correct in C-simulation, bias-only constant in real RTL cosimulation.
  * MAC_BLOCK past ~512 makes Vitis duplicate the weight ROM instead of
    splitting it: measured 0 BRAM at 512 against 6080 at 1216.
"""

import warnings

import keras
import pytest

import keras_hexagdly as hgly
from keras_hexagdly.hls4ml_ext import (
    check_hls_config,
    hex_reuse_config,
    patch_model_for_hls,
    recommended_reuse_factor,
)

hls4ml = pytest.importorskip("hls4ml")

from keras_hexagdly.hls4ml_handler import register_hex_gather_layers  # noqa: E402

register_hex_gather_layers()  # config_from_keras_model needs the custom handlers


def _patched(kernel_size=2, cin=16, cout=32, hw=12, flatten=True):
    inp = keras.Input(shape=(hw, hw, cin), name="image")
    x = hgly.Conv2d(
        filters=cout,
        kernel_size=kernel_size,
        strides=2,
        share_neighbors=False,
        use_bias=True,
        name="conv",
    )(inp)
    out = keras.layers.Flatten(name="flat")(x) if flatten else x
    return patch_model_for_hls(keras.Model(inp, out), strategy="linebuffer")


def _hex_layers(model):
    return [layer for layer in model.layers if hasattr(layer, "mac_weights")]


def test_recommended_reuse_factor_caps_mac_block():
    model = _patched()
    for layer in _hex_layers(model):
        n_rows, cin, cout = (int(d) for d in layer.mac_weights.shape)
        rf = recommended_reuse_factor(layer)
        mac_block = -(-(n_rows * cin) // rf) * cout
        assert mac_block <= 512, f"{layer.name}: MAC_BLOCK={mac_block}"
        # and it should be the SMALLEST such factor -- one less must overflow
        if rf > 1:
            smaller = -(-(n_rows * cin) // (rf - 1)) * cout
            assert smaller > 512


def test_recommended_reuse_factor_ignores_plain_layers():
    inp = keras.Input(shape=(8, 8, 4))
    dense = keras.layers.Dense(3)
    keras.Model(inp, dense(keras.layers.Flatten()(inp)))
    assert recommended_reuse_factor(dense) == 1


def test_check_flags_known_bad_rtl_combination():
    model = _patched()
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vivado"
    )
    cfg["Model"]["Strategy"] = "Latency"
    cfg["Model"]["ReuseFactor"] = 1
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        issues = check_hls_config(cfg, model)
    assert any("Latency" in i and "WRONG RTL" in i for i in issues)
    assert any("MAC_BLOCK" in i for i in issues)


def test_check_is_quiet_without_a_reshape_downstream():
    """The RTL failure needs a repack_stream-triggering layer; without one the
    Latency/RF=1 combination is legitimate and must not be flagged."""
    model = _patched(flatten=False)
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vivado"
    )
    cfg["Model"]["Strategy"] = "Latency"
    cfg["Model"]["ReuseFactor"] = 1
    hex_reuse_config(cfg, model, verbose=False)  # keep MAC_BLOCK in range
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        issues = check_hls_config(cfg, model)
    assert not any("WRONG RTL" in i for i in issues), issues


def test_hex_reuse_config_clears_every_issue():
    model = _patched()
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vivado"
    )
    cfg["Model"]["Strategy"] = "Latency"
    cfg["Model"]["ReuseFactor"] = 1
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        hex_reuse_config(cfg, model, verbose=False)
        assert check_hls_config(cfg, model) == []
    # and it must write an explicit per-layer entry, not rely on Model-level
    for layer in _hex_layers(model):
        assert "ReuseFactor" in cfg["LayerName"][layer.name]


def test_hex_reuse_config_leaves_recognised_layers_alone():
    model = _patched()
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vivado"
    )
    hex_names = {layer.name for layer in _hex_layers(model)}
    before = {n: dict(d) for n, d in cfg["LayerName"].items() if n not in hex_names}
    hex_reuse_config(cfg, model, verbose=False)
    for name, entry in before.items():
        assert cfg["LayerName"][name] == entry


def _sane_config(model):
    """A config hex_reuse_config has already made valid, so the only thing left
    for a test to observe is the guard itself."""
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vivado"
    )
    cfg["Model"]["Strategy"] = "Resource"
    cfg["Model"]["ReuseFactor"] = 2
    hex_reuse_config(cfg, model, verbose=False)
    return cfg


class TestSupportGuards:
    """The 2D io_stream path is the one covered by RTL cosimulation; the others
    pass C-simulation but are not something to build on. The guards make that
    refusal explicit rather than leaving it to the README."""

    def test_3d_layers_are_refused_by_default(self):
        inp = keras.Input(shape=(4, 8, 8, 2))
        model = keras.Model(inp, hgly.Conv3d(4, kernel_size=1)(inp))
        with pytest.raises(NotImplementedError, match="not a validated export path"):
            patch_model_for_hls(model)

    def test_3d_maxpool_is_refused_too(self):
        inp = keras.Input(shape=(4, 8, 8, 2))
        model = keras.Model(inp, hgly.MaxPool3d(kernel_size=1)(inp))
        with pytest.raises(NotImplementedError):
            patch_model_for_hls(model)

    def test_3d_opt_out_works(self):
        """The escape hatch has to actually work -- the 3D path passes csim, so
        refusing it outright would remove usable (if unvalidated) function."""
        inp = keras.Input(shape=(4, 8, 8, 2))
        model = keras.Model(inp, hgly.Conv3d(4, kernel_size=1)(inp))
        assert patch_model_for_hls(model, allow_unvalidated=True) is not None

    def test_2d_is_not_affected(self):
        assert _patched() is not None

    def test_io_parallel_is_refused_by_default(self):
        model = _patched()
        with pytest.raises(NotImplementedError, match="io_parallel|not a validated"):
            check_hls_config({"Model": {}}, model, io_type="io_parallel")

    def test_io_parallel_opt_out_works(self):
        """The opt-out must let the call through; it may still report the usual
        configuration warnings, which is a separate concern."""
        model = _patched()
        cfg = _sane_config(model)
        assert (
            check_hls_config(cfg, model, io_type="io_parallel", allow_unvalidated=True)
            == []
        )

    def test_io_stream_is_the_default_and_passes(self):
        model = _patched()
        cfg = _sane_config(model)
        assert check_hls_config(cfg, model) == []

    def test_guard_messages_carry_the_measurement(self):
        """Someone hitting these should learn why, not just that."""
        inp = keras.Input(shape=(4, 8, 8, 2))
        model = keras.Model(inp, hgly.Conv3d(4, kernel_size=1)(inp))
        with pytest.raises(NotImplementedError) as e:
            patch_model_for_hls(model)
        assert "cosimulation" in str(e.value) and "min" in str(e.value)

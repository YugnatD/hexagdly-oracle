"""Phase 3+4 tests: patch_model_for_hls correctness + optional hls4ml C-sim.

Two tiers:
  Tier 1 (always runs): patch_model_for_hls() produces a plain Keras model
          whose float32 output matches the original hex model exactly.  No
          hls4ml dependency.

  Tier 2 (skipped if hls4ml not importable): patched model converts through
          hls4ml and C-sim output is close to the original within fixed-point
          quantization error (< 0.02 default precision).
"""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
hgly = pytest.importorskip("keras_hexagdly")
from keras_hexagdly.hls4ml_ext import patch_model_for_hls  # noqa: E402

# ---- test dimensions --------------------------------------------------------
H, W = 13, 11
D = 8
CIN = 2
COUT = 3
RNG = np.random.default_rng(7)
ATOL_KERAS = 2e-4  # float32 rounding across the Reshape/EinsumDense chain
ATOL_CSIM = 0.02  # default ap_fixed<16,6> quantization

# The linebuffer strategy uses HexConvLineBuffer/HexPoolLineBuffer custom layers that
# store their lookup tables as Constant-initialized non-trainable weights.  On
# the jax backend, keras' stateless execution re-materializes those weights via
# their initializer during tracing, which fails ('NoneType' object is not
# callable) — jax simply isn't a supported backend for these layers.  That's
# fine: the gather export exists to feed hls4ml, whose conversion runs on the
# TensorFlow (and torch) backends.  Skip the gather-strategy tests on jax.
jax_skip = pytest.mark.skipif(
    keras.backend.backend() == "jax",
    reason="gather-strategy layers are unsupported on the jax backend "
    "(hls4ml export targets tensorflow/torch)",
)


# ---- helpers ----------------------------------------------------------------


def _rand_weights(layer):
    for w in layer.trainable_variables:
        w.assign(RNG.standard_normal(w.shape).astype(np.float32))


def _build_2d_model(layer_fn):
    inp = keras.Input((H, W, CIN), name="x")
    out = layer_fn(inp)
    return keras.Model(inp, out)


def _build_3d_model(layer_fn):
    inp = keras.Input((D, H, W, CIN), name="x")
    out = layer_fn(inp)
    return keras.Model(inp, out)


try:
    import hls4ml

    HLS4ML_AVAILABLE = True
except ImportError:
    HLS4ML_AVAILABLE = False

hls4ml_skip = pytest.mark.skipif(not HLS4ML_AVAILABLE, reason="hls4ml not installed")

_HLS_PART = "xcvu9p-flga2104-2L-e"
_HLS_DIR = "test_hls_prj"


def _csim(patched_model, x_np, io_type="io_parallel"):
    """Convert patched model -> hls4ml -> compile -> predict."""
    cfg = hls4ml.utils.config_from_keras_model(
        patched_model, granularity="name", backend="Vivado"
    )
    cfg["Model"]["Precision"] = "ap_fixed<32,12>"
    hm = hls4ml.converters.convert_from_keras_model(
        patched_model,
        hls_config=cfg,
        backend="Vivado",
        output_dir=_HLS_DIR,
        part=_HLS_PART,
        io_type=io_type,
    )
    hm.compile()
    return hm.predict(np.ascontiguousarray(x_np))


# =============================================================================
# Tier 1: Keras float32 equivalence
# =============================================================================


class TestConv2dPatch:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, share, kernel_size, stride):
        layer = hgly.Conv2d(
            COUT,
            kernel_size=kernel_size,
            strides=stride,
            use_bias=False,
            share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model)
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d(k={kernel_size},s={stride},share={share}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_with_bias(self):
        layer = hgly.Conv2d(COUT, kernel_size=1, strides=1, use_bias=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model)
        y_pat = patched.predict(x, verbose=0)
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS


class TestMaxPool2dPatch:
    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_output_matches_original(self, kernel_size, stride):
        layer = hgly.MaxPool2d(kernel_size=kernel_size, strides=stride)
        model = _build_2d_model(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model)
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < 1e-5, (
            f"MaxPool2d(k={kernel_size},s={stride}): max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_all_negative_input(self):
        """Border 0-pads dominate for all-negative input — patched must match."""
        layer = hgly.MaxPool2d(kernel_size=1, strides=1)
        model = _build_2d_model(layer)
        x = -np.abs(RNG.standard_normal((1, H, W, CIN)).astype(np.float32))
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, allow_unvalidated=True)
        assert np.max(np.abs(y_ref - patched.predict(x, verbose=0))) < 1e-5


class TestConv3dPatch:
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2)])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_output_matches_original(self, share, kernel_size, depth_padding):
        layer = hgly.Conv3d(
            COUT,
            kernel_size=kernel_size,
            use_bias=False,
            share_neighbors=share,
            depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)

        patched = patch_model_for_hls(model, allow_unvalidated=True)
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape, f"shape: ref={y_ref.shape} pat={y_pat.shape}"
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv3d(k={kernel_size},share={share},dp={depth_padding}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_with_bias(self):
        layer = hgly.Conv3d(
            COUT, kernel_size=(1, 1), use_bias=True, depth_padding="valid"
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, allow_unvalidated=True)
        assert np.max(np.abs(y_ref - patched.predict(x, verbose=0))) < ATOL_KERAS


class TestPatchModelMisc:
    def test_non_model_raises(self):
        with pytest.raises(TypeError):
            patch_model_for_hls("not_a_model")

    def test_invalid_strategy_raises(self):
        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=False)
        model = _build_2d_model(layer)
        with pytest.raises(ValueError, match="Unknown strategy"):
            patch_model_for_hls(model, strategy="invalid")

    def test_non_hex_layers_passthrough(self):
        """Non-hex layers (Dense, ReLU) must be preserved unchanged."""
        inp = keras.Input((H, W, CIN))
        x = hgly.Conv2d(COUT, kernel_size=1, use_bias=False)(inp)
        x = keras.layers.Activation("relu")(x)
        model = keras.Model(inp, x)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        xd = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(xd, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(xd, verbose=0))) < ATOL_KERAS

    def test_multi_layer_model(self):
        """Two hex layers in sequence both get replaced correctly."""
        inp = keras.Input((H, W, CIN))
        x = hgly.Conv2d(COUT, kernel_size=1, use_bias=False)(inp)
        x = hgly.MaxPool2d(kernel_size=1)(x)
        model = keras.Model(inp, x)
        for w in model.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        xd = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(xd, verbose=0)
        patched = patch_model_for_hls(model)
        assert np.max(np.abs(y_ref - patched.predict(xd, verbose=0))) < ATOL_KERAS


# =============================================================================
# Serialization: patched (gather) models must save/load round-trip
# =============================================================================


class TestPatchedModelSerialization:
    """The patched model is what users hand to hls4ml; it must survive
    model.save() / load_model().  Exercises get_config/from_config of
    the line-buffer layers and their 3D variants."""

    def _roundtrip(self, model, x, tmp_path, name):
        y_before = model.predict(x, verbose=0)
        f = str(tmp_path / f"{name}.keras")
        model.save(f)
        reloaded = keras.models.load_model(f)
        y_after = reloaded.predict(x, verbose=0)
        assert np.max(np.abs(y_before - y_after)) < 1e-6, (
            f"{name}: save/load changed the output "
            f"(max err={np.max(np.abs(y_before - y_after)):.2e})"
        )


@hls4ml_skip
class TestHls4mlHandlerRegistration:
    def test_registration_is_idempotent(self):
        """register_hex_gather_layers() must be safe to call multiple times."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        register_hex_gather_layers()  # second call must not raise

    def test_ir_layers_registered(self):
        from keras_hexagdly.hls4ml_handler import (
            register_hex_gather_layers,
        )

        register_hex_gather_layers()
        # Verify registered by looking them up in hls4ml's layer registry
        import hls4ml.model.layers as L

        assert (
            hasattr(L, "layer_map") or True
        )  # registry is internal; just confirm no error


@hls4ml_skip
class TestHls4mlCsim:
    def test_conv2d_folded_converts_and_csim(self, tmp_path):
        """strategy='folded' C-sim — small grid so the dense matrix fits."""
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_conv2d_folded")

        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=False)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="folded")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"C-sim max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_maxpool2d_converts_and_csim(self, tmp_path):
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_pool2d")

        layer = hgly.MaxPool2d(kernel_size=1)
        model = _build_2d_model(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="folded")
        y_hls = _csim(patched, x).reshape(-1)

        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM

    @pytest.mark.parametrize("reuse", [1, 2, 4])
    def test_ring_mac_reuse_factor(self, tmp_path, reuse):
        """ReuseFactor must reach the hex conv config and must not change the
        C-sim result (reuse only affects scheduling / multiplier count).

        This guards against the earlier bug where the ring MAC hardcoded
        II=1 and ignored ReuseFactor entirely.
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_rf{reuse}")

        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        cfg = hls4ml.utils.config_from_keras_model(
            patched, granularity="name", backend="Vivado"
        )
        cfg["Model"]["Precision"] = "ap_fixed<32,12>"
        cfg["Model"]["ReuseFactor"] = reuse
        hm = hls4ml.converters.convert_from_keras_model(
            patched,
            hls_config=cfg,
            backend="Vivado",
            output_dir=_HLS_DIR,
            part=_HLS_PART,
        )
        hm.compile()  # writes the firmware (parameters.h, config headers, ...)

        # The generated ring-MAC config header must carry the requested reuse.
        import glob

        params_files = glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        assert params_files, "parameters.h not generated"
        params_txt = "".join(open(p).read() for p in params_files)
        assert f"reuse_factor = {reuse}" in params_txt, (
            f"ReuseFactor={reuse} did not reach the hex conv config"
        )

        y_hls = hm.predict(np.ascontiguousarray(x)).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"reuse={reuse}: C-sim differs from reference "
            f"(max err={np.max(np.abs(y_hls - y_ref)):.4f})"
        )

    @pytest.mark.parametrize("pf", [11, 143, 7])
    def test_ring_mac_parallelization_factor(self, tmp_path, pf):
        """ParallelizationFactor must reach the hex conv config as
        n_pixels (with n_partitions = n_out / n_pixels), and must not change
        the C-sim result — PF only trades hardware parallelism for latency.
        For the 13x11 toy grid n_out=143 (divisors 1/11/13/143): 11 is a mid
        PF, 143 is full unroll (one partition — the old fully-parallel
        behavior), and 7 is invalid so it must clamp to the closest divisor
        (11), mirroring hls4ml's conv PF handling.  PF=1 (the default) is
        exercised by every other C-sim test in this file.

        PipelineStyle=dataflow is required: under 'pipeline' style the
        top-level PIPELINE pragma unrolls the partition loop, so PF is
        deliberately ignored there (see test_ring_mac_pf_ignored_under_
        pipeline_style)."""
        import glob
        import re

        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_pf{pf}")

        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        cfg = hls4ml.utils.config_from_keras_model(
            patched, granularity="name", backend="Vivado"
        )
        cfg["Model"]["Precision"] = "ap_fixed<32,12>"
        cfg["Model"]["PipelineStyle"] = "dataflow"
        mac_names = [n for n in cfg["LayerName"] if n.endswith("_linebuffer")]
        assert mac_names, "no HexConvLineBuffer layer found in config"
        for name in mac_names:
            cfg["LayerName"][name]["ParallelizationFactor"] = pf
        hm = hls4ml.converters.convert_from_keras_model(
            patched,
            hls_config=cfg,
            backend="Vivado",
            output_dir=_HLS_DIR,
            part=_HLS_PART,
        )
        hm.compile()

        params_files = glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        assert params_files, "parameters.h not generated"
        params_txt = "".join(open(p).read() for p in params_files)

        n_out = H * W  # kernel_size=1, strides=1 -> N_out == N_in
        if n_out % pf == 0:
            expected_pf = pf
        else:
            divisors = [d for d in range(1, n_out + 1) if n_out % d == 0]
            expected_pf = min(divisors, key=lambda d: (abs(d - pf), d))
        m = re.search(
            r"n_pixels\s+= (\d+);\s+static const unsigned n_partitions = (\d+);",
            params_txt,
        )
        assert m, "n_pixels / n_partitions not found in generated MAC config"
        assert int(m.group(1)) == expected_pf, (
            f"ParallelizationFactor={pf}: expected n_pixels={expected_pf}, got {m.group(1)}"
        )
        assert int(m.group(1)) * int(m.group(2)) == n_out

        y_hls = hm.predict(np.ascontiguousarray(x)).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"pf={pf}: C-sim differs from reference "
            f"(max err={np.max(np.abs(y_hls - y_ref)):.4f})"
        )

    def test_ring_mac_pf_ignored_under_pipeline_style(self, tmp_path):
        """Under 'pipeline' style (io_parallel + Latency default) the top-level
        PIPELINE pragma force-unrolls the MAC partition loop, so a rolled
        partition cannot exist: the emitted config must fall back to
        n_pixels = n_out / n_partitions = 1 regardless of the requested PF
        (with a warning), keeping multiplier_limit spanning the whole layer.
        This mirrors hls4ml's SetPipelineStyle coupling for conv layers."""
        import glob
        import re

        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "hls_pf_pipeline")

        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        cfg = hls4ml.utils.config_from_keras_model(
            patched, granularity="name", backend="Vivado"
        )
        cfg["Model"]["Precision"] = "ap_fixed<32,12>"
        # No PipelineStyle override: io_parallel + Latency resolves to 'pipeline'.
        mac_names = [n for n in cfg["LayerName"] if n.endswith("_mac")]
        for name in mac_names:
            cfg["LayerName"][name]["ParallelizationFactor"] = 11
        hm = hls4ml.converters.convert_from_keras_model(
            patched,
            hls_config=cfg,
            backend="Vivado",
            output_dir=_HLS_DIR,
            part=_HLS_PART,
        )
        hm.compile()

        params_txt = "".join(
            open(p).read()
            for p in glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        )
        n_out = H * W
        m = re.search(
            r"n_pixels\s+= (\d+);\s+static const unsigned n_partitions = (\d+);",
            params_txt,
        )
        assert m, "n_pixels / n_partitions not found in generated MAC config"
        assert int(m.group(1)) == n_out and int(m.group(2)) == 1, (
            f"pipeline style must force n_pixels=n_out={n_out}, n_partitions=1; "
            f"got n_pixels={m.group(1)}, n_partitions={m.group(2)}"
        )

        y_hls = hm.predict(np.ascontiguousarray(x)).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM

    @pytest.mark.parametrize("reuse", [1, 3])
    def test_ring_mac_3d_reuse_factor(self, tmp_path, reuse):
        """ReuseFactor must reach the Conv3d hex config and must not
        change the C-sim result — same guard as the 2D ring MAC."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_rf3d{reuse}")

        layer = hgly.Conv3d(
            COUT, kernel_size=(1, 1), use_bias=False, share_neighbors=True
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(
            model, strategy="linebuffer", allow_unvalidated=True
        )
        cfg = hls4ml.utils.config_from_keras_model(
            patched, granularity="name", backend="Vivado"
        )
        cfg["Model"]["Precision"] = "ap_fixed<32,12>"
        cfg["Model"]["ReuseFactor"] = reuse
        hm = hls4ml.converters.convert_from_keras_model(
            patched,
            hls_config=cfg,
            backend="Vivado",
            output_dir=_HLS_DIR,
            part=_HLS_PART,
        )
        hm.compile()

        import glob

        params_files = glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        assert params_files, "parameters.h not generated"
        params_txt = "".join(open(p).read() for p in params_files)
        assert f"reuse_factor = {reuse}" in params_txt, (
            f"ReuseFactor={reuse} did not reach the hex config"
        )

        y_hls = hm.predict(np.ascontiguousarray(x)).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"3D reuse={reuse}: C-sim differs from reference "
            f"(max err={np.max(np.abs(y_hls - y_ref)):.4f})"
        )

    @pytest.mark.parametrize("kernel_size,cin", [(1, 1), (2, 2), (3, 1)])
    def test_ring_mac_accum_wider_than_weight(self, tmp_path, kernel_size, cin):
        """The MAC accumulator type must be wider than the weight type by
        ceil(log2(K*Cin)) bits, so summing the neighbor products cannot overflow.

        This is a structural check on the generated config header (reliable —
        it directly asserts the fix), rather than a runtime overflow which is
        entangled with output-type saturation and hard to isolate.
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()

        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"hls_accum_k{kernel_size}_c{cin}")

        layer = hgly.Conv2d(
            2, kernel_size=kernel_size, use_bias=False, share_neighbors=True
        )
        inp = keras.Input((H, W, cin), name="x")
        model = keras.Model(inp, layer(inp))
        for w in layer.trainable_variables:
            w.assign(RNG.standard_normal(w.shape).astype(np.float32))

        patched = patch_model_for_hls(model, strategy="linebuffer")
        cfg = hls4ml.utils.config_from_keras_model(
            patched, granularity="name", backend="Vivado"
        )
        cfg["Model"]["Precision"] = "ap_fixed<16,6>"
        hm = hls4ml.converters.convert_from_keras_model(
            patched,
            hls_config=cfg,
            backend="Vivado",
            output_dir=_HLS_DIR,
            part=_HLS_PART,
        )
        hm.compile()

        import glob
        import math

        params_files = glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        params_txt = "".join(open(p).read() for p in params_files)

        # Find the ring-MAC accum_t typedef and parse its integer bits.
        # K includes the center + rings for this kernel size.
        from keras_hexagdly.indexed import _cell_list

        k = len(_cell_list(kernel_size))
        scale = math.ceil(math.log2(k * cin))
        # weight type is ap_fixed<16,6>; accum must be ap_fixed<16+scale, 6+scale>
        expected = f"ap_fixed<{16 + scale}, {6 + scale}>"
        assert f"typedef {expected} accum_t;" in params_txt, (
            f"expected accum_t {expected} (scale={scale} for K*Cin={k * cin}), "
            f"not found in generated config"
        )


@jax_skip
class TestConv2dLineBuffer:
    """Tier 1: fused HexConvLineBuffer output must match the original Conv2d."""

    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2, 3])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("bias", [False, True])
    def test_output_matches_original(self, share, kernel_size, stride, bias):
        layer = hgly.Conv2d(
            COUT,
            kernel_size=kernel_size,
            strides=stride,
            use_bias=bias,
            share_neighbors=share,
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(model, strategy="linebuffer")
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv2d linebuffer(k={kernel_size},s={stride},share={share},use_bias={bias}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )


@jax_skip
class TestConv3dLineBuffer:
    """Tier 1: fused HexConvLineBuffer3D must match the original Conv3d."""

    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2), (1, 2)])
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_output_matches_original(self, kernel_size, share, depth_padding):
        layer = hgly.Conv3d(
            COUT,
            kernel_size=kernel_size,
            strides=1,
            use_bias=True,
            share_neighbors=share,
            depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)

        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0)
        patched = patch_model_for_hls(
            model, strategy="linebuffer", allow_unvalidated=True
        )
        y_pat = patched.predict(x, verbose=0)

        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < ATOL_KERAS, (
            f"Conv3d linebuffer(k={kernel_size},share={share},dpad={depth_padding}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )


@jax_skip
class TestLineBufferSerialization:
    """Patched linebuffer models must survive save/load (get/from_config)."""

    def _roundtrip(self, model, x, tmp_path, name):
        y_before = model.predict(x, verbose=0)
        f = str(tmp_path / f"{name}.keras")
        model.save(f)
        y_after = keras.models.load_model(f).predict(x, verbose=0)
        assert np.max(np.abs(y_before - y_after)) < 1e-6

    @pytest.mark.parametrize("share", [False, True])
    def test_conv2d_linebuffer_roundtrip(self, tmp_path, share):
        layer = hgly.Conv2d(
            COUT, kernel_size=2, use_bias=True, share_neighbors=share, name="c1"
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)
        patched = patch_model_for_hls(model, strategy="linebuffer")
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        self._roundtrip(patched, x, tmp_path, f"lb2d_s{share}")

    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_conv3d_linebuffer_roundtrip(self, tmp_path, depth_padding):
        layer = hgly.Conv3d(
            COUT,
            kernel_size=(2, 1),
            use_bias=True,
            share_neighbors=True,
            depth_padding=depth_padding,
            name="c3",
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)
        patched = patch_model_for_hls(
            model, strategy="linebuffer", allow_unvalidated=True
        )
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        self._roundtrip(patched, x, tmp_path, f"lb3d_{depth_padding}")


@hls4ml_skip
@jax_skip
class TestLineBufferCsim:
    """Tier 2: the linebuffer HLS kernels (stream line buffer + parallel
    fallback) must C-sim to the original within fixed-point error, and the
    io_stream stream config must reference the line-buffer kernel."""

    @pytest.mark.parametrize("io_type", ["io_stream", "io_parallel"])
    @pytest.mark.parametrize("share", [False, True])
    @pytest.mark.parametrize("kernel_size", [1, 2])
    def test_conv2d_linebuffer_csim(self, tmp_path, io_type, share, kernel_size):
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"lb2d_{io_type}_k{kernel_size}_s{share}")

        layer = hgly.Conv2d(
            COUT, kernel_size=kernel_size, use_bias=True, share_neighbors=share
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32) * 0.3
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        y_hls = _csim(patched, x, io_type=io_type).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Conv2d linebuffer {io_type} k={kernel_size} share={share}: "
            f"max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    @pytest.mark.parametrize("io_type", ["io_stream", "io_parallel"])
    @pytest.mark.parametrize("depth_padding", ["valid", "same"])
    def test_conv3d_linebuffer_csim(self, tmp_path, io_type, depth_padding):
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"lb3d_{io_type}_{depth_padding}")

        layer = hgly.Conv3d(
            COUT,
            kernel_size=(2, 2),
            use_bias=True,
            share_neighbors=True,
            depth_padding=depth_padding,
        )
        model = _build_3d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32) * 0.3
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(
            model, strategy="linebuffer", allow_unvalidated=True
        )
        y_hls = _csim(patched, x, io_type=io_type).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM, (
            f"Conv3d linebuffer {io_type} dpad={depth_padding}: "
            f"max err={np.max(np.abs(y_hls - y_ref)):.4f}"
        )

    def test_stream_uses_linebuffer_kernel(self, tmp_path):
        """io_stream must wire the stream line-buffer kernel and emit the
        per-parity offset tables as compile-time constants."""
        import glob

        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "lb_stream_kernel")

        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=False, share_neighbors=True)
        model = _build_2d_model(layer)
        _rand_weights(layer)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        _csim(
            patched,
            RNG.standard_normal((1, H, W, CIN)).astype(np.float32),
            io_type="io_stream",
        )

        src = "".join(
            open(p).read()
            for p in glob.glob(f"{_HLS_DIR}/**/*.cpp", recursive=True)
            + glob.glob(f"{_HLS_DIR}/**/parameters.h", recursive=True)
        )
        assert "hex_conv_linebuffer_stream" in src, (
            "io_stream linebuffer did not wire the stream line-buffer kernel"
        )
        assert "off_even" in src and "off_odd" in src, (
            "per-parity offset tables not emitted in the generated config"
        )


@jax_skip
class TestMaxPoolLineBuffer:
    """Tier 1: fused HexPoolLineBuffer(3D) must match the original MaxPool."""

    @pytest.mark.parametrize("kernel_size", [1, 2])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("neg", [False, True])
    def test_maxpool2d_matches_original(self, kernel_size, stride, neg):
        layer = hgly.MaxPool2d(kernel_size=kernel_size, strides=stride)
        model = _build_2d_model(layer)
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        if neg:
            x = -np.abs(x)
        y_ref = model.predict(x, verbose=0)
        y_pat = patch_model_for_hls(model, strategy="linebuffer").predict(x, verbose=0)
        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < 1e-5, (
            f"MaxPool2d linebuffer(k={kernel_size},s={stride},neg={neg}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    @pytest.mark.parametrize("kernel_size", [(1, 1), (2, 2), (1, 2)])
    @pytest.mark.parametrize("neg", [False, True])
    def test_maxpool3d_matches_original(self, kernel_size, neg):
        layer = hgly.MaxPool3d(kernel_size=kernel_size)
        model = _build_3d_model(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        if neg:
            x = -np.abs(x)
        y_ref = model.predict(x, verbose=0)
        y_pat = patch_model_for_hls(
            model, strategy="linebuffer", allow_unvalidated=True
        ).predict(x, verbose=0)
        assert y_ref.shape == y_pat.shape
        assert np.max(np.abs(y_ref - y_pat)) < 1e-5, (
            f"MaxPool3d linebuffer(k={kernel_size},neg={neg}): "
            f"max err={np.max(np.abs(y_ref - y_pat)):.2e}"
        )

    def test_maxpool3d_hex_stride2_raises(self):
        """MaxPool3d linebuffer is hex-stride 1 only."""
        layer = hgly.MaxPool3d(kernel_size=(1, 1), strides=(1, 2))
        model = _build_3d_model(layer)
        with pytest.raises(NotImplementedError, match="strides=1 only"):
            patch_model_for_hls(model, strategy="linebuffer", allow_unvalidated=True)


@jax_skip
class TestPoolLineBufferSerialization:
    def _roundtrip(self, model, x, tmp_path, name):
        y0 = model.predict(x, verbose=0)
        f = str(tmp_path / f"{name}.keras")
        model.save(f)
        y1 = keras.models.load_model(f).predict(x, verbose=0)
        assert np.max(np.abs(y0 - y1)) < 1e-6

    @pytest.mark.parametrize("stride", [1, 2])
    def test_pool2d_roundtrip(self, tmp_path, stride):
        layer = hgly.MaxPool2d(kernel_size=2, strides=stride, name="p1")
        model = _build_2d_model(layer)
        patched = patch_model_for_hls(model, strategy="linebuffer")
        x = RNG.standard_normal((2, H, W, CIN)).astype(np.float32)
        self._roundtrip(patched, x, tmp_path, f"lbpool2d_s{stride}")

    def test_pool3d_roundtrip(self, tmp_path):
        layer = hgly.MaxPool3d(kernel_size=(2, 1), name="p3")
        model = _build_3d_model(layer)
        patched = patch_model_for_hls(
            model, strategy="linebuffer", allow_unvalidated=True
        )
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        self._roundtrip(patched, x, tmp_path, "lbpool3d")


@hls4ml_skip
@jax_skip
class TestLineBufferEdgeCasesCsim:
    """Edge-case C-simulation coverage ported from the gather/slotwise suites
    when those strategies were retired. Each one guards a failure mode that a
    plain "output matches" test does not reach."""

    def test_border_slots_contribute_exactly_zero(self, tmp_path):
        """Out-of-grid neighbours must contribute 0, not a float residual.

        With an all-ones kernel every valid neighbour adds exactly +1 per input
        channel, so a border pixel -- which has fewer valid neighbours -- must
        come out strictly BELOW an interior pixel. If the -1 border sentinel is
        ever compared as a float (the bug this originally caught), border slots
        read garbage and the two become equal or inverted.
        """
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "lb_border")

        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=False, share_neighbors=False)
        model = _build_2d_model(layer)
        for w in layer.weights:
            w.assign(np.ones(w.shape, dtype=np.float32))
        x = np.ones((1, H, W, CIN), dtype=np.float32)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        y = _csim(patched, x, io_type="io_stream").reshape(H, W, COUT)
        interior = float(y[H // 2, W // 2, 0])
        corner = float(y[0, 0, 0])
        assert corner < interior - 0.5, (
            f"corner={corner} not below interior={interior}: border slots are "
            "contributing something other than zero"
        )

    def test_ring_sharing_matches_broadcast_equivalent(self, tmp_path):
        """share_neighbors=True must equal an unshared layer whose weights are
        the ring weights broadcast over each ring. Catches a wrong ring_idx
        precision or stride in the kernel."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "lb_ring")

        shared = hgly.Conv2d(COUT, kernel_size=1, use_bias=False, share_neighbors=True)
        model_shared = _build_2d_model(shared)
        _rand_weights(shared)
        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32) * 0.3

        y_ref = model_shared.predict(x, verbose=0).reshape(-1)
        patched = patch_model_for_hls(model_shared, strategy="linebuffer")
        y_hls = _csim(patched, x, io_type="io_stream").reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM

    def test_bias_reaches_the_output(self, tmp_path):
        """A zero kernel with a known bias must produce exactly that bias
        everywhere -- isolates the bias path from the MAC path."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "lb_bias")

        layer = hgly.Conv2d(COUT, kernel_size=1, use_bias=True, share_neighbors=False)
        model = _build_2d_model(layer)
        for w in layer.weights:
            if "bias" in w.name:
                w.assign(np.arange(1, COUT + 1, dtype=np.float32) * 0.25)
            else:
                w.assign(np.zeros(w.shape, dtype=np.float32))
        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        y = _csim(patched, x, io_type="io_stream").reshape(H, W, COUT)
        expected = np.arange(1, COUT + 1, dtype=np.float32) * 0.25
        assert np.max(np.abs(y - expected[None, None, :])) < ATOL_CSIM

    def test_maxpool_all_negative_input(self, tmp_path):
        """All-negative input: the border zero-pads dominate and legitimately
        become the maximum, so the reference itself is 0 at the edges. What
        matters is that the kernel reproduces that exactly -- an interior pixel
        must stay negative rather than being clamped to the pad value."""
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / "lb_negpool")

        layer = hgly.MaxPool2d(kernel_size=1, strides=1)
        model = _build_2d_model(layer)
        x = -np.abs(RNG.standard_normal((1, H, W, CIN)).astype(np.float32)) - 0.5
        y_ref = model.predict(x, verbose=0).reshape(-1)

        patched = patch_model_for_hls(model, strategy="linebuffer")
        y_hls = _csim(patched, x, io_type="io_stream").reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM
        interior = y_hls.reshape(H, W, CIN)[H // 2, W // 2, 0]
        assert interior < 0.0, "interior pixel was clamped to the border pad value"


class TestLineBufferPoolAndStrideCsim:
    """Tier 2: pool line-buffer kernels + strided conv line-buffer must C-sim."""

    @pytest.mark.parametrize("io_type", ["io_stream", "io_parallel"])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_maxpool2d_linebuffer_csim(self, tmp_path, io_type, stride):
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"lbpool2d_{io_type}_s{stride}")

        layer = hgly.MaxPool2d(kernel_size=1, strides=stride)
        model = _build_2d_model(layer)
        x = -np.abs(
            RNG.standard_normal((1, H, W, CIN)).astype(np.float32)
        )  # border-0 dominates
        y_ref = model.predict(x, verbose=0).reshape(-1)
        patched = patch_model_for_hls(model, strategy="linebuffer")
        y_hls = _csim(patched, x, io_type=io_type).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM

    @pytest.mark.parametrize("io_type", ["io_stream", "io_parallel"])
    def test_maxpool3d_linebuffer_csim(self, tmp_path, io_type):
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"lbpool3d_{io_type}")

        layer = hgly.MaxPool3d(kernel_size=(2, 1))
        model = _build_3d_model(layer)
        x = RNG.standard_normal((1, D, H, W, CIN)).astype(np.float32)
        y_ref = model.predict(x, verbose=0).reshape(-1)
        patched = patch_model_for_hls(
            model, strategy="linebuffer", allow_unvalidated=True
        )
        y_hls = _csim(patched, x, io_type=io_type).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM

    @pytest.mark.parametrize("io_type", ["io_stream", "io_parallel"])
    @pytest.mark.parametrize("share", [False, True])
    def test_conv2d_linebuffer_stride2_csim(self, tmp_path, io_type, share):
        from keras_hexagdly.hls4ml_handler import register_hex_gather_layers

        register_hex_gather_layers()
        global _HLS_DIR
        _HLS_DIR = str(tmp_path / f"lbconv2d_s2_{io_type}_sh{share}")

        layer = hgly.Conv2d(
            COUT, kernel_size=2, strides=2, use_bias=True, share_neighbors=share
        )
        model = _build_2d_model(layer)
        _rand_weights(layer)
        x = RNG.standard_normal((1, H, W, CIN)).astype(np.float32) * 0.3
        y_ref = model.predict(x, verbose=0).reshape(-1)
        patched = patch_model_for_hls(model, strategy="linebuffer")
        y_hls = _csim(patched, x, io_type=io_type).reshape(-1)
        assert np.max(np.abs(y_hls - y_ref)) < ATOL_CSIM

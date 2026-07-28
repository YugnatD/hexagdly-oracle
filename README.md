# hexagdly-oracle

The single, physical home of the hexagdly test suite -- for both what's
shared between `keras-hexagdly` and `pytorch-hexagdly`, and what currently
only applies to one of them.

The two libraries under test:

- **[keras-hexagdly](https://github.com/YugnatD/keras-hexagdly)** — Keras 3 port
  (any backend, channels-last), plus `share_neighbors`, `depth_padding` and
  hls4ml/FPGA export.
- **[pytorch-hexagdly](https://github.com/YugnatD/pytorch-hexagdly)** — PyTorch
  fork of [ai4iacts/hexagdly](https://github.com/ai4iacts/hexagdly), plus
  `share_neighbors`.

Both consume this repo's tests from CI rather than vendoring them, so a change
here is what actually gates their builds. Neither consuming repo keeps a local copy of
test logic that lives here, even for keras-only content like hls4ml export:
the point isn't just deduplication, it's having exactly one place a test for
this family of libraries can be found, read, and fixed.

Where a test genuinely applies to both frameworks (hand-verified layer
outputs, `share_neighbors` weight-sharing), it is written ONCE and run
against both through a small per-framework adapter -- not copy-pasted into a
"test_layers_keras.py" and a "test_layers_pytorch.py" that differ only by
s/hgly/hex/ and NCHW<->NHWC transposes (an earlier, halfway version of this
repo did exactly that; see git history). Where a test currently only applies
to one framework (hls4ml export doesn't exist yet on the pytorch side;
`depth_padding`/some validation behavior are keras-hexagdly-only design
decisions), it still lives here, single-backed, structured so adding the
other framework's backend later is additive, not a rewrite.

A fix to shared ground truth now needs to happen once, in one place, instead
of two copies risking silent divergence -- which is exactly what happened
once: a parity-probe bug in keras-hexagdly's `ring_maps_2d(2)` classified 4 of
19 taps into the wrong ring, and the only reason it surfaced was a manual
line-by-line diff against pytorch-hexagdly's own hardcoded table, not any
automated check.

## Layout

- `src/hexagdly_oracle/` — framework-agnostic (numpy only) ground truth and
  oracle functions. Nothing here imports keras or torch.
  - `geometry.py` — per-output-column-parity `(dr, dc) -> weight_group`
    tables for kernel_size 1 and 2, for each share_neighbors mode, plus the
    no-share `(sub_kernel, row, col) -> per-parity (dr, dc)` offset tables.
    Derived by firing single-pixel impulses through the hex grid and
    recording exactly which input pixel each kernel cell reads.
  - `weight_maps.py` — `HARDCODED_MAPS`: the `(mode, kernel_size, parity) ->
    (sub_kernel_group_arrays, num_weights)` tables used to validate each
    framework's own `weight_maps_2d`-equivalent derivation. Previously
    `keras-hexagdly` read this out of `pytorch_hexagdly`'s *private*
    `_HARDCODED_MAPS` attribute directly; this is now the shared, public
    source both sides depend on.
  - `hex_reference.py` — hand-verified Conv2d/Conv3d/MaxPool2d/MaxPool3d/
    Conv2d_CustomKernel expected-output arrays (two independent oracles: a
    sequential grid and a sparse impulse grid), ported from HexagDLy's own
    upstream test suite.
  - `oracle.py` — `oracle()` (kernel_size=1), `oracle_k2()` (kernel_size=2),
    `oracle_noshare()`: recompute expected conv output directly from grid +
    weights + geometry table, independent of any framework's own
    weight-materialization code path.
  - `fixtures.py` — deterministic test grids and weight dicts.
  - `testing.py` — `check_grid()`, a tiny elementwise-tolerance comparison
    helper.
- `tests/` — the actual test suite. Every file is import-guarded
  (`pytest.importorskip`) so it is skipped, not collection-errored, when its
  framework(s) aren't installed:
  - `test_oracle.py`, `test_hex_reference.py` — self-consistency of the
    ground truth itself, no framework needed.
  - `test_layers.py` — every hand-verified layer test and share_neighbors
    oracle test, written ONCE and run against whichever of `_KerasBackend` /
    `_PytorchBackend` is importable. This is the dual-backed core; everything
    else below is currently single-backed (keras only).
  - `test_cross_framework.py` — the single place that owns "do the two ports
    agree with each other" (not just with this package's oracle). Needs both
    `keras_hexagdly` and `pytorch_hexagdly` installed.
  - `test_geometry_keras.py`, `test_edge_cases_keras.py`,
    `test_indexed_equivalence_keras.py`, `test_depth_padding_keras.py`,
    `test_mixed_precision_keras.py`, `test_serialization_keras.py`,
    `test_vs_upstream_hexagdly_keras.py` — keras-hexagdly-only content (no
    pytorch-hexagdly equivalent exists): impulse-response geometry checks,
    input validation, the `indexed`-representation equivalence proof behind
    the hls4ml export path, `depth_padding="same"`, mixed-precision policies,
    serialization round-trips, and cross-checks against the plain upstream
    `hexagdly` PyPI package (a DIFFERENT oracle than `test_cross_framework.py`
    -- upstream `hexagdly`, not this user's `pytorch_hexagdly` fork).
  - `test_hls4ml_ext_keras.py` — hls4ml export correctness (patch_model_for_hls)
    and optional C-sim, keras-only since pytorch-hexagdly has no hls4ml
    support yet. Tier 2 (C-sim) additionally needs Vitis HLS on PATH and is
    skipped gracefully without it.

## Scope: stride=1 only (share_neighbors oracle)

Every oracle function and hardcoded table in `geometry.py`/`weight_maps.py`
is valid for `stride=1` only. The per-output-column-parity offset table is
**not** stride-invariant for kernel_size=2 (verified empirically: the
odd-parity offsets shift from a `(-2..+2, 0)` window at stride=1 to
`(-1..+3, 0)` at stride>=2), so reusing a stride=1-derived table at stride>1
silently produces wrong numbers. Neither `keras-hexagdly` nor
`pytorch-hexagdly` extends these hardcoded tables past stride=1 for this
reason. Stride>1 coverage for share_neighbors lives in
`test_cross_framework.py` instead, via the real forward pass of both ports,
which needs no assumption about how the offset table behaves across strides.

## Installing

Not published to PyPI. Install directly from git, pinning a tag:

```
pip install "git+https://github.com/YugnatD/hexagdly-oracle.git@v0.3.0"
```

Optional extras select which part of the test suite you can actually run:
`layers-keras`, `layers-keras-hls4ml` (adds `hls4ml`), `layers-pytorch`,
`cross-framework` (needs both frameworks), or `all`. Each consuming repo's
own local test suite has nothing left that needs `hexagdly-oracle` installed
at all -- these extras are only for running hexagdly-oracle's own suite.

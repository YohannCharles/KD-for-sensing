# CSI Hardening Matrix

Protocol status and claim provenance are tracked in `docs/experiment_protocols.md` and `docs/result_claims_registry.md`. The full matrix is `formal`; the debug matrix is `debug` and must pass clone/parity checks before any full-sweep interpretation.
Formal A/B/C/D configs are lightweight overlays on `_base/csi_only.yaml`; inspect the resolved config from `kd-sensing-train` for the full merged values.

Use `scripts/run_csi_hardening_matrix.sh` for staged execution and logging. Run commands through the project conda environment, for example:

```bash
conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh quick
NEW_RUN=1 conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh csi-only
```

Recommended order:

0. Debug matrix first: run the five configs under `configs/csi/hardening_matrix/debug` and do not interpret the full sweep until `csi_debug_A0_clone_generated` is close to `csi_debug_A0_original`.

1. First batch: A0, A1, A2, B3, B4, B5, B6, C1, C2.
2. Second batch: D1, D2, D3, D4.
3. Third batch: E0-E3 under `configs/fusion/csi_hardening_matrix`.

A1 is the only mild pilot-estimation run and uses `csi_estimation.mode: est_snr` with train-time 25-35 dB sampling and 30 dB eval diagnostics. B/C/D runs explicitly set `csi_estimation.mode: none`; they isolate hardening or encoder variables and must not inherit A1 pilot noise.

A2 is the destructive degradation negative control and keeps destructive physical pilot noise metadata so analysis can separate it from mild pilot estimation. B and D groups use information-preserving `csi_hardening`; D group configs intentionally do not enable `csi_degradation`.

The current authoritative `A0_original` reference in this workspace is the actual resolved training artifact:

`outputs/csi_hardening_matrix_20260520_164406/Town10_skybridge_seed24/csi_A0_clean_full_teacher/final_config.yaml`

The old `outputs/csi_hardening_matrix_20260520_164406` sweep used uncalibrated physical pilot noise in non-A0 variants. Keep the raw artifacts, but treat any analysis from that root as invalid or pending unless it has the new pilot-scaling diagnostics and passes the gate.

If that artifact is unavailable in a fresh workspace, run `csi_debug_A0_original` first and pass its `resolved_config.yaml` as `A0_ORIGINAL_CONFIG` when launching the debug stage.

Separate debug and full sweep commands:

```bash
NEW_RUN=1 conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh debug
conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh csi-ab
conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh csi-d
conda run -n kd_mm_beam python scripts/analyze_csi_hardening_sweep.py \
  --runs_root "$SCENE_ROOT" \
  --pattern 'csi_*' \
  --clean_teacher_run csi_A0_clean_full_teacher \
  --out "$SCENE_ROOT/analysis_csi_ABCD"
```

Debug decision rules:

- If `A0_original` learns but `A0_clone_generated` stays near random, fix config generation/inheritance before discussing hardening.
- If `A0_clone_generated` learns but `C1_view_gate_warmup_only` fails, inspect view gate warmup and fusion broadcasting.
- If `A0_clone_generated` learns but `C2_no_internal_gru_only` fails, inspect the no-internal-GRU encoder-to-representation path.
- If only B3/B4-style hardening variants fail after clone parity passes, inspect hardening transforms and fixed-by-seed state.
- If A1 pilot estimation fails while B/C paths are healthy, inspect pilot estimator SNR/noise power calculation and `noise_power_signal_ratio`.

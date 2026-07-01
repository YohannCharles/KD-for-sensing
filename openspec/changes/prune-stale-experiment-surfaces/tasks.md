## 1. Baseline Audit

- [ ] 1.1 Re-run current status checks: `openspec status --change prune-stale-experiment-surfaces`, `git status --short`, and CodeGraph caller checks for `build_loso_dataloaders`, `build_loso_source_train_loader`, `build_loso_target_stage_loader`, and `load_full_sweep_manifest`.
- [ ] 1.2 Inventory every current reference to `cnn_hybrid_jepa_visual_prior_sweep`, `jepa_visual_architecture_sweep`, `kd_sensing.cli.beambench_check_dataset`, `kd_sensing.engine.loso_data`, `configs/scene31/`, RBMA strong-encoder configs, M2Beam single-modal configs, and local queue scripts.
- [ ] 1.3 Confirm no planned deletion touches `dataset/`, `outputs/`, `logs/`, cache, checkpoint, TensorBoard event files, `All_models/`, or system configuration files.

## 2. Governance Drift Fixes

- [ ] 2.1 Update `openspec/specs/beambench-baseline-reproduction/spec.md` so BeamBench data checking uses `kd_sensing.baselines.beambench.dataset_check` or an equivalent retained package entry, not the deleted `kd_sensing.cli.beambench_check_dataset`.
- [ ] 2.2 Update `docs/project_surface_inventory.md`, `docs/agent_navigation.md`, README/docs references, and architecture boundary expectations for stale current references found in task 1.2.
- [ ] 2.3 Add or adjust architecture guardrails so deleted current references fail, while archive/history/local-manual references remain allowed when clearly labelled.
- [ ] 2.4 Run `openspec validate prune-stale-experiment-surfaces --strict` after the governance-only edits.

## 3. JEPA Sweep Surface

- [ ] 3.1 Migrate model architecture summary and tests from old `load_full_sweep_manifest` assumptions to current `jepa_visual_architecture_sweep` manifest support.
- [ ] 3.2 Decide the minimal legacy path: delete `src/kd_sensing/diagnostics/cnn_hybrid_jepa_visual_prior_sweep.py` entirely if no current consumer remains, otherwise keep only a small read-only compatibility reader.
- [ ] 3.3 Remove old full-sweep runner behavior: job graph execution, generated shell scripts, output-root cleanup, GPU scheduling, and tests that only protect that legacy runner.
- [ ] 3.4 Update `configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml` references: either remove them, mark them historical, or route current docs/tests to `configs/diagnostics/jepa_visual_architecture_sweep_manifest.yaml`.
- [ ] 3.5 Run focused checks: `conda run -n kd_mm_beam pytest tests/test_jepa_visual_architecture_sweep.py tests/test_model_architecture_summary.py -q`.

## 4. LOSO Helper Surface

- [ ] 4.1 Verify `src/kd_sensing/engine/loso_data.py` has no current internal caller, console script, registry registration, README/docs current entry, or required test dependency.
- [ ] 4.2 Delete `src/kd_sensing/engine/loso_data.py` and only its dedicated tests if task 4.1 confirms no public current dependency; otherwise replace it with a deprecation stub that does not implement training/adaptation loops.
- [ ] 4.3 Update `openspec/specs/cross-scene-loso-workflow/spec.md` and inventory wording so LOSO supporting semantics live in fold/few-shot planning, not an implicit engine dataloader facade.
- [ ] 4.4 Run focused checks covering LOSO planning or imports: `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`.

## 5. Local/Manual Scene31 and RBMA Surface

- [ ] 5.1 Classify `scripts/run_next_v3_experiments.sh`, `scripts/run_rbma_strong_encoder_4gpu_queue.sh`, `scripts/run_m2beam_single_modal_scene31_queue.sh`, and `scripts/run_rbma_missing_workflow.py` as delete, local/manual retained, or unified-runner covered.
- [ ] 5.2 Delete fixed GPU shell scripts when `scripts/run_rbma_missing_workflow.py`, documented `kd-sensing-train` commands, or another minimal local/manual runner covers the same config list.
- [ ] 5.3 Classify `configs/scene31/`, `configs/fusion/experiments/rbma_missing_workflow_strong_encoders/`, and `configs/fusion/experiments/m2beam_single_modal_scene31/`; delete only configs whose conclusions are captured or whose run path is replaced.
- [ ] 5.4 Update `docs/result_claims_registry.md`, `docs/experiment_matrix.md`, `docs/experiment_protocols.md`, and `docs/mainline_model_catalog.md` so retained Scene31/RBMA configs remain pending/local/manual and checkpoint placeholders do not become promoted claims.
- [ ] 5.5 Run focused checks: `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`.

## 6. Local Bytecode Cleanup

- [ ] 6.1 Delete ignored `__pycache__` directories and `.pyc` files under `src/`, `scripts/`, and `tests/`.
- [ ] 6.2 Confirm `git status --short` does not include tracked bytecode deletions and does not include `dataset/`, `outputs/`, `logs/`, cache, checkpoint, or history weight changes.

## 7. Final Validation

- [ ] 7.1 Run `openspec validate prune-stale-experiment-surfaces --strict`.
- [ ] 7.2 Run `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`.
- [ ] 7.3 Run `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`.
- [ ] 7.4 Run additional focused tests changed by the implementation, especially JEPA sweep/model summary, BeamBench dataset check, LOSO planning, and local/manual config tests.
- [ ] 7.5 Record any skipped validation and remaining local/manual surfaces in the final implementation note.

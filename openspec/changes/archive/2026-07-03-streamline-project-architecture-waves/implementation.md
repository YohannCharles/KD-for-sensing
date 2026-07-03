## Implementation Notes

### Wave 0 Baseline

Captured at 2026-07-03 Asia/Shanghai.

- Active changes: `openspec list --json` shows only `streamline-project-architecture-waves`, with 0/93 tasks complete before implementation. There are no completed active changes to archive or defer.
- Worktree: `git status --short` shows only untracked `openspec/changes/streamline-project-architecture-waves/`. `git diff --stat` is empty before implementation edits.
- Local artifacts: `dataset/DeepSense6G/` and `dataset/_downloads/` are ignored by `.gitignore:43`; `.codegraph/` daemon/index files are ignored through `.git/info/exclude`; `.pytest_cache/` and multiple `__pycache__/` directories exist locally and are ignored. `dataset/.gitkeep` is currently absent and not tracked.
- Tracked boundary files: `.gitignore`, `.codegraph/.gitignore`, and `.codex/skills/**/SKILL.md` are the only tracked paths in the checked artifact/tool areas.

Baseline validation:

- `openspec validate streamline-project-architecture-waves --strict`: passed.
- `openspec validate --all --strict`: passed, 111 items.
- `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`: passed, 28 tests.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: failed with 18 passed / 2 failed. Existing failures are:
  - `test_openspec_specs_match_lifecycle_inventory`: `scene31-next-round-experiment-workflow` exists under current specs but is missing from the lifecycle inventory table.
  - `test_scripts_are_classified_in_inventory`: several tracked local scripts are missing inventory classification, including `scripts/analyze_btapa_tau1_seeds.py`, `scripts/analyze_night_grid.py`, `scripts/analyze_proto_vs_btapa_seeds.py`, `scripts/eval_night_grid.py`, `scripts/generate_experiment_grid.py`, and `scripts/generate_scene31_next_round.py`.

Wave order and rollback:

- Wave 1 dataset adapter work must preserve `DATASETS.build({"type": "deepsense6g"})` and `DATASETS.build({"type": "mmw"})` behavior and roll back by reverting adapter/helper moves only.
- Wave 2 runtime work must preserve `train(cfg)`, output layout, checkpoint/status/log schema, and shared evaluation output schema.
- Wave 3 modular forward work must preserve public `forward` signature, output keys, and `adapt_model_output` behavior.
- Wave 4 diagnostics/runtime artifact work must preserve package CLI names and manifest/output schema.
- Wave 5 config/script/import work must classify local/manual surfaces and remove only undocumented internal paths.
- Wave 6/7 docs/spec/guardrail work must resolve the two architecture baseline failures without weakening retired-route or runtime-artifact guards.
- Rollback must not restore retired CLI/config/registry/import routes; if a wave cannot be completed, defer that wave explicitly rather than adding compatibility wrappers.

### Wave 1 Dataset Adapter

Owner mapping:

- Sequence sample core: `src/kd_sensing/data/datasets/deepsense6g_contract.py` owns CSV path resolution, enabled modality normalization, beam target source normalization, target path selection, sample path metadata, sequence position parsing, and beam label cache mode.
- Modality readers/cache coordination: `src/kd_sensing/data/datasets/deepsense6g_loaders.py` owns `DeepSense6GModalityLoader` and image/LiDAR cache wiring; image/radar/GPS/LiDAR/mmWave/CSI transforms remain in their domain `transform_ops` owners.
- Target providers/sample assembly: `src/kd_sensing/data/datasets/deepsense6g_targets.py` owns occlusion/position/soft target provider state; `src/kd_sensing/data/datasets/deepsense6g_sample_assembly.py` owns beam and auxiliary target tensor assembly.
- Scaler/normalizer setup: `src/kd_sensing/data/datasets/deepsense6g_scalers.py` owns GPS/mmWave/CSI/LiDAR/position target scaler orchestration and keeps existing artifact formats.
- MMW family adapter: `src/kd_sensing/data/datasets/mmw_family_adapter.py` now owns MMW init preparation, condition layout defaults, derived CSV column setup, beam label calibration setup, geometry/radio/path/physical/physics setup, metadata/domain metadata assembly, and physics sample field patching. `MMWDataset` remains the `DATASETS.register("mmw")` owner and still subclasses `DeepSense6GDataset` for the shared sample contract, but delegates family-specific orchestration to `MMWFamilyAdapter`.

Implementation:

- Added `MMWFamilyInit`, `prepare_mmw_family_init()`, `MMWFamilyAdapter`, and `apply_mmw_physics_sample_fields()` in `mmw_family_adapter.py`.
- Reduced `MMWDataset.__init__()` to adapter init preparation, `DeepSense6GDataset` construction, and adapter construction.
- Reduced `MMWDataset.__getitem__()` to shared sample retrieval plus `family_adapter.augment_sample()`.
- Updated `tests/test_mmw_town10_preparation.py` to assert the registry dataset uses `MMWFamilyAdapter` while preserving existing sample schema assertions.

Validation:

- `conda run -n kd_mm_beam python -m compileall -q src/kd_sensing/data/datasets/mmw.py src/kd_sensing/data/datasets/mmw_family_adapter.py`: passed.
- `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py::test_mmw_dataset_loads_mmwave_only_and_image_fusion_lazily tests/test_mmw_town10_preparation.py::test_mmw_dataset_factory_registers_scene_defaults -q`: passed, 2 tests.
- `conda run -n kd_mm_beam pytest tests/test_beam_label_calibration.py::test_mmw_dataset_maps_explicit_future_label_and_records_provenance tests/test_beam_label_calibration.py::test_mmw_soft_and_physical_distributions_follow_calibrated_class_order tests/test_beam_label_calibration.py::test_mmw_run_and_prediction_metadata_declare_label_space -q`: passed, 3 tests.
- `conda run -n kd_mm_beam pytest tests/test_deepsense6g_contract_helpers.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py tests/test_csi_modality.py tests/test_mmw_town10_preparation.py -q`: passed, 176 tests, 50 warnings.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: failed with the same 2 Wave 0 baseline failures (`scene31-next-round-experiment-workflow` missing from lifecycle inventory and unclassified tracked scripts). No new dataset/import/facade failure appeared.

### Wave 2 Training / Evaluation Runtime

Implementation:

- Added `src/kd_sensing/engine/training_run_context.py` with `TrainingRunContext` to carry cfg, objective metadata, run directory, artifact writer, dataloaders, normalization artifacts, device/runtime state, model/optimizer/scheduler/scaler, checkpoint manager, TensorBoard writer, extensions, recorder, validation loader, final test metrics, and final artifacts.
- Refactored `trainer._train_inner()` into phase helpers: `_prepare_training_run_context()`, `_build_training_resources()`, `_restore_training_state()`, `_run_training_loop_phase()`, and `_finalize_training_run()`.
- Kept `train(cfg)`, run directory creation, status files, checkpoint layout, `train_log.json`, `final_config.yaml`, TensorBoard startup scalars, CSI RMS handoff, extension setup, early stopping, final test evaluation, and failure status behavior compatible.
- Split supervised `run_evaluation_pass()` loop responsibilities with `_prepare_evaluation_batch()`, `_prepare_evaluation_targets()`, `_run_supervised_evaluation_step()`, existing metadata/output recording helpers, and existing final metric aggregation.

Validation:

- `conda run -n kd_mm_beam python -m compileall -q src/kd_sensing/engine/trainer.py src/kd_sensing/engine/training_run_context.py src/kd_sensing/engine/evaluation_pass.py`: passed.
- `conda run -n kd_mm_beam pytest tests/test_evaluation_pass.py -q`: passed, 8 tests.
- `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_prediction_objectives.py tests/test_evaluation_pass.py tests/test_modality_difficulty.py -q`: passed, 188 tests, 5 warnings.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: failed with the same 2 Wave 0 baseline failures. No new runtime/import/facade failure appeared.

### Wave 3 Modular Forward

Implementation:

- Refactored `ModularSequenceModel.forward()` into staged internal methods while preserving the public signature and output keys.
- Added stages for raw/reliability input collection, encoder dependency resolution and encoder/projector execution, core input/availability assembly, core/head execution, geometry prior/logit fusion/safe rerank post-processing, forward output assembly, runtime metadata, and auxiliary diagnostics/heads.
- Kept stage helpers private to `src/kd_sensing/models/modular.py`; no README/docs public API was introduced for these helpers.

Validation:

- `conda run -n kd_mm_beam python -m compileall -q src/kd_sensing/models/modular.py`: passed.
- `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_cls_token_transformer_fusion.py tests/test_geometry_prior_beam_fusion.py tests/test_amber_full_architecture.py tests/test_u_mask_beam_jepa.py -q`: passed, 57 tests, 27 warnings.
- `conda run -n kd_mm_beam pytest tests/test_model_architecture_summary.py -q`: passed, 15 tests.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: failed with the same 2 Wave 0 baseline failures. No new model/import/facade failure appeared.

### Wave 4 Diagnostics Runner / Runtime Artifact

Implementation:

- Audited JEPA benchmark, JEPA visual analysis, run-index, runtime cleanup/organize, and MMW GPS v2 owners. Existing JEPA benchmark responsibilities already live in manifest/common/scenario/predictive/artifact/plot owner modules, with the public shortcut benchmark facade preserving CLI and runner compatibility.
- Refactored `src/kd_sensing/engine/mmw_town_gps_v2.py` with `MMWTownGpsV2RunContext`, `_prepare_mmw_town_gps_v2_run_context()`, `_run_mmw_town_gps_v2_protocols()`, `_mmw_town_protocol_sample_sets()`, and `_write_mmw_town_gps_v2_artifacts()`.
- Kept `run_mmw_town_gps_v2()` public parameters, return fields, output directory layout, CSV/JSON/NPY artifact names, metadata keys, label-space handling, support manifest, logits/probability export, and config snapshot behavior compatible.

Validation:

- `conda run -n kd_mm_beam python -m compileall -q src/kd_sensing/engine/mmw_town_gps_v2.py`: passed.
- `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q`: passed, 5 tests.
- `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_jepa_visual_analysis.py tests/test_run_index.py tests/test_runtime_artifact_cleanup.py tests/test_runtime_output_layout.py -q`: passed, 56 tests, 1 warning.
- `conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help`: passed.
- `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help`: passed.
- `conda run -n kd_mm_beam kd-sensing-runs --help`: passed.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: still failed with the two Wave 0 inventory baseline failures before Wave 5 documentation cleanup.

### Wave 5 Config / Script / Entry Surface

Implementation:

- Classified the Scene31 next-round/night-grid config families in `docs/project_surface_inventory.md`: `configs/scene31/templates/main_v3_proto_es20_base.yaml` is the generator base; `configs/scene31/night_grid/` and `configs/scene31/next_round/` are manifest-backed local/manual config families, not root canonical entries.
- Added the missing `scene31-next-round-experiment-workflow` lifecycle row to the OpenSpec capability inventory.
- Classified the 12 previously unregistered tracked scripts: Scene31 config generators, fresh-eval helpers, research diagnostics, summaries, and shell orchestration. Each entry records owner role and ignored output/log boundary.
- Updated README, experiment matrix, AI navigation, and maintainer context index so Scene31 local queues are described as manifest-backed local/manual surfaces; real training continues through `kd-sensing-train --config <yaml>`.
- Did not delete Scene31 YAML: the generator/manifest tests depend on explicit run-name, seed, epoch, sampler, loss-weight, missing-pattern, and output-boundary semantics.

Validation:

- Missing-script inventory check using tracked `scripts/*.{py,sh}`: passed, no unclassified tracked scripts.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: passed, 20 tests. This resolves both Wave 0 baseline failures.
- `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_cli_help.py tests/test_scene31_next_round.py -q`: passed, 31 tests.
- `conda run -n kd_mm_beam kd-sensing-train --help`: passed.
- `conda run -n kd_mm_beam kd-sensing-evaluate --help`: passed.
- `conda run -n kd_mm_beam kd-sensing-preprocess --help`: passed.

### Wave 6 Import Surface / Helper Consolidation

Implementation:

- Audited package markers, `__all__`, registry import surface, facade-sensitive imports, and internal owner paths.
- No additional facade deletion was required in this wave: existing architecture tests already reject internal facade回流, package `models` and BeamBench package markers are verified not to grow heavyweight `__all__`, and newly introduced helpers stay private in their owner modules.
- Recorded the no-code outcome as a guardrail result rather than adding compatibility wrappers or broad package-level exports.

Validation:

- `conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_architecture_boundaries.py -q`: passed, 45 tests.
- `conda run -n kd_mm_beam pytest tests/test_model_architecture_summary.py tests/test_config_load_characterization.py -q`: passed, 26 tests.

### Wave 7 OpenSpec / Docs / Guardrails

Implementation:

- Updated `openspec/specs/project-architecture/spec.md` with the post-streamlining owner boundary: dataset family setup, training context, evaluation batch steps, forward diagnostics, protocol dispatch, and artifact writing must stay in narrow owner/context/stage helpers rather than flowing back into public facades or giant entry functions.
- Updated `project-hotspot-governance`, `project-import-surface-consolidation`, and `project-health-guardrails` specs with archive-ready guardrail wording for hotspot回流, lightweight package markers, and inventory-driven script lifecycle checks.
- Updated `scene31-next-round-experiment-workflow` to state that night-grid/next-round is a manifest-backed local/manual workflow, not a package CLI, and that outputs remain in ignored local roots.
- Architecture boundary tests were kept structural: they read pyproject, real tracked paths, inventory, OpenSpec lifecycle rows, and import probes rather than duplicating the full script/config database.

Validation:

- `openspec validate streamline-project-architecture-waves --strict`: passed.
- `openspec validate --all --strict`: passed, 111 items.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: passed, 20 tests.

### Cross-Wave Regression

Validation:

- `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py tests/test_component_registry.py tests/test_architecture_boundaries.py -q`: passed, 73 tests.
- Package CLI help smoke passed for `kd-sensing-train`, `kd-sensing-evaluate`, `kd-sensing-preprocess`, and `kd-sensing-runs`.
- Diagnostics/runtime CLI help smoke passed for `kd-sensing-jepa-visual-analysis`, `kd-sensing-jepa-gps-shortcut-benchmark`, `kd-sensing-clean-runtime-artifacts`, and `kd-sensing-organize-runtime-outputs`.
- `conda run -n kd_mm_beam pytest -q`: passed, 958 tests, 98 warnings.

Public compatibility:

- No package CLI names, console script names, root config paths, dataset registry names, model registry names, public `run_mmw_town_gps_v2()` parameters, `train(cfg)` public behavior, `ModularSequenceModel.forward()` signature, or diagnostics output schemas were intentionally changed.
- Internal implementation moved to new private owners/context/stage helpers. These helper names are not public API and should not be imported by downstream code.

Artifact boundary:

- No real `dataset/`, `outputs/`, `logs/`, cache, checkpoint, TensorBoard, or local training artifact was intentionally added to the source change.
- Test-created temporary files stayed under pytest temp directories or ignored local output roots.

Rollback notes:

- Wave 1 can be rolled back by reverting `mmw_family_adapter.py` and the corresponding `MMWDataset` delegation change.
- Wave 2 can be rolled back by inlining `TrainingRunContext` phases and evaluation batch-step helpers, but must preserve the existing status/checkpoint/finalization tests.
- Wave 3 can be rolled back by restoring the previous `forward()` body, but any rollback must preserve current output keys and architecture summary tests.
- Wave 4 can be rolled back by inlining MMW GPS v2 run context/protocol/artifact helpers only; JEPA/run-index/runtime cleanup modules were not materially changed.
- Waves 5-7 should not be rolled back independently unless another change also preserves the now-green architecture lifecycle/script inventory checks.

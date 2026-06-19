# Implementation Notes

## Wave 0 Baseline

- Date: 2026-06-18
- Initial `git status --short`:
  - `M src/kd_sensing/baselines/beambench/image_ae_gps.py`
  - `?? openspec/changes/right-size-hotspot-modularity/`
- Runtime artifact noise: none recorded in `git status --short`; ignored `dataset/`, `outputs/`, `logs/`, cache, checkpoint and TensorBoard files remain out of scope.

### Hotspot Line Baseline

| Path | Lines |
| --- | ---: |
| `src/kd_sensing/engine/data_factory.py` | 989 |
| `src/kd_sensing/preprocessing/sequences.py` | 670 |
| `src/kd_sensing/baselines/beambench/image_ae_gps.py` | 2438 |
| `src/kd_sensing/data/datasets/deepsense6g.py` | 1188 |
| `src/kd_sensing/data/datasets/mmw.py` | 990 |
| `src/kd_sensing/engine/trainer.py` | 566 |
| `src/kd_sensing/diagnostics/jepa_benchmark_common.py` | 851 |
| `src/kd_sensing/diagnostics/jepa_benchmark_scenario_d.py` | 1259 |
| `src/kd_sensing/diagnostics/jepa_benchmark_runner.py` | 870 |
| `src/kd_sensing/losses/jepa.py` | 106 |
| `src/kd_sensing/losses/gps_lidar_bgam_losses.py` | 118 |
| `src/kd_sensing/models/csi_encoder.py` | 326 |

### Public Surface Baseline

- `src/kd_sensing/engine/data_factory.py`: `build_dataset`, `build_dataloaders`, `build_split_dataset`, `build_protocol_split_datasets`, `build_dataloader`, `build_dataloader_kwargs`, `resolve_dataloader_split_config`, `shutdown_dataloader_workers`, `prepare_lidar_normalizer`.
- `src/kd_sensing/preprocessing/sequences.py`: `SequenceColumnPlan`, `SequenceSplit`, `SplitProtocolPlan`, `generate_sequence_data`, `resolve_sequence_column_plan`, `build_sequence_windows`, `sequence_window_columns`, `select_balanced_sequence_split`, `write_split_metadata`, `label_distribution_summary`, `SequencePreprocessor`.
- `src/kd_sensing/baselines/beambench/image_ae_gps.py`: `ImageAEGPSDirectTrainingConfig`, `BeamBenchImageAEGPSDataset`, `BeamBenchImageOnlyDataset`, `BeamBenchImageAEGPSFeatureDataset`, `BeamBenchDenseModel`, `BeamBenchImageAEGPSDirectModel`, `run_image_ae_gps_training`, `run_image_ae_gps_paper_split_training`, `run_image_ae_gps_paper_split_evaluation`, `train_camera_ae_for_image_gps_baseline`, `evaluate_image_ae_gps_model`, `resolve_image_ae_gps_config`, `timestamped_default_output`.
- Dataset/trainer/diagnostics hotspots expose the public symbols recorded in `docs/maintainer_context_index.yaml`; helper imports should move to narrow owner modules during later waves.

### Initial Validation

- `openspec validate right-size-hotspot-modularity --strict`: passed.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: failed with one existing hotspot budget red point, `src/kd_sensing/baselines/beambench/image_ae_gps.py:run_image_ae_gps_paper_split_training` at 267 lines versus budget 265. This is a Wave 2 target, not introduced by this implementation session.

## Wave 1 Data Factory

- Preserved public owner functions: `build_dataset`, `build_dataloaders`, `build_split_dataset`, `build_protocol_split_datasets`, `build_dataloader`, `build_dataloader_kwargs`, `resolve_dataloader_split_config`, `shutdown_dataloader_workers`, `prepare_lidar_normalizer`.
- Added loader-only helper module: `src/kd_sensing/engine/data_factory_loaders.py`.
- Added protocol/scene split helper module: `src/kd_sensing/engine/data_factory_protocols.py`.
- Added group split helper module: `src/kd_sensing/engine/data_factory_groups.py`.
- Added internal validation split helper module: `src/kd_sensing/engine/data_factory_validation.py`.
- Added scaler/normalizer coordination helper module: `src/kd_sensing/engine/data_factory_scalers.py`.
- `src/kd_sensing/engine/data_factory.py` is now a 220-line public build owner; helper modules do not import the public owner.
- Validation:
  - `conda run -n kd_mm_beam python -m compileall -q src/kd_sensing/engine/data_factory.py src/kd_sensing/engine/data_factory_loaders.py src/kd_sensing/engine/data_factory_protocols.py src/kd_sensing/engine/data_factory_groups.py src/kd_sensing/engine/data_factory_validation.py src/kd_sensing/engine/data_factory_scalers.py`: passed.
  - `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_epoch_subsampling.py -q`: 116 passed, 1 warning.
  - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: 65 passed.

## Wave 1b Sequence Preprocessing

- Preserved public imports from `kd_sensing.preprocessing.sequences`: `generate_sequence_data`, `SequencePreprocessor`, `SequenceColumnPlan`, `SequenceSplit`, `SplitProtocolPlan`, `resolve_sequence_column_plan`, `build_sequence_windows`, `sequence_window_columns`, `select_balanced_sequence_split`, `write_split_metadata`, `label_distribution_summary`.
- Added column planning module: `src/kd_sensing/preprocessing/sequence_columns.py`.
- Added window materialization module: `src/kd_sensing/preprocessing/sequence_windows.py`.
- Added balanced split/protocol module: `src/kd_sensing/preprocessing/sequence_splits.py`.
- Added metadata/JSON-ready module: `src/kd_sensing/preprocessing/sequence_metadata.py`.
- `src/kd_sensing/preprocessing/sequences.py` is now a 164-line owner/orchestration module.
- Validation:
  - `conda run -n kd_mm_beam python -m compileall -q src/kd_sensing/preprocessing/sequences.py src/kd_sensing/preprocessing/sequence_columns.py src/kd_sensing/preprocessing/sequence_windows.py src/kd_sensing/preprocessing/sequence_splits.py src/kd_sensing/preprocessing/sequence_metadata.py`: passed.
  - `conda run -n kd_mm_beam pytest tests/test_preprocessing_formats.py tests/test_config_load_characterization.py -q`: 10 passed.
  - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: 65 passed.

## Wave 2 BeamBench Image AE+GPS

- Preserved public imports from `kd_sensing.baselines.beambench.image_ae_gps`: config dataclass, dataset/model classes, training/evaluation/paper-split entrypoints, paper GPS constants, `TARGET_TABLE_III_ROW`, `resolve_image_ae_gps_config`, and `timestamped_default_output`.
- Split implementation into:
  - `image_ae_gps_config.py`: config dataclass, normalization, runtime/device/optimizer helpers, GPS calibration metadata.
  - `image_ae_gps_datasets.py`: direct/image-only/feature datasets plus loader/image/metadata helpers.
  - `image_ae_gps_models.py`: dense/fusion model and batch-to-logits helper.
  - `image_ae_gps_ae.py`: Camera AE training and frozen feature cache.
  - `image_ae_gps_training.py`: direct fusion training loop and selection split helpers.
  - `image_ae_gps_evaluation.py`: evaluation pass and prediction rows.
  - `image_ae_gps_paper_split.py`: paper split train/eval orchestration and summary artifacts.
  - `image_ae_gps_reports.py`: CSV/JSON-ready/report metadata helpers.
- `src/kd_sensing/baselines/beambench/image_ae_gps.py` is now a 42-line public owner/re-export module.
- The existing `run_image_ae_gps_paper_split_training` budget red point is resolved at 265 AST lines in `image_ae_gps_paper_split.py`.
- Validation:
  - `conda run -n kd_mm_beam python -m compileall -q ...image_ae_gps*.py`: passed.
  - `conda run -n kd_mm_beam pytest tests/test_beambench_image_ae_gps_direct.py -q`: 7 passed.
  - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: 65 passed.

## Wave 3 Datasets And Trainer

- DeepSense6G second pass:
  - Added `src/kd_sensing/data/datasets/deepsense6g_sample_assembly.py` for beam target tensor and auxiliary target tensor assembly.
  - Added `src/kd_sensing/data/datasets/deepsense6g_scalers.py` for streaming feature stats used by GPS/mmWave scaler fitting.
  - Existing resource-reader glue and target-provider adapter boundaries remain in `deepsense6g_loaders.py` and `deepsense6g_targets.py`.
- MMW second pass:
  - Added `mmw_columns.py` for derived CSI/BS GPS/radar CSV helper logic.
  - Added `mmw_geometry.py` for geometry/availability payload and tensor helpers.
  - Added `mmw_radio_semantic.py` for radio/path semantic and collate-safe helper logic.
- Trainer second pass:
  - Added `src/kd_sensing/engine/trainer_runtime_helpers.py` for final test evaluation, CSI RMS handoff, epoch setter recursion and dataloader shutdown coordination.
- Updated hotspot metadata/inventory with extracted helper targets and remaining cohesion boundaries.
- Validation:
  - `conda run -n kd_mm_beam python -m compileall -q ...deepsense6g/mmw/trainer...`: passed.
  - `conda run -n kd_mm_beam pytest tests/test_deepsense6g_contract_helpers.py -q`: 5 passed.
  - `conda run -n kd_mm_beam pytest tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py tests/test_csi_modality.py -q`: 154 passed, 45 warnings.
  - `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_beam_label_calibration.py -q`: 29 passed, 2 warnings.
  - `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_epoch_subsampling.py -q`: 116 passed, 1 warning.
  - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: 65 passed.

## Wave 4 JEPA Benchmark

- Captured public compatibility surface from `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark`, `jepa_benchmark_common`, `jepa_benchmark_scenario_d` and `jepa_benchmark_runner`: facade imports for benchmark constants, manifest errors/warnings, scalar/metadata helpers, Scenario D/CxD helpers, runner summary/source helpers and `run_jepa_gps_shortcut_benchmark` remain available.
- Split common helpers:
  - `jepa_benchmark_common_types.py`: `BenchmarkManifestError` and `WarningRecord`.
  - `jepa_benchmark_io.py`: JSON/CSV/path/hash helpers.
  - `jepa_benchmark_scalars.py`: numeric conversion, metric scaling, relative drop, slope and AUC helpers.
  - `jepa_benchmark_metadata.py`: metadata rows, sample ids, batch size, stable seed and case-study row helpers.
- Split Scenario D/CxD analysis:
  - `jepa_benchmark_scenario_d_normalization.py`: Scenario D and CxD suite normalization.
  - `jepa_benchmark_cxd_phase.py`: Scenario D matrix, CxD phase rows, heatmaps and artifact writing.
  - `jepa_benchmark_cxd_dominance.py`: diagnostic records, modality dominance and ResNet/JEPA crossing detection.
  - `jepa_benchmark_cxd_failure_modes.py`: CxD failure decomposition.
  - `jepa_benchmark_scenario_d_metrics.py`: Scenario D/CxD metric columns and synthetic metric row generation.
  - `jepa_benchmark_cxd_helpers.py`: CxD pairing, strict comparability and sorting helpers shared by phase/dominance/failure modules.
- Split runner helpers:
  - `jepa_benchmark_runner_summary.py`: robustness summary, shortcut reliance, analysis bundle reader and case studies.
  - `jepa_benchmark_runner_sources.py`: model metric sources and per-suite metric row generation.
  - `jepa_benchmark_runner_manifest.py`: runner manifest builder.
- Right-sized owners after split:
  - `jepa_benchmark_common.py`: 642 lines, constants plus imported helper surface.
  - `jepa_benchmark_scenario_d.py`: 86 lines, Scenario D/CxD compatibility re-export owner.
  - `jepa_benchmark_runner.py`: 318 lines, benchmark run orchestration.
- Updated architecture tests, maintainer context index and inventory to point CxD phase/dominance/runner helper ownership at the new narrow modules.
- Validation:
  - `conda run -n kd_mm_beam python -m compileall -q ...jepa_benchmark*.py`: passed.
  - `conda run -n kd_mm_beam python -c "from kd_sensing.diagnostics.jepa_gps_shortcut_benchmark import ..."`: passed.
  - `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`: 16 passed.
  - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: 65 passed.

## Wave 5 Consolidation

- Low-value boundary scan:
  - `find src/kd_sensing -type f \( -name '*utils*.py' -o -name '*helpers.py' \)` found only `src/kd_sensing/diagnostics/jepa_benchmark_cxd_helpers.py` and `src/kd_sensing/engine/trainer_runtime_helpers.py`.
  - `jepa_benchmark_cxd_helpers.py` is retained as a CxD-domain shared helper because phase, dominance and failure modules all use the same strict comparability, pairing, sort and crossing helpers.
  - `trainer_runtime_helpers.py` is retained as a runtime helper boundary for final evaluation, CSI RMS config handoff, epoch recursion and dataloader shutdown; no public facade imports from it.
  - No duplicate `utils` aggregation, no no-value compatibility facade and no single-call wrapper introduced by this change were found.
- Keep-and-test / monitor:
  - `src/kd_sensing/losses/jepa.py`, `src/kd_sensing/losses/gps_lidar_bgam_losses.py` and `src/kd_sensing/models/csi_encoder.py` remain keep-and-test targets through `docs/maintainer_context_index.yaml` Wave 5 metadata and the inventory note. They are below hotspot thresholds and domain-cohesive.
- Documentation:
  - README now links `docs/maintainer_context_index.yaml`.
  - `docs/project_surface_inventory.md` lists full JEPA narrow module paths and the three keep-and-test module paths.
- Final validation:
  - `openspec validate right-size-hotspot-modularity --strict`: passed.
  - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`: 65 passed.
  - Focused tests passed:
    - `tests/test_training_io_workflow.py tests/test_epoch_subsampling.py`: 116 passed, 1 warning.
    - `tests/test_preprocessing_formats.py tests/test_config_load_characterization.py`: 10 passed.
    - `tests/test_beambench_image_ae_gps_direct.py`: 7 passed.
    - `tests/test_jepa_gps_shortcut_benchmark.py`: 16 passed.
    - `tests/test_deepsense6g_contract_helpers.py`: 5 passed.
    - `tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py tests/test_csi_modality.py`: 154 passed, 45 warnings.
    - `tests/test_mmw_town10_preparation.py tests/test_beam_label_calibration.py`: 29 passed, 2 warnings.
  - Full `conda run -n kd_mm_beam pytest -q`: 844 passed, 1 failed, 71 warnings. The failure is `tests/test_gps_conditioned_jepa.py::test_gps_query_downstream_configs_load_and_record_metadata`, where unmodified GPS-query experiment configs currently set `gps_query_pool.k_queries: 2` while the test expects `4`.

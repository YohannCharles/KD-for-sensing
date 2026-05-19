## 1. MMW layout and configuration

- [x] 1.1 Add MMW layout helpers for `dataset/MMW/<condition>/Sensor_Data`, `Channel_Data` and `Prepared/<scenario>` without changing DeepSense6G defaults.
- [x] 1.2 Add `configs/preprocess/mmw_town10_skybridge.yaml` with configurable sensor zip, channel zip, condition, scenario, `seq_len`, `pred_len`, `num_beams`, split seed and output root.
- [x] 1.3 Add a CLI entrypoint such as `scripts/mmw/prepare_town10_skybridge.py` that runs through `conda run -n kd_mm_beam python ...` and supports dry-run, force rebuild and config overrides.

## 2. Zip indexing and frame alignment

- [x] 2.1 Implement MMW zip validation and extraction/indexing utilities that reject missing zip paths with absolute-path errors.
- [x] 2.2 Implement sensor frame indexing for Town10 skybridge CAV/RSU agents, including CAV yaml, pcd and camera0-camera3 paths plus RSU yaml, pcd, camera/depth/radar paths when present.
- [x] 2.3 Implement channel file discovery for `_paths.npy` and `_paths.npz`, matching by CAV agent and six-digit frame id.
- [x] 2.4 Implement frame-level skip reason accounting for missing enabled modalities, missing channel files, malformed metadata and non-contiguous frames.

## 3. Channel-derived beam labels

- [x] 3.1 Add channel payload loading that supports `.npz`, dict-like `.npy` and array payloads, with field and shape diagnostics.
- [x] 3.2 Reuse or extend the existing DFT codebook helper to derive 64-beam power vectors from MMW channel/path data without requiring TensorFlow or Sionna at runtime.
- [x] 3.3 Write finite 64-value power txt files under `Prepared/Town10_skybridge_seed24/beam_power/<agent>/<frame>.txt`.
- [x] 3.4 Record channel-to-beam metadata including algorithm version, codebook type, `num_beams`, antenna counts, source channel field and input/output relative paths.
- [x] 3.5 Add deterministic unit tests for channel-to-beam derivation, invalid dimensions and NaN/Inf rejection using small synthetic channel fixtures.

## 4. Manifest, sequences and split artifacts

- [x] 4.1 Generate a frame manifest with agent id, frame id, sensor paths, channel path, beam power path and optional RSU/modal metadata.
- [x] 4.2 Generate sequence CSV windows within a single CAV agent and continuous frame segment, producing `camera*`, `lidar*`, `gps*`, `mmwave*`, `beam*` and `future_beam*` columns.
- [x] 4.3 Implement train/test split by `seq_index` or contiguous segment, and write split metadata with seed, ratio, seq assignment, window counts and beam label distribution.
- [x] 4.4 Write `metadata.json` and `sanity_report.json` with zip provenance, agent/frame counts, skip reasons, modality coverage, channel failures, beam histogram and artifact paths.
- [x] 4.5 Add tests for manifest fields, sequence window alignment, future label alignment and split leakage prevention.

## 5. Dataset loading integration

- [x] 5.1 Register or build an MMW dataset path for `data.dataset.type: mmw` and `data.dataset.scene: town10_skybridge_seed24`.
- [x] 5.2 Support lazy modality loading from prepared MMW CSV so mmWave-only reads only `mmwave*`, `beam*` and `future_beam*`.
- [x] 5.3 Support MMW image+mmWave fusion loading with front camera `camera0` as the compatible image input while preserving extra camera paths in metadata.
- [x] 5.4 Ensure MMW `input_beam`, `target_beam` and `mmwave` tensors keep `[seq_len]`, `[num_pred]` and `[seq_len, 64]` sample shapes.
- [x] 5.5 Add data-loading tests for mmWave-only, image+mmWave fusion and missing disabled modalities using `conda run -n kd_mm_beam pytest ...`.

## 6. Documentation and verification

- [x] 6.1 Update README or preprocessing docs with the MMW Town10 command, expected local zip paths, output directory layout and no-download assumption.
- [x] 6.2 Run targeted tests with `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_mmwave_modality.py tests/test_training_io_workflow.py -q`.
- [x] 6.3 Run a dry-run smoke command with `conda run -n kd_mm_beam python scripts/mmw/prepare_town10_skybridge.py --config configs/preprocess/mmw_town10_skybridge.yaml --dry-run`.
- [x] 6.4 When real zip files are available, run a small real-data smoke prepare and confirm `metadata.json`, `sanity_report.json`, train/test CSV and beam power files are produced.
- [x] 6.5 Run `openspec status --change prepare-mmw-town10-skybridge-dataset` and confirm proposal, design, specs and tasks remain apply-ready.

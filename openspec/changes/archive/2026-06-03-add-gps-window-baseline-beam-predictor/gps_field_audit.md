## MMW GPS Window Field Audit

Date: 2026-06-02

Local ready root: `dataset/MMW/sunny/Prepared`

Ready scenarios with `l5p3_group_safe` train/test splits:

- `Town10_Hroad_seed42`
- `Town10_crossroad_seed24`
- `Town10_curvyroad_seed42`
- `Town10_skybridge_seed24`

Prepared sequence CSV fields used by the GPS window baseline:

- History GPS paths: `gps1..gps5`
- History relative geometry JSON: `geometry1..geometry5`
- History frame ids: `history_frame_ids_json`
- Target/evaluation labels: `future_beam_label1..future_beam_label3`
- Evaluation beam power paths: `future_beam1..future_beam3`
- Scenario/sample metadata: `sample_id`, `target_sample_id`, `scene_slug`, `sensor_scenario`, `condition`, `town`, `agent`

Frame manifest fields available for direct inspection:

- CAV GPS YAML: `gps`
- RSU metadata/YAML paths: `rsu_json`
- Per-frame relative geometry: `relative_geometry_json`
- Beam label and beam power path: `beam_label`, `beam_power_path`
- Channel/path fields: `channel_path`, `channel_fields_json`

Observed direct geometry fields in prepared metadata:

- `relative_range`
- `relative_azimuth`
- `relative_elevation`
- `heading_difference`
- `relative_velocity`
- `local_x`
- `local_y`
- `local_z`

Leakage boundary recorded in implementation:

- Prediction reads only history GPS/geometry metadata, frame ids, codebook config, and optional history beam fallback.
- Future labels and future beam power paths are used only after prediction for evaluation metrics.
- Path/radio/channel oracle fields are rejected by `guard_no_target_oracle`.


# Config protocol index

This file points configuration families to the current protocol and claim documents. It is an index only; do not copy full result tables here.

- Mainline model catalog: `docs/mainline_model_catalog.md`
- Experiment protocols: `docs/experiment_protocols.md`
- Result and claim registry: `docs/result_claims_registry.md`

| family | configs | status in protocol table |
| --- | --- | --- |
| DeepSense6G GPS+LiDAR BGAM | `configs/deepsense6g_gps_lidar_bgam.yaml` | quick validation formal; support/query BGAM, Top8 candidate support only |
| MMW Town GPS v2 | `configs/mmw_town_gps_adapter_v2.yaml` | formal diagnostic; `within_scene_train` is sanity/upper-bound style evidence |
| MMW Town GPS+LiDAR BGAM | `configs/mmw_town_gps_lidar_bgam.yaml` | quick validation formal; consumes local GPS v2 prior and LiDAR cache |
| BEV-Fusion 2604 | `configs/fusion/experiments/bev_fusion_2604/` | formal, lowmem, smoke and ablation variants are separated |
| JEPA Image+GPS | `configs/fusion/experiments/jepa_image_gps/` | BeamBench-fair and 2604-style families must not be mixed |
| CSI hardening | `configs/csi/hardening_matrix/`, `configs/fusion/csi_hardening_matrix/` | formal matrix plus debug matrix |
| Difficulty profiles | `configs/difficulty/` | diagnostic/training profiles, not new modalities |
| Diagnostics / benchmark manifests | `configs/diagnostics/` | smoke, benchmark or diagnostic-only as declared by manifest |

All runtime outputs remain local ignored artifacts under `outputs/`, `outputs/cache/`, `logs/` or explicit manifest output roots.

# Config protocol index

This file points configuration families to the current protocol and claim documents. It is an index only; do not copy full result tables here.

- Mainline model catalog: `docs/mainline_model_catalog.md`
- Experiment protocols: `docs/experiment_protocols.md`
- Result and claim registry: `docs/result_claims_registry.md`

| family | configs | status in protocol table |
| --- | --- | --- |
| MMW Town GPS v2 | `configs/mmw_town_gps_adapter_v2.yaml` | formal diagnostic; `within_scene_train` is sanity/upper-bound style evidence |
| BEV-Fusion 2604 | `configs/fusion/experiments/bev_fusion_2604/` | formal, lowmem, smoke and ablation variants are separated |
| JEPA Image+GPS | `configs/fusion/experiments/jepa_image_gps/` | 31 retained entity YAML; BeamBench-fair, 2604-style, predictive, geometry-prior and safe-rerank families must not be mixed |
| RBMA missing workflow | `configs/fusion/experiments/rbma_missing_workflow/`, `configs/fusion/experiments/rbma_missing_workflow_strong_encoders/` | local/manual pending evidence; strong-encoder overlays require local checkpoint placeholders |
| Scene31 generated/local overlays | `configs/scene31/` | generated queues keep manifests/base/generators; retained entity YAML are local/manual or pending evidence inputs |
| CSI hardening | `configs/csi/hardening_matrix/`, `configs/fusion/csi_hardening_matrix/` | formal matrix plus debug matrix |
| Difficulty profiles | `configs/difficulty/` | diagnostic/training profiles, not new modalities |
| Diagnostics / benchmark manifests | `configs/diagnostics/` | smoke, benchmark or diagnostic-only as declared by manifest |

Retired BGAM configs and the legacy viewer manifest config are intentionally absent; old paths fail fast instead of being regenerated as virtual configs.

All runtime outputs remain local ignored artifacts under `outputs/`, `outputs/cache/`, `logs/` or explicit manifest output roots.

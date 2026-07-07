# Agent Context Atlas

| Surface | Lifecycle | Spec / doc | Owner | Focused tests | Caveat |
| --- | --- | --- | --- | --- | --- |
| final C2 / U-MaskBeamJEPA | current | `openspec/specs/final-c2-ablation-v1/spec.md`, `openspec/specs/u-mask-beam-jepa/spec.md` | `src/kd_sensing/models/u_mask_beam_jepa.py` | `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py -q` | branch/loss variants protected this change |
| MMW Town GPS v2 | current/supporting | `openspec/specs/mmw-town-gps-adapter-v2/spec.md` | `src/kd_sensing/engine/mmw_town_gps_v2.py` | `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q` | no real dataset content in source |
| physics-informed MMW | current/supporting | `openspec/specs/physics-informed-mmw-beam-baseline/spec.md` | `src/kd_sensing/models/physics/` | `conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py -q` | outputs ignored |
| CSI hardening | current/supporting | `openspec/specs/csi-hardening-experiment-matrix/spec.md` | `configs/csi/`, `configs/fusion/csi_hardening_matrix/` | `conda run -n kd_mm_beam pytest tests/test_csi_modality.py -q` | configs protected |
| public CLI surface | current | `docs/project_surface_inventory.md` | `src/kd_sensing/diagnostics/cli_surface.py` | `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q` | no deleted CLI aliases |
| diagnostics | current/supporting | `docs/agent_context/diagnostics.md` | `src/kd_sensing/diagnostics/` | `conda run -n kd_mm_beam pytest tests/test_project_surface_doctor.py -q` | historical JEPA diagnostics retired |
| retired nonmainline | retired-tombstone | `openspec/specs/retired-route-summary/spec.md` | registry/config migration guards | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | no wrapper, stub or virtual config |

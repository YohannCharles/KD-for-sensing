# Result Claims Registry

| claim_id | Subject | Provenance | Status | Upgrade gate |
| --- | --- | --- | --- | --- |
| `CLAIM-FINAL-C2-UMASK-PENDING` | final C2 / U-MaskBeamJEPA missing-modality mainline | `configs/fusion/u_mask_beam_jepa_smoke.yaml`, `configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml`, final C2 launcher/summary | `pending` | real checkpoint, split, seed, label space, metric profile and missing-condition evidence must be complete |
| `CLAIM-SCENES31-34-MAIN-PENDING` | Scene31-34 pooled missing-modality local workflow | `scripts/run_scenes31_34_main.sh`, `scripts/generate_scenes31_34_main.py`, `python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact <artifact>` | `pending` | n>=3 comparable seeds, fresh eval, classifier/external caveats and paper tables complete |
| `CLAIM-PCPG-PENDING` | PCPG/radar-protected branch balance | `scripts/launch_pcpg_radar_balance_v1.py`, `scripts/summarize_pcpg_radar_balance_v1.py` | `pending` | complete local run manifest and strict comparison fields |
| `CLAIM-BPRR-PENDING` | BPRR reliability router | `scripts/launch_bprr_reliability_router_v1.py`, `scripts/summarize_bprr_reliability_router_v1.py` | `pending` | calibration, gate regularization and oracle upper-bound evidence complete |
| `CLAIM-MMW-GPS-V2-PENDING` | MMW Town GPS-only v2 | `configs/mmw_town_gps_adapter_v2.yaml`, `kd-sensing-mmw-town-gps-v2` | `pending / supporting` | local MMW dataset evidence and protocol summary complete |
| `CLAIM-PHYSICS-MMW-PENDING` | physics-informed MMW | `configs/fusion/physics_informed_mmw*.yaml`, `kd-sensing-inspect-mmw-physics` | `pending / supporting` | real data or declared synthetic smoke boundary plus focused tests |
| `CLAIM-CSI-HARDENING-PENDING` | CSI hardening | `configs/csi/`, `configs/fusion/csi_hardening_matrix/` | `pending / supporting` | hardening matrix summaries and focused tests complete |

Historical claims from Image+GPS JEPA, BeamBench, BEV-Fusion 2604, Vision-Position, old RBMA/KD/BTAPA/weakKD, WCL and TII are retired or deleted in post-C2 cleanup. They may be cited only as historical lineage; they do not provide current provenance and cannot upgrade a `pending` mainline claim.

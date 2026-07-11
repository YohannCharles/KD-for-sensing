# Mainline Model Catalog

| id | role | implementation | protected input | status | claim |
| --- | --- | --- | --- | --- | --- |
| `u-mask-beam-jepa-final-c2` | 当前默认缺失模态主线 | `src/kd_sensing/models/u_mask_beam_jepa.py` + `src/kd_sensing/losses/u_mask_beam_jepa.py` | `configs/fusion/u_mask_beam_jepa_smoke.yaml`, `configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml`, `scripts/launch_final_c2_ablation_v1.py`, `scripts/summarize_final_c2_ablation_v1.py` | `current / pending evidence` | `CLAIM-FINAL-C2-UMASK-PENDING` |
| `pcpg-radar-balance` | 保留的 U-Mask branch-balance follow-up | `scripts/launch_pcpg_radar_balance_v1.py`, `scripts/summarize_pcpg_radar_balance_v1.py` | generated configs under ignored outputs | `current local/manual` | `CLAIM-PCPG-PENDING` |
| `bprr-reliability-router` | 保留的 reliability/router follow-up | `scripts/launch_bprr_reliability_router_v1.py`, `scripts/summarize_bprr_reliability_router_v1.py` | generated configs under ignored outputs | `current local/manual` | `CLAIM-BPRR-PENDING` |
| `overnight-branch-router-v2` | 已归档结果的只读复盘 | `scripts/summarize_overnight_branch_router_v2.py` | existing ignored result roots | `supporting parser / historical result context` | `CLAIM-OVERNIGHT-BRANCH-ROUTER-V2` |
| `mmw-town-gps-v2` | MMW future/current supporting workflow | `kd_sensing.engine.mmw_town_gps_v2` | `configs/mmw_town_gps_adapter_v2.yaml` | `protected MMW` | `CLAIM-MMW-GPS-V2-PENDING` |
| `physics-informed-mmw` | physics-informed MMW baseline | `src/kd_sensing/models/physics/` | `configs/fusion/physics_informed_mmw*.yaml` | `protected MMW` | `CLAIM-PHYSICS-MMW-PENDING` |
| `csi-hardening` | CSI hardening workflow | `src/kd_sensing/models/csi_encoder.py` and CSI configs | `configs/csi/`, `configs/fusion/csi_hardening_matrix/` | `protected CSI` | `CLAIM-CSI-HARDENING-PENDING` |

Historical rows retired by `prune-post-c2-nonmainline-surface`: Image+GPS JEPA, BeamBench, BEV-Fusion 2604, Vision-Position, old RBMA/KD/BTAPA/weakKD sweep, WCL/TII source-audit paths and standalone architecture-summary CLI. They are not current recommended workflow rows and should not be used as provenance for new claims.

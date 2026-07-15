# Experiment Matrix

## Current Runs

| Priority | Experiment | Command | Evidence status |
| --- | --- | --- | --- |
| P0 | final C2 / U-MaskBeamJEPA smoke | `conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/u_mask_beam_jepa_smoke.yaml` | `pending` |
| P0 | U-Mask missing-modality eval matrix | `conda run -n kd_mm_beam kd-sensing-eval-u-mask-matrix --config configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml --checkpoint outputs/local/best.pth` | `pending / local checkpoint required` |
| P1 | Scene31-34 main local/manual | `bash scripts/run_scenes31_34_main.sh` | `pending`, outputs ignored |
| P1 | final C2 ablation launcher | `conda run -n kd_mm_beam python scripts/launch_final_c2_ablation_v1.py --help` | `local/manual` |
| P2 | PCPG radar balance | `conda run -n kd_mm_beam python scripts/launch_pcpg_radar_balance_v1.py --help` | `pending` |
| P2 | BPRR reliability router | `conda run -n kd_mm_beam python scripts/launch_bprr_reliability_router_v1.py --help` | `pending` |
| MMW | MMW Town GPS v2 | `conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --help` | protected supporting workflow |
| MMW | physics-informed MMW | `conda run -n kd_mm_beam kd-sensing-inspect-mmw-physics --help` | protected supporting workflow |

## Historical / Retired

Image+GPS JEPA, BeamBench, BEV-Fusion 2604, Vision-Position, WCL/TII source-audit and old RBMA/KD/BTAPA/weakKD sweep entries are historical or deleted. They must not appear as current recommended commands, and old missing YAML references are only valid in lines marked historical/retired.

H5/P1 Scene31-34 temporal matrix runs produced before group-safe split enforcement are `not_comparable`, not current evidence. They must not be rerun through the legacy per-sample split or promoted until the identity, validation/test and normalization gates in `docs/experiment_protocols.md` pass.

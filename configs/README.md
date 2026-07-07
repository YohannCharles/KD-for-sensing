# Config Surface

当前配置面只保留主线、MMW/CSI、通用 supervised/preprocess 和治理 smoke。

| Family | Paths | Lifecycle |
| --- | --- | --- |
| U-MaskBeamJEPA / final C2 | `configs/fusion/u_mask_beam_jepa_*.yaml`, `configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml` | current mainline |
| MMW / physics-informed MMW | `configs/mmw_town_gps_adapter_v2.yaml`, `configs/fusion/physics_informed_mmw*.yaml`, `configs/preprocess/mmw_*.yaml` | protected MMW workflow |
| CSI hardening | `configs/csi/`, `configs/fusion/csi_hardening_matrix/` | protected CSI workflow |
| canonical supervised | `configs/{image,radar,gps,lidar,mmwave,csi}/`, `configs/fusion/*.yaml` | current reusable baselines |
| preprocessing | `configs/preprocess/*.yaml` | current dataset/cache preparation |
| diagnostics | `configs/diagnostics/amber_lite_missing_modality_eval.yaml` | current smoke/evaluation diagnostic |

历史 BEV-Fusion 2604、Image+GPS JEPA、BeamBench、Vision-Position、RBMA/KD/BTAPA/weakKD sweep 实体配置已退役或删除；不要恢复为 current YAML 或 virtual config。

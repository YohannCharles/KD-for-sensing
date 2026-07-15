# Current Research Brief

## 当前主线

当前主线是 final C2 / U-MaskBeamJEPA 缺失模态波束预测：围绕 `configs/fusion/u_mask_beam_jepa_smoke.yaml`、`configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml`、final C2 launcher/summary 和 `kd-sensing-eval-u-mask-matrix` 收敛证据链。MMW/CSI 保留为 current supporting dataset workflow；MMW 可作为当前数据实验 campaign，但不替代默认主线。

## 冻结方法

本轮冻结 U-MaskBeamJEPA 已存在 fusion/router/loss 分支，不删 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router`。旧 Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、RBMA/KD/BTAPA/weakKD sweep 已退役为历史语境，不再作为当前推荐实验。

## 不要追

不要追旧 JEPA benchmark、旧 BeamBench Table III substitute、BEV-Fusion 2604、Vision-Position 或 WCL source-audit CLI 的补跑；这些只能作为 historical/pending note。不要从 `outputs/`、`logs/`、cache、checkpoint 或 archive 反推 current claim。

## Claim 升级条件

Claim 升级必须先过 `docs/result_claims_registry.md` 和 `docs/experiment_protocols.md`：真实 checkpoint、split、label space、metric profile、difficulty digest、seed 和 provenance 都齐全后，`pending` 或 `mock/smoke` 才能升级。`mock/smoke` 只能验证 schema，不产生论文数值。

## 下一步高价值实验

1. 补齐 final C2 / U-MaskBeamJEPA 多 seed evidence，并用 `kd-sensing-eval-u-mask-matrix` 更新缺失模态矩阵。
2. 保持 MMW GPS v2、physics-informed MMW 和 CSI hardening focused tests 绿色，为后续真实数据集工作留出入口。
3. 只在 claim registry 需要时运行 `kd-sensing-paper-export`，不要把 pending/historical 行混入主表。

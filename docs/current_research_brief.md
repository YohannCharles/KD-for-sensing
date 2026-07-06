# Current Research Brief

本简报帮助 agent 快速理解当前研究方向。它只做一页 orientation，不替代 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md` 或 OpenSpec specs。

## 当前主线

- 多模态少样本跨场景 beam prediction 是主问题，当前实现围绕 `src/kd_sensing` 包、`model.primary` 配置和 `kd-sensing-*` console scripts。
- Image+GPS JEPA query-pool、paired supervised/random controls、BeamBench-fair 和 2604-style 口径仍是核心对照面。
- Predictive JEPA robustness 是 pending 主线，只有完整 clean anchor 加 `image_missing`、`image_noise`、`gps_noise` stress-curve 证据后才能升级 claim。
- 缺失模态方向聚焦 U-MaskBeamJEPA、AMBER-lite、AMBER full architecture、RMBP-MM、TII-VLRG-style 和 Scene31/Scene31-34 local/manual workflows。
- MMW/CSI 方向聚焦 MMW GPS v2、physics-informed MMW、CSI hardening 和只读诊断。

## 冻结方法和主要对照

- 普通训练、评估、预处理和诊断优先使用 `pyproject.toml` 中的 `kd-sensing-*` console scripts。
- 普通 baseline 优先走 `model.primary.type: modular_sequence` 或组件 registry；whole-model exception 需要 current spec 或 active change 说明。
- BeamBench-fair、2604-style、Scene31/Scene31-34、MMW GPS v2 和 CSI hardening 不能跨 family 混比；比较前先查 `docs/experiment_protocols.md`。
- 正式结果只从 reviewed claim、明确 provenance、strict comparability、seed/checkpoint/split/metric 字段和必要 caveat 升级。

## 不要追的路线

以下路线已退役或只能作为历史、migration guard、防回流说明出现，不得恢复为当前 CLI、config、registry、package facade 或 quickstart：旧 KD、HiST/Hist、standalone Top8 selector、GPS residual、camera residual、BGAM、viewer manifest、仓库级 Gradio viewer、Raymobtime s008、AMR-Net_gps_image、JEPA-MSAC、CRAF/MARF/G2D 和 Multimodal-NF。

## Claim 升级条件

- 必须指向 `docs/result_claims_registry.md` 中的 claim status，不能把 pending、mock/smoke、upper-bound、historical ablation、blocked official reproduction 或 not-comparable candidate 写成正式结论。
- 必须记录 config 或 runner、commit 或 run date、dataset/scenes、split protocol、target/label space、metric profile、seed、checkpoint provenance 和 caveat。
- 缺失模态、stress 或 benchmark claim 还需要 paired seed、strict comparability、stress provenance、difficulty digest 和 warning/next action。
- Dashboard、HTML、JSONL ledger、paper export 草稿和本地 summary 都只是证据入口；它们不会自动改写正式 claim registry。

## 下一步高价值实验

- 补齐 BeamBench-fair Image+GPS JEPA paired control 矩阵，并保持同 split、同 label space、同 metric profile。
- 跑真实 Predictive JEPA robustness manifest，覆盖 clean anchor 和三类默认 stress curves。
- 收口 Scene31-34 main missing-modality all-baseline evidence，重点补 classifier/external/fresh eval、missing-count、per-scene、compute 和 paper table checklist。
- 继续 MMW GPS v2 / physics-informed MMW / CSI hardening 的 strict metadata、local output boundary 和 claim caveat 审计。
- 用 `kd-sensing-project-surface-doctor`、JEPA visual analysis、GPS shortcut benchmark 和 paper export 作为只读诊断，不把生成产物提交。

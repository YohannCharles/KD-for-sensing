# Mainline Experiment History

## Post-C2 Decision

After final C2, the maintained research surface is final C2 / U-MaskBeamJEPA missing-modality beam prediction, with MMW/CSI retained as future/current supporting dataset workflows. This cleanup preserves the useful lesson from the retired branches: evidence has to be family-consistent, split-consistent and claim-gated before promotion.

## Migrated Historical Notes

- Image+GPS JEPA and GPS-query variants were useful exploration but are now historical; do not rerun the deleted JEPA visual/GPS shortcut CLI as current evidence.
- BeamBench/Arnold22 substitutes, BEV-Fusion 2604, Vision-Position, WCL/TII audits and old RBMA/KD/BTAPA/weakKD sweep were removed from current workflow because they either lacked strict provenance, duplicated the current U-Mask direction, or depended on local-only artifacts.
- Scene31 one-shot runbooks and summaries were collapsed into the retained Scene31-34 main/final C2 launchers plus package diagnostics. Old conclusions stay caveats, not promoted claims.

All historical outputs remain local under ignored `outputs/` or archive context; no checkpoint, cache or log is part of source provenance.

## BeamBench Reproduction Retirement

2026-07-15 将旧根目录 BeamBench 复现报告、数据结构、patch notes 和运行流水账从 current surface 删除。官方复现当时因缺少官方 raw test data、权重和完整源码而 blocked；本地 Image AE + GPS substitute、future target、scene31-only、mock/smoke 和 `test_as_validation` 记录均属于 historical/not-comparable evidence，不能恢复为当前 claim。唯一权威状态保留在 `docs/result_claims_registry.md`，任何重新复现都必须另开 change 并使用当前独立 validation/final test 与 provenance gate。

## H5/P1 Temporal Evidence Invalidation

2026-07-15 的只读 split 复现发现，旧 H5/P1 launcher 对重叠 temporal window 使用逐样本 label-stratified split。Scene31-34 的全部 116 个 sequence group 均跨越 train、validation 或 test，且任意两 split 间存在重复历史/target 帧路径。因此修复前 H5/P1 结果统一标记为 `not_comparable`；只有完成 group-safe split、独立 validation/final test、train-only normalization 和身份审计后，新的多 seed 结果才可重新申请 claim 晋级。

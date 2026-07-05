## Why

项目已经有 run index、research dashboard、claim registry 和 paper export，但实验结果从本地 run 到论文表格之间仍需要人工拼接 provenance。为了避免 pending/mock/upper-bound 误入正式结论，需要更硬的 claim doctor、run card 和 paper export gate。

## What Changes

- 增加 claim doctor，输出每个 pending/unverified/not_comparable claim 缺失的 provenance 字段和下一步动作。
- 增加 run card，绑定 command、git commit、config digest、data/split digest、checkpoint provenance、metrics 和 claim candidate。
- 强化 paper export gate，确保 pending/mock/smoke/historical/upper-bound/not_comparable 默认不进入主表。
- 增加只读 dashboard，把 run index、claim candidates、OpenSpec active status、资源状态和 paper readiness 汇总到 ignored output root。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `research-claim-harvester`: 增加 claim doctor、run card 和 upgradable candidate 要求。
- `experiment-run-index`: 增加 run card 所需 provenance 字段的读取和引用边界。
- `paper-artifact-export`: 强化主表导出 gate。
- `mainline-experiment-documentation`: 增加 dashboard/readiness 与 claim registry 同步要求。

## Impact

- 可能扩展 `kd-sensing-research-dashboard`、run index、paper export 或新增只读诊断命令。
- 输出仍写 ignored `outputs/analysis/`、`outputs/paper_artifacts/` 或用户显式路径。
- 不自动修改 `docs/result_claims_registry.md`，除非用户明确要求。

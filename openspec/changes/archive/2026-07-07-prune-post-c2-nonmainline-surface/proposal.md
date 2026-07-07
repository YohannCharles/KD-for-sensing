## Why

当前仓库已经围绕 final C2 / U-MaskBeamJEPA 缺失模态波束预测收敛，但源码、脚本、配置和文档仍保留大量历史实验、一次性诊断和非主线复现表面。现在需要在不破坏后续 MMW 数据集工作、不误删主线 YAML/manifest、不动 U-MaskBeamJEPA 暂保留 fusion 分支的前提下，分波次退役低价值支持面，让主线更轻、更可验证。

## What Changes

- 建立 post-C2 清理边界：主线保留 final C2 / U-MaskBeamJEPA、缺失模态评估矩阵、当前 claim/evidence 链、MMW 数据集与 MMW 相关 workflow，以及仍被主线引用的 YAML/manifest。
- **BREAKING** 删除或降级非主线 package CLI、一次性研究脚本、历史报告入口和可由主线 generator/manifest 替代的本地 runbook；不提供同名兼容 wrapper。
- **BREAKING** 删除或归档不再服务当前主线的 Image+GPS JEPA 诊断、BeamBench/2604/Vision-Position 复现、旧 RBMA/KD/BTAPA/weakKD overlay、历史 Scene31 sweep 和相关测试入口；删除前必须记录当前引用、替代入口和回滚方式。
- 明确保留 MMW 支线：`data/mmw`、MMW dataset/preparation、MMW GPS v2、physics-informed MMW、CSI hardening、MMW/CSI configs、测试和 package CLI 继续作为 future dataset workflow 维护。
- 明确保留主线 YAML/manifest：凡被 final C2、current Scene31/Scene31-34 evidence、claim registry、experiment matrix、OpenSpec current spec、focused tests 或用户标记为主线输入的 YAML/manifest 不在本 change 删除范围。
- 明确暂不修改 U-MaskBeamJEPA 中已经存在但本轮不启用的 fusion 分支、router 分支和相关 forward/loss 开关；只允许文档标注后续单独评估触发条件。
- 更新 inventory、README/docs/OpenSpec current specs 和架构边界测试，使清理后的 current surface 不再把被删除入口描述为可运行或推荐入口。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-surface-cleanup`: 增加 post-C2 主线清理边界、保留清单、删除候选证据要求、分波次删除策略和禁止误删范围。
- `project-entrypoint-lifecycle`: 收敛 package CLI、`scripts/` runbook 和一次性诊断入口生命周期；明确 MMW 入口保留，非主线入口删除后不提供兼容 alias。
- `project-health-guardrails`: 增加清理验收检查，覆盖 protected YAML/manifest、MMW 保留、U-Mask fusion 分支暂不触碰、退役入口回流和 stale reference。
- `mainline-experiment-documentation`: 调整当前主线叙述与文档索引，使 final C2 / U-MaskBeamJEPA 缺失模态主线和保留 MMW future workflow 的边界一致。

## Impact

- 影响源码表面：`src/kd_sensing/diagnostics/`、`src/kd_sensing/baselines/`、`src/kd_sensing/cli/`、部分非主线 `src/kd_sensing/models/`、`scripts/`、`configs/`、`tests/`、README/docs、OpenSpec specs 和 architecture guardrail。
- 不影响范围：`dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard 产物、MMW/CSI 数据与 workflow、主线 final C2 YAML/manifest、当前 claim/evidence 输入、U-MaskBeamJEPA 现有 fusion 分支实现。
- 验证重点：`openspec validate prune-post-c2-nonmainline-surface --strict`、`openspec validate --all --strict`、`make verify-quick`、`make verify-cli-config`、`make verify-compile`，以及按删除波次追加 focused tests。

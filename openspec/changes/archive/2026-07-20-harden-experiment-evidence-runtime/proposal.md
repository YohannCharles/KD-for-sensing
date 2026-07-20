## Why

当前 MMW/T2 工作流在训练、固定 mask 评估、汇总和本地 screening 之间复制了部分证据逻辑。审查发现 profile、归一化、样本身份、指标语义和 partial 输出可在部分路径中失去约束；同时若干基线和数据输入问题会影响比较公平性或在运行很晚才失败。

本变更将这些边界收敛为可验证、fail-closed 的 current workflow，并消除已确认的热路径浪费，不恢复任何已退役路线。

## What Changes

- 统一 checkpoint、profile、candidate、normalization、split 和 mask provenance 的校验；不完整、被篡改或 partial 的 evidence 不得进入正式汇总。
- 修复 all-weather evaluator 与 package evaluator 的身份和归一化处理漂移，明确指标语义并拒绝混合不同定义。
- 修复 AMBER token padding、T2 inactive head/router 分支和 CMA 的稳定样本身份，使 baseline 与受控消融的参数和 mask 语义可审计。
- 收紧 MMW/DeepSense6G 数据与 CLI 契约：早期拒绝缺失资源、非法标签、未知参数和越界路径；减少重复标签 I/O，并保持可复现的增强和 runtime 设置。
- 精简评估热路径和运行生命周期：避免重复编码/逐 batch 同步、无用全量输出收集和遗留 worker；使 launcher manifest/status 写入原子且可恢复。
- 补齐 current README、维护路由、环境定义和验证入口，使 H4 generated workflow 与 public CLI 的实际行为一致。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `training-evaluation-runtime`: 训练、评估和 development workflow 必须使用一致的 runtime/provenance 边界，并安全管理 dataloader 生命周期。
- `experiment-artifact-registry`: evidence artifact 必须可验证、完整且明确标记 partial/development 状态。
- `mmw-baseline-multiseed-robustness-evidence`: MMW 汇总必须拒绝 profile、checkpoint、mask、样本或指标身份不一致的输入。
- `dataset-loader-behavior`: current 数据加载必须在读取期校验四模态资源、标签、路径和随机性契约。
- `project-entrypoint-lifecycle`: public CLI 必须拒绝未知参数，并准确表达 MMW generated-config workflow。
- `u-mask-beam-jepa`: active head、router 和 CMA identity 行为必须与声明的训练分支和参数统计一致。
- `t2-baseline-surface`: AMBER-Full/T2 baseline 的 token-mask 和参数语义必须保持比较公平性。

## Impact

- 影响 `src/kd_sensing/engine/`、`eval/`、`evaluation/`、`data/`、`models/`、`losses/`、`config/` 和 `utils/` 的 current owners。
- 影响 retained MMW screening/evaluation/summary scripts、三个 package CLI、相关 canonical 文档和 focused tests。
- 不增加第三方运行依赖；generated configs、split、logs、cache、metrics 和 checkpoint 仍只写入 ignored `outputs/`。

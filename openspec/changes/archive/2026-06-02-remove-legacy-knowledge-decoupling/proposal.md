## Why

旧知识解耦路线把跨场景知识简单拆成 shared 与 private 分支，并通过 orthogonality、shared scene confusion、private scene preservation 等 loss 约束迁移；多轮迁移实验已经显示目标场景精度长期在约 10% 徘徊，且容易形成 source prior collapse。现在需要把这条失败路线从主代码、配置、OpenSpec 契约和本地实验产物中退役，避免后续工作继续围绕不可行 baseline 消耗时间。

## What Changes

- **BREAKING** 删除 `v2_shared_private`、`shared_private`、`v3_decoupled`、`decoupled` 等旧 shared/private 简单解耦变体的模型注册、配置默认值、LOSO planner 支持、runner 默认值和相关测试期待。
- **BREAKING** 删除或停用旧解耦专属 loss 与诊断：orthogonality、shared scene confusion、private scene preservation、旧 shared/private cosine 诊断，以及只为这些 loss 服务的 scene classifier 输出。
- **BREAKING** 更新 HiST-Beam 适配契约：adapter/prototype、path/radio、image-only、history residual、V7 residual 等保留路线不得再依赖 `v3_decoupled` 作为 source checkpoint 或默认 baseline。
- 更新跨场景 LOSO workflow 的默认矩阵、summary comparison 和 quick conclusion 逻辑，使失败的 `v3_decoupled` 不再作为主 baseline；需要 baseline 时使用明确命名的 source-only、legal image-only 或 residual baseline。
- 清理源码控制内的旧配置、脚本和 README 示例，移除以旧解耦路线为核心的运行入口、默认 variant 列表和文档说明。
- 清理本地产物边界内的旧失败实验结果：对 `outputs/`、`logs/` 中匹配旧解耦路线的目录先生成删除清单，再删除对应日志、progress、metrics、summary、checkpoint/cache 等生成物；不提交这些本地产物。
- 保留归档 OpenSpec 中对旧失败路线的历史记录，但现行 specs 必须标记该路线已退役，不再要求实现可运行。

## Capabilities

### New Capabilities

<!-- 无。此次变更退役失败路线并收紧现有契约，不引入新的算法能力。 -->

### Modified Capabilities

- `hist-beam-cross-scene-adaptation`: 退役旧 `v2_shared_private`/`v3_decoupled` 简单 shared/private 解耦变体、专属 loss、模型输出和默认 source/adaptation 依赖。
- `cross-scene-loso-workflow`: 更新 LOSO 默认 variant、source checkpoint 选择、summary comparison 和 quick conclusion，不再以 `v3_decoupled` 作为主 baseline。
- `path-prototype-hist-beam-adaptation`: 移除 path prototype 路线对旧 `v3_decoupled` source-only baseline 或旧 shared/private loss 的要求，保留 path/radio/prototype 自身诊断。
- `radio-semantic-hist-beam-adaptation`: 移除 radio prototype 路线对旧简单解耦 baseline 与 scene confusion/private preservation loss 的要求。
- `project-architecture`: 明确本次退役允许清理用户指定的本地 `outputs/`、`logs/` 失败实验产物，但清理动作必须生成清单并限于未纳入源码的运行产物。

## Impact

- 受影响代码：`src/kd_sensing/models/fusion/hist_beam.py`、`src/kd_sensing/engine/hist_beam_losses.py`、`src/kd_sensing/engine/hist_beam_loso_config.py`、LOSO summary/comparison、image-only source mapping、prototype source checkpoint 决策、配置解析和相关测试。
- 受影响配置/脚本：`configs/hist_beam/*` 中包含 `v2_shared_private`、`v3_decoupled`、orthogonality、scene_confusion、scene_private 的配置，`scripts/run_*` 中以旧解耦为 baseline 的矩阵脚本，以及 README 示例。
- 受影响 OpenSpec：现行 specs 中关于 shared/private 简单解耦、`v3_decoupled` baseline、旧 loss 组合和旧矩阵覆盖的要求需要删除或替换。
- 受影响本地产物：`outputs/`、`logs/` 中旧 shared/private 解耦、V3/V2 baseline、以 `v3_decoupled` 为 source baseline 的失败迁移结果；清理前必须列出路径和匹配原因。
- 验证重点：架构边界测试、HiST-Beam 模型/loss/LOSO 相关测试、保留路线的 smoke test、OpenSpec strict validate，以及确认清理清单未包含当前 image-only legal probe 与 target-shot geometry residual foundations 的活跃产物。

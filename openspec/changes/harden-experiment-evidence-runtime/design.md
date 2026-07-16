## Context

MMW 的训练、package evaluation、fixed-mask matrix、screening launcher 和 summary 目前各自拥有一部分 provenance、数据验证和指标逻辑。它们的共同目标是受限的 T2/baseline current surface，但重复实现使 profile、normalization、partial 结果和指标定义可能在路径之间漂移。模型侧还存在 AMBER spatial padding 与 U-Mask inactive branch 的语义偏差。

本设计与 `establish-h4-t2-design-screen` 的 outer-test 隔离修复配合；本变更不启动训练、不修改已有本地产物，也不恢复任何 retired route。

## Goals / Non-Goals

**Goals:**

- 让训练、评估、fixed-mask 和 summary 对同一 evidence identity 使用同一套 fail-closed 校验。
- 修复已确认的 baseline/mask/metric 正确性问题，并使不参与训练的参数不进入 optimizer 语义。
- 在不改变 retained public CLI 数量的前提下，拒绝未知命令行参数和未知 override。
- 将热路径的无效计算、重复 I/O、全量输出缓存和 worker 泄漏降到当前最小实现。
- 提供可复现的环境定义和 focused regression，所有 Python 验证继续使用 `kd_mm_beam`。

**Non-Goals:**

- 不运行新的 40-epoch 或多 seed 训练，不重新解释既有 evidence。
- 不增加模型、数据集、public CLI、第三方运行依赖或 compatibility route。
- 不将生成配置、split、cache、日志、checkpoint 或结果纳入 tracked source。
- 不把 development 或 partial evidence 升级为 claim。

## Decisions

### 1. 以 canonical payload digest 作为 evidence identity

profile 与 generated design config 的 digest 由一个去除自指 hash 字段后的规范化 payload 计算。checkpoint、probe、manifest、fixed-mask evaluator 和 summary 均比较实际重新计算的值，而不是只信任记录字符串。profile canonical values 同时必须和实际 training/scheduler 字段一致。

直接信任 manifest 中已写入的 hash 被拒绝，因为它不能发现 YAML 在 dry-run 后被修改。

### 2. fixed-mask evaluator 复用 package 侧的 checkpoint 准备规则

fixed-mask evaluator 使用 checkpoint metadata 的 profile、GPS 和 normalization artifact 校验，并将加载到的 scaler 传入 dataloader builder。正式矩阵没有完整 artifact 时失败；partial 模式保留为显式 development 输出，summary 拒绝它。

复制 package evaluator 全流程被拒绝，因为 fixed-mask 需要自己的 mask loop；只提取/复用 checkpoint preparation 与 validators。

### 3. 指标名称绑定数学定义

训练 validation 与固定 mask 的 `adba` 统一使用 progressive top-3 DBA。top-1 proximity 指标保留为显式命名字段，linear/circular top-k 共用可选 distance-mode helper。每行 evidence 写入 metric profile，汇总要求一致。

保留同名不同公式被拒绝，因为表面兼容会掩盖不可比较结果。

### 4. 模型按有效 token/训练分支处理

AMBER 将每个 modality 的原始 spatial-token 计数转换为逐 token availability，padding token 一律 key-padding；其 pooled diagnostic 只平均有效 token。U-Mask 保留 inactive head 以避免不必要的 checkpoint surface 破坏，但冻结其参数、从 optimizer 排除，并在 metadata 中说明 active/inactive branch。`reliability_mean` 保留 router diagnostics，但零权重时不构造 oracle-loss graph。

完全删除 inactive head 被拒绝：本地 current checkpoint 仍可由严格 loader 使用，而冻结/排除已能消除训练与参数统计歧义。

### 5. 数据和 CLI 在入口 fail closed

MMW CSV/preflight/loader 使用相同的最小 schema：连续四模态列、BS GPS、label 范围、派生雷达资源与受根目录限制的相对路径。DeepSense future-beam labels 在数据集构造期验证并缓存 argmax。CLI 仅接受显式 `--override` 与无前缀 `key=value` 形式的已知路径，其他 unknown 参数立即失败。

运行时延迟失败或 silent default 被拒绝，因为它会造成浪费训练和不可审计数据污染。

### 6. 性能优化保持语义等价

evaluation 默认流式累计 metrics，不保留完整 logits/labels/metadata；只有明确要求时才 capture。固定 mask evaluator 复用每 batch 的 immutable prepared inputs，并在可行处避免重复 host synchronization。DataLoader worker 由所有 evaluation owner 在 finally 中关闭。LiDAR augmentation 使用从 worker/epoch/样本派生的 RNG，并对不支持等价 augmentation 的预计算 BEV 明确拒绝。

不在本变更中引入 cache daemon、外部数据库或复杂并行框架。

## Risks / Trade-offs

- [严格 evidence 校验使旧本地产物不能再汇总] → 这是 fail-closed 的预期；旧产物保持 local historical evidence，不自动迁移。
- [AMBER token mask 修复改变 baseline 数值] → 标记现有本地 checkpoint 为不可与修复后模型混合，后续正式比较使用重新训练的行。
- [冻结 inactive U-Mask head 仍保留 checkpoint 参数] → metadata 明确 active branch，若未来需要缩小 checkpoint 再单独提出迁移 change。
- [严格 CLI override 拒绝曾经可用的拼写错误/隐式 key] → 提供明确错误信息和已知路径提示。
- [streaming evaluation 改变内部 API] → 保留显式 capture 开关，并覆盖当前 validator/diagnostic consumers。

## Migration Plan

1. 先完成 H4 outer-test isolation，冻结未修复 design-screen 的选择和外部评估。
2. 添加 evidence/data/CLI regression，随后实现 shared validators 和 fixed-mask/summary fail-closed 行为。
3. 修复 metric、baseline token 和 inactive branch 语义；相关 checkpoint 不与新结果混合。
4. 收紧运行生命周期、I/O 和 deterministic settings，补 environment/CI/documentation。
5. 运行 focused tests、全量 pytest 和 OpenSpec validation；出现不兼容的 local artifact 时保留为 historical，不添加兼容层。

## Open Questions

- 无：本变更固定采用 strict current workflow；旧本地产物的可用性不改变 current contract。

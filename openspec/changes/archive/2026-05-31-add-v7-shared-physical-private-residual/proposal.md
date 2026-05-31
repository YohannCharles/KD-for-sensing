## Why

当前 HiST-Beam v3/v4/v6 变体仍以普通 hard beam 分类或 prototype 适配为主，shared/private 解耦没有显式约束 shared 分支学习跨场景可迁移的物理传播结构，private 分支也容易在少样本 target adaptation 中承担完整预测而产生过拟合或负迁移。

新增 `v7_shared_physical_private_residual`，用 beamspace power distribution 监督 shared 分支，并把 private 限定为 gated residual correction，可在不引入历史 label 的前提下验证更物理可迁移的跨场景少样本适配路径。

## What Changes

- 新增 HiST-Beam 变体 `v7_shared_physical_private_residual`，保持现有 `hist_beam_fusion` 注册入口和 v3/v4/v6 配置兼容。
- 新增训练期物理监督标签 `beamspace_power_label`，优先由 beam power/RSS vector 归一化生成；缺失时可由 path 参数和 AoD 近似构造，且必须显式记录不可用原因。
- 新增 shared physical head、shared beam head、private residual head 和 scalar residual gate，最终预测为 `logits_final = logits_shared + alpha * delta_logits_private`。
- 新增 v7 loss 组合：shared hard CE、final hard CE、beamspace KL、physical head KL、residual L2、gate L1 和 shared/private difference loss。
- 新增 v7 target adaptation 策略：冻结 shared backbone、shared head 和 physical head，只训练 target private adapter、private residual head、gate，以及配置允许的 norm affine 参数。
- 新增 v7 评估和 LOSO summary 字段，用于同时比较 shared-only 与 final residual 输出，并诊断 `alpha`、residual norm 和 physical KL。
- 不新增、不读取、不依赖历史 label；现有 history-anchor / residual-delta 路径保持独立，v7 默认禁止把 `input_beam` 或历史 beam 作为模型输入。

## Capabilities

### New Capabilities
- `beamspace-physical-labels`: 定义 `beamspace_power_label` 的生成、缓存、batch 字段、source/target 使用边界和诊断统计。

### Modified Capabilities
- `hist-beam-cross-scene-adaptation`: 增加 `v7_shared_physical_private_residual` 变体、v7 loss、v7 target adaptation 冻结策略、v7 指标与 summary 输出要求。

## Impact

- 影响数据层：`kd_sensing.data.datasets.mmw`、beam power/path 解析 helper、物理标签缓存与 dataset batch 字段。
- 影响模型层：`kd_sensing.models.fusion.hist_beam` 中的 variant 枚举、配置解析、forward 输出和 v7 heads/gate。
- 影响训练层：`kd_sensing.engine.hist_beam_losses`、`hist_beam_training`、`hist_beam_adaptation` 和 batch target 准备逻辑。
- 影响评估层：`kd_sensing.evaluation.hist_beam_outputs`、`engine.evaluation_pass`、LOSO summary/conclusion 字段。
- 影响配置与测试：新增 `configs/hist_beam/v7_shared_physical_private_residual.yaml`，补充模型构建、loss、adaptation 冻结、BSP 标签、防 target oracle 泄漏和指标写出测试。

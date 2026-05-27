## ADDED Requirements

### Requirement: HiST-Beam execute run 产物
HiST-Beam quick validation 的每个 source-only evaluation 和 adapted evaluation run MUST 输出可追踪产物。产物 MUST 至少包含 `metrics.json`、target_test predictions、配置快照、fold/split/sampling metadata 和当前 variant metadata。

#### Scenario: source-only evaluation 写出标准产物
- **WHEN** execute runner 完成 `v0_flat` 或 `v3_decoupled` 的 source-only target_test evaluation
- **THEN** run directory MUST 包含 `metrics.json`
- **AND** run directory MUST 包含 target_test predictions
- **AND** artifact metadata MUST 记录 target scene、source scenes、variant、budget、seed 和 source checkpoint path

#### Scenario: adapted evaluation 写出标准产物
- **WHEN** execute runner 完成 `v4_adapter`、`v5_adapter_proto` 或 `v6_full_finetune` 的 adapted target_test evaluation
- **THEN** run directory MUST 包含 `metrics.json`
- **AND** run directory MUST 包含 target_test predictions
- **AND** artifact metadata MUST 记录 adaptation checkpoint path、source checkpoint path 和 adaptation strategy

#### Scenario: predictions 包含对比所需字段
- **WHEN** 系统写出 HiST-Beam target_test predictions
- **THEN** predictions MUST 至少包含 sample id、scene、true beam、predicted beam、top-k predictions、coarse true/pred、fine true/pred 和 variant metadata
- **AND** predictions MUST 标明样本来自 `target_test`

### Requirement: Adaptation 效率指标
HiST-Beam adaptation run MUST 记录 trainable parameter count、total parameter count、trainable ratio、adaptation time 和 prototype coverage 或不可用原因。这些指标 MUST 写入 run-level `metrics.json` 或 run metadata，并 MUST 被 LOSO summary 汇总。

#### Scenario: adapter run 记录 trainable ratio
- **WHEN** 系统执行 `v4_adapter` 或 `v5_adapter_proto` adaptation
- **THEN** metrics 或 metadata MUST 记录 trainable parameter count、total parameter count 和 trainable ratio
- **AND** trainable ratio MUST 反映实际参与优化的参数集合

#### Scenario: full fine-tuning run 记录全量参数比例
- **WHEN** 系统执行 `v6_full_finetune` adaptation
- **THEN** metrics 或 metadata MUST 记录 trainable parameter count、total parameter count 和 trainable ratio
- **AND** trainable ratio MUST 表示全部或等价全量参数参与训练

#### Scenario: adaptation time 可横向比较
- **WHEN** adaptation stage 完成
- **THEN** metrics 或 metadata MUST 记录 total adaptation time
- **AND** 若可获得 epoch 信息，系统 MUST 记录 adapt time per epoch

#### Scenario: prototype coverage 不可用时说明原因
- **WHEN** variant 未启用 prototype alignment 或 prototype artifact 缺失
- **THEN** 系统 MUST 将 prototype coverage 标记为不可用
- **AND** 系统 MUST 记录不可用原因，不得用 `0` 伪造 coverage

#### Scenario: few-shot 标签从 DeepSense6G beam power 路径解析
- **WHEN** target adapt CSV 的 `future_beamN` 或 `beamN` 字段是 beam-power 文件路径而非整数标签
- **THEN** few-shot sampler MUST 读取该 power vector 并使用 `argmax` 作为 beam label
- **AND** sampler MUST 优先使用 `future_beam_labelN` 或 `beam_labelN` 显式标签列
- **AND** sampling manifest MUST 记录 labeled sample 的 beam、coarse group 和 label source

#### Scenario: quick validation adaptation 超参显式可见
- **WHEN** 用户使用完整 `quick_validation` 配置执行 target adaptation
- **THEN** 配置 MUST 显式声明 adaptation epochs、entropy weight 和 prototype weight
- **AND** 0-label adaptation MUST NOT 因缺少默认权重而静默跳过所有有效更新

### Requirement: Quick validation 对比结论
系统 MUST 基于 quick validation summary 输出机器可读的快速验证结论。结论 MUST 比较 adapter/prototype variants 相对 source-only 和 full fine-tuning baseline 的效果与效率，并 MUST 明确标记缺失或不可比的 run。

#### Scenario: adapter 与 source-only 对比
- **WHEN** 同一 target scene、budget 和 seed 下存在 `v3_decoupled` source-only metrics 以及 `v4_adapter` 或 `v5_adapter_proto` adapted metrics
- **THEN** 结论 MUST 比较 Top-1、Top-3、Top-5、coarse accuracy 和 fine accuracy
- **AND** 结论 MUST 标明 adapter variant 是否优于 source-only

#### Scenario: adapter prototype 与 full fine-tuning 对比
- **WHEN** 同一 target scene、budget 和 seed 下存在 `v5_adapter_proto` 和 `v6_full_finetune` metrics
- **THEN** 结论 MUST 比较 accuracy 指标、trainable ratio 和 adaptation time
- **AND** 结论 MUST 标明 adapter+prototype 是否在效果或效率上优于 full fine-tuning

#### Scenario: 缺失 run 时结论不可判定
- **WHEN** 生成结论所需的 source-only、adapter、prototype 或 full fine-tuning run 缺失
- **THEN** 结论 MUST 将对应比较标记为 `inconclusive`
- **AND** 结论 MUST 记录缺失的 variant、target scene、budget、seed 和原因

#### Scenario: 结论文件写入执行输出目录
- **WHEN** quick validation execute 完成或 partial failure 结束
- **THEN** 系统 MUST 在输出目录写出 `quick_validation_conclusion.json` 或等价机器可读文件
- **AND** 结论文件 MUST 引用产生依据的 LOSO summary 路径

#### Scenario: smoke 资源探针不伪装方法结论
- **WHEN** 用户运行轻量 `quick_smoke` 配置
- **THEN** 结论文件 MAY 标记关键 adapter/full-finetune 对比为 `inconclusive`
- **AND** 系统 MUST 通过 missing/inconclusive 原因说明该配置只覆盖了资源探针矩阵

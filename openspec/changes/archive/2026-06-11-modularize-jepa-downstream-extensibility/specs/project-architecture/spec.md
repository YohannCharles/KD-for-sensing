## ADDED Requirements

### Requirement: JEPA downstream 扩展实现边界
项目 MUST 将 JEPA Stage 1 预训练主模型、JEPA downstream pooler/adapter、模块化 conditioned encoder、optimizer 参数组和 runtime metadata 维护在职责清晰的窄模块中。新增 JEPA downstream pooler 或 adapter MUST 不要求修改 dataset、训练主循环、checkpoint schema 或旧兼容入口。

#### Scenario: 新增 JEPA pooler 不修改训练主循环
- **WHEN** 开发者新增一个 JEPA downstream pooler
- **THEN** 变更 MUST 限定在 JEPA downstream pooler/adapter 模块、注册代码、配置和测试
- **AND** 不需要修改 `engine.trainer` 主循环或 supervised beam loss/metric 流程

#### Scenario: 新增 JEPA adapter 不修改 dataset
- **WHEN** 开发者新增一个 JEPA downstream adapter
- **THEN** 变更 MUST 不要求修改 DeepSense6G dataset、GPS transform、image preprocessing 或 DataLoader 构建逻辑
- **AND** adapter MUST 通过模型配置和 registry 接入

#### Scenario: 不恢复退役入口
- **WHEN** JEPA downstream extensibility change 落地
- **THEN** 系统 MUST 不新增 KD/distillation、HiST/Hist、Top8 selector、GPS residual、camera residual 或 legacy fusion 兼容入口
- **AND** 新能力 MUST 通过当前 `src/kd_sensing` 包结构和 registry 边界接入

### Requirement: optimizer 参数组构建位于 optim 模块
训练引擎 MUST 将参数组解析、模块名 pattern 匹配、重复匹配检测、未匹配参数处理和参数组 summary 维护在 `kd_sensing.engine.optim` 或等价窄模块中。训练主循环 MUST 只消费构建好的 optimizer 和 summary。

#### Scenario: 修改 JEPA 参数组不触碰 trainer
- **WHEN** 开发者调整 JEPA context encoder、GPS encoder、pooler、core 或 head 的参数组匹配规则
- **THEN** 主要变更 MUST 限定在 optimizer 构建模块及其测试
- **AND** 不需要编辑 `engine.trainer` 的 epoch 或 batch 编排逻辑

#### Scenario: 参数组 summary 写入现有日志路径
- **WHEN** 训练使用多个 optimizer 参数组
- **THEN** 现有训练日志和 TensorBoard scalar 映射 MUST 能记录每组 learning rate 和参数数量
- **AND** 未声明参数组时 MUST 保持现有单 `main` 组日志字段

### Requirement: runtime metadata 收集位于 run metadata 模块
JEPA downstream 结构 metadata MUST 由 `engine.run_metadata`、artifact writer 或等价窄模块收集。模型和子模块 MAY 暴露只读 metadata 方法；训练主循环 MUST 不手写 JEPA downstream 专属字段。

#### Scenario: 模型声明 metadata 被聚合
- **WHEN** `model.primary` 或其子模块提供 JEPA downstream training strategy metadata
- **THEN** runtime metadata 收集模块 MUST 将其写入 `final_config.yaml` 或等价运行 metadata
- **AND** metadata MUST 包含 pooler、adapter、checkpoint、freeze 和参数组摘要中的正式字段

#### Scenario: config fallback 兼容历史配置
- **WHEN** metadata 在模型构建前需要从配置生成
- **THEN** run metadata 模块 MAY 使用配置解析作为 fallback
- **AND** fallback MUST 与模型声明 metadata 的核心字段保持一致

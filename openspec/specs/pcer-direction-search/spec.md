# pcer-direction-search Specification

## Purpose
TBD - created by archiving change screen-pcer-direction-candidates. Update Purpose after archive.
## Requirements
### Requirement: 八方向必须共享快速筛选身份
系统 MUST 为 B0-B7 使用相同 MMW inner train/validation、历史 development test、seed、16 epoch、batch、optimizer、scheduler、prototype/backbone 初始化、mask curriculum、固定评测 mask 和 validation-best checkpoint 规则。系统 MUST 不从不同历史最终 checkpoint 初始化候选。

#### Scenario: 生成八组配置
- **WHEN** launcher 准备方向筛选
- **THEN** 八组 config MUST 只在声明的 router/target/evidence 机制上不同
- **AND** 所有生成 config MUST 记录共同协议 checksum 和 claim-ineligible 身份

### Requirement: 运行前必须执行真实 batch 审计
每个方向 MUST 在训练启动前通过 config parse、forward/backward、finite loss、availability 归一化和非零目标梯度检查，并记录 common/new loss 量级。新增加权 loss 只可依据训练 batch 自动调整一次，不得读取 validation 或 test。

#### Scenario: 新增 loss 量级越界
- **WHEN** 新增加权 loss 高于 beam loss 或低于其 1%
- **THEN** preflight MUST 按统一规则最多调整一次对应 lambda
- **AND** MUST 保存原值、调整值、训练 batch 指标和调整原因

### Requirement: GPU0-7 任务必须独立完成
系统 MUST 按 B0->GPU0 至 B7->GPU7 的固定逻辑映射启动任务、保存 PID/日志/config/checkpoint，并允许单任务失败而其他任务继续。系统 MUST 不终止无关 GPU 进程。

#### Scenario: 单方向失败
- **WHEN** 一个训练任务因明确代码错误退出
- **THEN** launcher MUST 标记该方向失败并等待其他任务
- **AND** 修复后 MUST 只重跑失败方向，不得静默跳过

### Requirement: 最佳 checkpoint 必须固定评测并诊断机制
系统 MUST 用各方向 validation-best checkpoint 评测 S0-S5、S3 四个缺失模态、统一汇总指标和声明的 mechanism diagnostics。B0/B1/B5/B6 MUST 完成 dynamic/global/mask-mean replacement；B2/B3/B4 MUST 输出 target 分布、对齐和梯度冲突；B7 MUST 输出单模态 evidence 指标。

#### Scenario: 生成统一比较
- **WHEN** 八组训练与评测完成
- **THEN** combined table MUST 同时包含只读历史 A0-A3 与 B0-B7
- **AND** Pareto 排名 MUST 按预注册 Full、Masked、Hard、S3、机制与成本规则分类 Winner/Promising/Reject

### Requirement: 筛选必须在方向判断后停止
系统 MUST 在单 seed development 结果上回答八个研究问题并给出唯一下一步方向，但 MUST 不自动启动 multi-seed 或下一轮完整实验。

#### Scenario: 没有复杂方向胜出
- **WHEN** 没有候选明确超过历史 A1 且满足 S3/Full gate
- **THEN** 报告 MUST 不勉强选择复杂方法
- **AND** MUST 在简单 B0/B1、B7 evidence 学习或放弃 router 创新之间依据机制证据选择


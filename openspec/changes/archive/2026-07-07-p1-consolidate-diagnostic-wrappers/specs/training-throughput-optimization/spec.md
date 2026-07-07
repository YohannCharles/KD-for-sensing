## ADDED Requirements

### Requirement: Training throughput profiling 与推荐必须共享 owner
Training IO profiling、瓶颈汇总和 parallel training recommendation MUST 由一个 throughput owner 或 package CLI mode 管理。项目 SHOULD 不保留独立 recommendation script 重复解析配置、硬件或 profiling fields。

#### Scenario: recommendation 读取 profiling owner 输出
- **WHEN** 协作者需要 parallel training recommendation
- **THEN** 推荐逻辑 SHOULD 读取 profiling owner 的 output summary 或由同一 owner mode 计算
- **AND** MUST 不复制 profiling input discovery、config parsing 或 metrics formatting 逻辑

#### Scenario: profile script 合并后字段保持
- **WHEN** `scripts/profile_training_io.py` 行为迁移到 package CLI 或 owner module
- **THEN** sampling fields、throughput metrics、IO bottleneck labels 和 recommendation inputs MUST 保持稳定或同步更新 current spec
- **AND** focused tests MUST 覆盖 profiling output 和 recommendation mode

### Requirement: Throughput wrapper 保留必须说明独立价值
若 throughput profiling 或 recommendation wrapper 因外部复现实验仍需保留，项目 MUST 在 inventory 或 current spec 中记录 retained-with-reason。

#### Scenario: 保留脚本有删除触发条件
- **WHEN** throughput wrapper 暂时保留
- **THEN** retained-with-reason MUST 包含独立契约、替代 owner 缺口和未来删除触发条件
- **AND** docs MUST 不把 wrapper 描述为优先于 owner CLI 的推荐入口

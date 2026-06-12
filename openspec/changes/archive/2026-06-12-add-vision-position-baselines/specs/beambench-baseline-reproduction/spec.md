## ADDED Requirements

### Requirement: Camera AE + GPS Direct baseline suite preset
系统 MUST 将 Arnold22/BeamBench 风格的 Camera AE + GPS Direct 本地实现作为 Vision-Position baseline suite 的 `camera_ae_gps` preset 暴露。该 preset MUST 保持 Camera AE encoder、GPS direct feature encoder、late fusion classifier 和 BeamBench 指标口径的可审计性，不得被 residual、gated、attention 或 candidate reranker 模型替代。

#### Scenario: preset 与论文目标行结构一致
- **WHEN** 用户构建 `camera_ae_gps` baseline preset
- **THEN** 模型 MUST 使用 Camera AE encoder 输出 image latent
- **AND** 模型 MUST 使用 GPS direct 或 MLP feature encoder
- **AND** 模型 MUST 将 image latent 与 GPS feature 融合后输出 64-beam classifier logits

#### Scenario: AE checkpoint 缺失时清晰失败
- **WHEN** `camera_ae_gps` preset 配置要求 frozen AE checkpoint 但路径为空或文件不存在
- **THEN** 系统 MUST 拒绝构建或运行该 preset
- **AND** 错误信息 MUST 说明需要提供或训练 Camera AE checkpoint

#### Scenario: suite metadata 不冒充官方结果
- **WHEN** `camera_ae_gps` preset 完成训练、评估或 dry-run
- **THEN** report 或 run metadata MUST 记录是否使用官方 pretrained 权重、官方测试集和官方完整训练搜索流程
- **AND** 若任一条件不满足，结果 MUST NOT 声称等同 Arnold22/BeamBench Table III 官方数值

#### Scenario: metrics 字段与 BeamBench 报告一致
- **WHEN** `camera_ae_gps` preset 输出 baseline metrics
- **THEN** 输出 MUST 包含 top-1 和 top-3 accuracy
- **AND** 若输出 DBA 或 top-3 DBA，字段名和报告 MUST 声明使用官方非环形 DBA、64-beam circular DBA 或其它 metric profile

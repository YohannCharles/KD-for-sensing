# geometry-residual-beam-labels Specification

## Purpose
定义 geometry coarse beam、circular residual beam label 和 clipped residual class 的数据契约，用于把绝对 beam 预测拆解为可解释的几何先验与残差学习目标，并保证 label space 可逆、可诊断且默认不影响 absolute label 路径。
## Requirements
### Requirement: geometry residual label 路线已退役
专门用于把绝对 beam 拆成 geometry coarse beam 与 residual/delta class 的 label-space 路线不再属于当前支持能力。系统 MUST 不再要求 dataset 暴露 `beam_geo`、`beam_residual`、`residual_class`、GPS local delta class 或 geometry residual target provider。

#### Scenario: 默认数据样本不含 geometry residual 契约
- **WHEN** 用户运行当前保留训练或评估配置
- **THEN** dataset sample MUST 不要求 geometry residual 字段
- **AND** 项目 MUST 不保留专门服务 residual/delta 路线的 target provider 文档或测试


# deepsense6g-camera-ae-residual-correction Specification

## Purpose
定义 DeepSense6G camera AE residual correction 工作流的输入、manifest、AE 训练/特征导出、GPS anchored residual 模型、loss、训练协议和诊断输出边界，确保 camera 只作为 GPS v2 prior 之上的 residual 或 rerank 信号使用，并守住 target query 防泄漏约束。
## Requirements
### Requirement: DeepSense6G camera residual 已退役
DeepSense6G camera-assisted residual correction 不再属于当前支持能力。系统 MUST 不再提供 camera residual manifest、camera residual model/loss/training、candidate reranker、plotter、comparison CLI 或默认 residual 配置。

#### Scenario: camera residual 入口不存在
- **WHEN** 开发者检查安装入口和配置
- **THEN** 项目 MUST 不声明 `kd-sensing-run-deepsense6g-camera-residual`、plot、compare 或 manifest 入口
- **AND** 项目 MUST 不保留 `configs/deepsense6g_camera_residual.yaml` 作为当前配置

### Requirement: Camera residual manifest data loading 已退役
Camera residual manifest Dataset/DataLoader 路线已退役，不属于当前数据加载支持面。项目 MUST 不再要求从 camera residual manifest 构建训练或评估 DataLoader；旧 manifest 字段只能作为历史说明、migration guard 或本地产物审计背景出现。

#### Scenario: camera residual loader 不作为当前入口
- **WHEN** 用户引用 camera residual 配置、manifest loader 或旧 residual ablation
- **THEN** 系统 MUST 不把该路径作为当前训练或评估入口
- **AND** 错误信息、文档或 guardrail MUST 明确 camera residual 已退役

#### Scenario: 历史 camera residual 产物只读
- **WHEN** cleanup、run index 或历史报告扫描到 camera residual manifest、AE feature 或 summary
- **THEN** 这些文件 MAY 被只读展示或列为清理候选
- **AND** 系统 MUST 不因此恢复 camera residual Dataset/DataLoader

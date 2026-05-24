## ADDED Requirements

### Requirement: Multimodal-NF 运行产物一致性
Multimodal-NF 训练和评估运行产物 MUST 在 `final_config.yaml`、`resolved_config.yaml`、`startup_summary.json`、`metrics.json` 和 runtime metadata 中保持 dataset type、objective、modalities、num classes、codebook metadata 和 enabled heads 的一致性。

#### Scenario: final config 与 startup summary 一致
- **WHEN** Multimodal-NF 训练启动并写出 `final_config.yaml` 与 `startup_summary.json`
- **THEN** 两个产物 MUST 记录相同的 dataset type、objective、enabled modalities 和 num beam classes
- **AND** 若 startup summary 包含模型 heads，head 输出类别数 MUST 与 codebook metadata 一致

#### Scenario: metrics objective 可追溯
- **WHEN** Multimodal-NF 训练或评估写出 `metrics.json`
- **THEN** metrics MUST 包含当前 objective 的名称、primary metric、metric mode 和 available metrics
- **AND** metrics 中的 objective metadata MUST 与 `final_config.yaml` runtime metadata 保持一致

#### Scenario: 配置矛盾时拒绝启动
- **WHEN** Multimodal-NF 配置中的 objective、target schema、model heads 或 codebook metadata 互相矛盾
- **THEN** 系统 MUST 在训练前或启动早期拒绝运行
- **AND** 错误信息 MUST 指向矛盾字段和可修正的配置路径

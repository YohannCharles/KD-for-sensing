## ADDED Requirements

### Requirement: Multimodal-NF helper 拆分兼容
Multimodal-NF preprocessing 和 dataset helper 拆分后，审计、index 构建、split assignment、codebook metadata、HDF5 inspection 和 dataset sample 契约 MUST 保持兼容。公开 preprocessor registry 名称和配置入口 MUST 不改变。

#### Scenario: 审计入口保持
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/multimodal_nf_audit.yaml`
- **THEN** 命令 MUST 继续调用 Multimodal-NF 审计流程
- **AND** 输出字段和缺失数据错误语义 MUST 保持兼容

#### Scenario: index 构建入口保持
- **WHEN** 用户运行 Multimodal-NF index 构建配置
- **THEN** 系统 MUST 继续输出相同语义的 sample index 和 metadata
- **AND** split assignment、sample_id、codebook metadata 和 target 引用 MUST 保持兼容

#### Scenario: dataset sample 保持
- **WHEN** 测试从 Multimodal-NF dataset 取样
- **THEN** flat sample keys、metadata、enabled modality lazy loading 和 target fields MUST 与拆分前兼容
- **AND** focused tests MUST 不依赖真实全量数据

## MODIFIED Requirements

### Requirement: DeepSense6G TopK candidate optional modality loading 支撑语义
DeepSense6G TopK candidate optional modality loading 不再作为当前支撑语义维护。BGAM 与 standalone Top8 selector 退役后，系统 MUST 不保留仅服务这些路线的 candidate dataset helper、optional modality loading、manifest availability 解析或 GPS context-only candidate support。

#### Scenario: TopK candidate helper 不作为 current dataset 路径
- **WHEN** 开发者检查当前 data 模块和 config load tests
- **THEN** 项目 MUST 不要求 DeepSense6G TopK candidate dataset helper 可导入或可构建
- **AND** 保留 workflow MUST 不通过 BGAM/TopK candidate helper 读取 image、camera AE、LiDAR 或 radar optional modality

### Requirement: TopK candidate normalization fit boundary
TopK candidate normalization fit boundary 不再作为当前支撑能力维护。若未来 current workflow 需要 candidate feature normalization，MUST 在对应 OpenSpec capability 中重新定义 fit split、query exclusion 和 metadata 规则。

#### Scenario: TopK candidate normalizer 不作为保留验证对象
- **WHEN** 开发者运行配置加载、数据集或架构边界测试
- **THEN** 测试 MUST 不要求 TopK candidate normalizer artifact、BGAM candidate scaler 或 selector scaler 可用
- **AND** 已退役的 candidate normalizer 不得作为兼容 facade 保留

## REMOVED Requirements

### Requirement: GPS+LiDAR BGAM 按需模态加载
**Reason**: BGAM dataset 和 ablation 已退役。
**Migration**: 无兼容迁移；保留 data loaders 继续由各自 current specs 约束。

### Requirement: GPS+LiDAR BGAM 防泄漏数据边界
**Reason**: 该数据边界只服务 BGAM dataset/runner。
**Migration**: 保留 workflow 继续使用各自 anti-leakage 和 split metadata 约束。

### Requirement: GPS+LiDAR BGAM manifest column mapping
**Reason**: BGAM manifest loader 已退役。
**Migration**: 无兼容迁移；不得保留 BGAM column mapping 兼容层。

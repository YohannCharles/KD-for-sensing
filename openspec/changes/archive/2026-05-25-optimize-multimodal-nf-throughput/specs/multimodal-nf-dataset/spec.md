## ADDED Requirements

### Requirement: Multimodal-NF image/LiDAR 派生缓存
Multimodal-NF dataset MUST 支持对 image 和 LiDAR 模态使用显式配置的派生缓存或重打包数据源，以减少训练时对原始 gzip HDF5 大数组的重复解压。派生缓存 MUST 是本地产物，默认写入 `dataset/MultimodalNF/cache`、`outputs/` 或用户显式配置的 ignored 目录；系统 MUST 不自动改写、删除或移动原始 Multimodal-NF HDF5 文件。

#### Scenario: 派生缓存关闭时保持原始读取
- **WHEN** 用户未启用 Multimodal-NF image/LiDAR 派生缓存或设置策略为 `off`
- **THEN** dataset MUST 继续从原始 HDF5 文件读取启用模态
- **AND** 返回的 sample keys、tensor shape、dtype 语义和 target 字段 MUST 与现有契约兼容

#### Scenario: 派生缓存命中
- **WHEN** 用户为 image 或 LiDAR 设置派生缓存策略且缓存 metadata 与原始文件 fingerprint、profile、split、`seq_len` 和 `num_pred` 匹配
- **THEN** dataset MUST 从派生缓存读取对应启用模态
- **AND** 返回 tensor MUST 与原始 HDF5 读取路径在 shape、dtype 语义和样本顺序上等价
- **AND** runtime metadata MUST 记录该模态实际使用派生缓存、缓存路径、策略和匹配的原始 fingerprint

#### Scenario: read_only 缓存缺失
- **WHEN** 用户设置派生缓存策略为 `read_only` 且所需 image 或 LiDAR 缓存不存在或 metadata 不匹配
- **THEN** dataset 构建 MUST 失败
- **AND** 错误信息 MUST 包含模态名、策略、期望缓存路径和缺失或不匹配原因

#### Scenario: auto 或 rebuild 生成缓存
- **WHEN** 用户设置派生缓存策略为 `auto` 或 `rebuild`
- **THEN** 系统 MUST 能通过显式预处理流程或受控 dataset 初始化流程生成所需派生缓存
- **AND** 生成结果 MUST 写出 metadata sidecar，记录原始文件路径或 fingerprint、profile、shape、dtype、样本数、生成时间、`seq_len` 和 `num_pred`
- **AND** 生成过程 MUST 使用原子写入或等价机制，避免多进程读取半成品缓存

#### Scenario: 未启用模态不访问缓存
- **WHEN** 用户运行 GPS-only、CSI-only 或不包含 image/LiDAR 的 fusion 配置
- **THEN** dataset MUST 不解析、不创建、不读取 image 或 LiDAR 派生缓存路径
- **AND** 对应原始 image/LiDAR 文件缺失不得阻止当前任务运行

### Requirement: Multimodal-NF 派生缓存审计与可追踪性
Multimodal-NF 审计、runtime metadata 和 profile 输出 MUST 能说明 image/LiDAR 数据来自原始 HDF5 还是派生缓存。系统 MUST 将缓存策略、缓存覆盖率和缓存有效性记录为机器可读字段。

#### Scenario: 审计报告包含缓存状态
- **WHEN** 用户运行 Multimodal-NF 审计或缓存预热流程
- **THEN** 输出 MUST 记录每个启用缓存模态的缓存路径、存在状态、样本数、profile、shape、dtype 和原始 fingerprint 匹配结果
- **AND** 输出 MUST 不要求真实大缓存文件进入源码变更

#### Scenario: runtime metadata 记录实际数据源
- **WHEN** 训练或评估构建 Multimodal-NF dataset
- **THEN** runtime metadata MUST 按 split 和模态记录 `source_kind`，其值区分原始 HDF5 和派生缓存
- **AND** metadata MUST 记录派生缓存策略、缓存路径和是否发生生成或回退

#### Scenario: 缓存失效清晰失败或回退
- **WHEN** 派生缓存 metadata 与当前原始文件、profile 或窗口参数不一致
- **THEN** `read_only` 策略 MUST 清晰失败
- **AND** `auto` 策略 MUST 清晰记录回退到原始 HDF5 或重新生成缓存的行为

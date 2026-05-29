## ADDED Requirements

### Requirement: MMW dataset 初始化内存有界
MMW dataset 初始化 MUST 避免为了 normalizer、CSV 派生列或 metadata 准备而无界持有所有样本的大数组。GPS、mmWave 和 CSI 等模态的 normalizer 拟合 MUST 使用 streaming 或可释放的临时统计，并 MUST 在拟合完成后避免把 per-sample sequence cache 常驻到 DataLoader worker。

#### Scenario: GPS/mmWave scaler 拟合不保留全量样本缓存
- **WHEN** MMW train dataset 启用 GPS 或 mmWave normalization
- **THEN** scaler 拟合 MUST 能通过 streaming 或临时数组完成
- **AND** 拟合完成后 dataset MUST 不保留所有样本的 GPS/mmWave sequence 大数组缓存
- **AND** runtime metadata MUST 记录 scaler 来源、样本数和是否使用 streaming 拟合

#### Scenario: DataLoader worker 不复制初始化大缓存
- **WHEN** MMW dataset 使用多 worker DataLoader
- **THEN** worker 进程 MUST 不因 dataset 初始化阶段的 per-sample feature cache 而复制全量样本大数组
- **AND** profile 或 metadata MUST 能报告 worker 内存风险相关配置

### Requirement: MMW image 序列按需加载与缓存等价
MMW image modality MUST 按 enabled modalities 和 seq_len 读取 RGB/ImageNet image 序列。启用 image-derived cache 时，dataset MUST 保持与原始 image 读取路径一致的样本字段、shape、dtype 和 label 语义。

#### Scenario: image-derived cache 保持 batch 契约
- **WHEN** MMW fusion 配置启用 image modality、`seq_len=8` 和 image-derived cache
- **THEN** 单样本 `image` tensor MUST 保持 `[seq_len, 3, H, W]`
- **AND** batch 后 image 输入 MUST 与未启用 cache 时的 shape 和 dtype 一致
- **AND** `input_beam`、`target_beam`、GPS 和 mmWave 字段 MUST 不因 image cache 改变

#### Scenario: 未启用 image 不读取 image 路径
- **WHEN** MMW fusion 配置的 modalities 为 `["gps", "mmwave"]`
- **THEN** dataset MUST 不读取 camera 列对应文件
- **AND** dataset MUST 不初始化 image transform 或 image-derived cache

### Requirement: LOSO stage dataset 构建边界
LOSO 数据构建流程 MUST 支持按 stage 构建当前阶段所需的数据集和 DataLoader，避免 source training 阶段提前构建 target adapt/test dataset。

#### Scenario: source_train 只构建 source loader
- **WHEN** LOSO executor 进入 `source_train` stage
- **THEN** 系统 MUST 只构建 source train dataset 和 loader
- **AND** 系统 MUST 不构建 target adapt 或 target test dataset

#### Scenario: target stage 延迟构建 target loader
- **WHEN** LOSO executor 进入 target adaptation 或 target test evaluation stage
- **THEN** 系统 MUST 在该 stage 内构建所需 target dataset 和 loader
- **AND** source stage 的 DataLoader worker MUST 已关闭或不再持有

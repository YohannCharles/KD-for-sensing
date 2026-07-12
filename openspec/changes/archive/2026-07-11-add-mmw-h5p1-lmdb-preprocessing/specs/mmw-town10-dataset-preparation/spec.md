## ADDED Requirements

### Requirement: 已下载 MMW 多天气统一 H5/P1 准备
系统 MUST 支持对 `sunny`、`rainy` 和 `foggy` 条件下已下载的 MMW Town03 场景生成输入窗口 5、预测窗口 1 的准备产物。每个场景的 split metadata MUST 记录 `seq_len=5` 和 `pred_len=1`，且窗口不得跨 vehicle 或不连续 frame。

#### Scenario: rainy 和 foggy 场景生成 H5/P1 split
- **WHEN** 用户对 rainy 或 foggy 的已下载 Town03 场景运行准备流程
- **THEN** 对应 `Prepared/<scenario>/splits` MUST 生成 H5/P1 的 `all_sequences.csv`、`train.csv` 和 `test.csv`
- **AND** metadata MUST 记录 condition、scenario、输入窗口 5 和预测窗口 1

#### Scenario: sunny 旧窗口重建为 H5/P1
- **WHEN** sunny 场景已有其它窗口长度的准备产物且用户显式运行 H5/P1 重建
- **THEN** sequence split MUST 按已有 frame manifest 重建为 H5/P1
- **AND** 新 metadata MUST 不再声明旧的 8/3 窗口

### Requirement: MMW split-level LMDB 样本缓存
预处理系统 MUST 能使用当前 dataset registry 为 MMW dataset 生成 split-level LMDB 样本缓存，并 MUST 复用现有 LMDB key 和读取契约。缓存 metadata MUST 记录 dataset type、condition、scenario、split、样本数、`seq_len` 和 `num_pred`。

#### Scenario: MMW H5/P1 LMDB 生成
- **WHEN** 配置使用 `dataset.type: mmw`、`seq_len: 5` 和 `num_pred: 1` 生成 train/test LMDB
- **THEN** 每个 split MUST 写入与对应 dataset 长度一致的样本数
- **AND** metadata MUST 记录 `seq_len=5`、`num_pred=1` 和 MMW condition/scenario

#### Scenario: DeepSense6G 旧 LMDB 入口保持兼容
- **WHEN** 用户继续使用 `deepsense6g_sample_lmdb_cache` 预处理类型
- **THEN** 系统 MUST 继续生成可由现有 sample cache reader 读取的 LMDB
- **AND** 旧配置 MUST 不要求改名后才能运行

### Requirement: MMW 可再生成缓存产物边界
MMW 图像、LiDAR 和 sample LMDB 等可再生成缓存 MUST 默认写入 `outputs/cache/MMW/<condition>/`，并 MUST 按 condition、场景、cache kind 和窗口版本避免路径冲突。

#### Scenario: 三种天气缓存隔离
- **WHEN** sunny、rainy 和 foggy 的 H5/P1 缓存同时存在
- **THEN** 每种 condition MUST 使用独立的 `outputs/cache/MMW/<condition>/` 子树
- **AND** 任一 condition 的缓存生成 MUST 不覆盖其它 condition 或其它窗口版本的产物

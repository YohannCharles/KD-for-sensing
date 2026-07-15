## MODIFIED Requirements

### Requirement: DataLoader 运行参数可配置
训练和评估入口 MUST 支持通过配置控制 DataLoader 运行参数，包括 batch size、`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor` 和 `drop_last`。当 `num_workers=0` 时，系统 MUST 不传入仅适用于多 worker 的参数。train、validation 和 test MUST 使用按 split 域分离的稳定 generator；其状态及有状态 sampler state MUST 可进入 current checkpoint runtime state。Persistent worker MUST 跨 epoch 保持存活，并只在整个 run finalization、loader 替换或异常退出时关闭。

#### Scenario: 多 worker DataLoader
- **WHEN** 配置设置 `data.dataloader.num_workers: 4`、`persistent_workers: true`、`pin_memory: true` 和 `prefetch_factor: 2`
- **THEN** 训练和评估 DataLoader MUST 使用这些参数
- **AND** train loader MUST 按配置决定是否 `drop_last`

#### Scenario: 单进程 DataLoader
- **WHEN** 配置设置 `data.dataloader.num_workers: 0`
- **THEN** 系统 MUST 不向 DataLoader 传入 `persistent_workers` 或 `prefetch_factor`
- **AND** DataLoader MUST 能在 CPU-only smoke test 中正常迭代

#### Scenario: 评估复用 loader 参数解析
- **WHEN** 用户运行评估入口并配置 DataLoader 参数
- **THEN** 评估入口 MUST 使用与训练入口一致的参数解析逻辑
- **AND** 评估入口 MUST 保持 `shuffle: false`

#### Scenario: 每个 split 使用独立稳定 generator
- **WHEN** 系统使用同一 experiment seed 构建 train、validation 和 test DataLoader
- **THEN** 每个 split MUST 从 seed、split 名称和 dataset fingerprint 派生独立 generator
- **AND** 构建或迭代其它 split MUST 不改变当前 split 的样本顺序
- **AND** generator identity MUST 可记录到 runtime metadata

#### Scenario: 恢复 generator 与 sampler state
- **WHEN** current checkpoint 恢复 DataLoader generator 或有状态 sampler
- **THEN** 系统 MUST 在创建下一 iterator 前加载对应 split state
- **AND** 恢复后的下一 batch 顺序 MUST 与相同环境下连续训练一致

#### Scenario: Validation persistent worker 跨 epoch 保持
- **WHEN** validation loader 配置 `persistent_workers: true` 并在多个 epoch 运行 validation
- **THEN** training loop MUST 不在每次 validation 返回后调用私有 worker shutdown
- **AND** 同一 loader 的 persistent worker MUST 保持可复用

#### Scenario: Run finalization 关闭 worker
- **WHEN** training/evaluation 正常完成或异常退出
- **THEN** finalization MUST 尝试关闭 train、validation 和 test 的可关闭 worker/iterator
- **AND** cleanup 异常 MUST 不覆盖原始训练异常

## ADDED Requirements

### Requirement: Synthetic sample 必须按稳定 index 确定
Synthetic dataset MUST 用 base seed、split、dataset schema identity 和 sample index 派生局部随机状态。相同 identity 的 `dataset[index]` MUST 返回相同样本，且不得依赖该 dataset 或其它 split 此前的访问顺序。派生 MUST 使用跨进程稳定算法，不得使用进程随机化的 Python `hash()`。

#### Scenario: 重复读取相同 index
- **WHEN** 同一个 synthetic dataset 实例多次读取相同 index
- **THEN** 每个模态、target、history index 和 auxiliary label tensor MUST 相同
- **AND** 中间读取其它 index MUST 不改变结果

#### Scenario: 不同访问顺序
- **WHEN** 两个相同配置/seed/split 的 synthetic dataset 以不同 index 顺序访问
- **THEN** 对每个相同 index 得到的样本 MUST 相同
- **AND** DataLoader shuffle 顺序只能改变访问顺序，不能改变 index 内容

#### Scenario: Worker 数不改变 index 内容
- **WHEN** 相同 synthetic split 分别用 `num_workers: 0` 和多 worker DataLoader 读取
- **THEN** 按 sample index 对齐后的样本内容 MUST 相同
- **AND** worker 调度 MUST 不共享或竞争 dataset 内可变 generator

#### Scenario: Seed 或 split 域分离
- **WHEN** base seed 或 split 名称发生变化
- **THEN** derived sample identity MUST 随之变化
- **AND** train、validation 和 test MUST 不因 index 相同而共享同一随机流

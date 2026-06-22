## ADDED Requirements

### Requirement: Target-shot split 可直接消费 Mapping rows
target-shot split runtime MUST 支持直接消费 `Mapping[str, Any]` rows，并从 mapping 中解析 `sample_id`、split、dataset type、domain fields、resource refs、target refs 和 metadata。独立 `SampleRow` 文件不是 dataset runtime contract 的必要组成部分。

#### Scenario: Mapping row 构建 domain key
- **WHEN** `build_domain_key` 或 `build_target_shot_split` 接收普通 mapping row
- **THEN** 系统 MUST 解析 scenario、weather、town、sample id 和 metadata
- **AND** 行为 MUST 与旧 `SampleRow.to_dict()` 结果兼容

#### Scenario: 删除独立 SampleRow 文件
- **WHEN** 项目内没有代码构造或公开推荐 `kd_sensing.data.dataset_runtime.SampleRow`
- **THEN** 本 change MAY 删除 `dataset_runtime.py`
- **AND** target-shot split tests MUST 继续覆盖 mapping rows、JSON metadata strings 和 artifact round trip

### Requirement: 轻量 row 类型如保留必须归属实际 owner
如果实现仍需要 row dataclass，类型 MUST 放在实际消费 owner 中，例如 `target_shot_splits.py` 或明确 metadata helper。项目 MUST 不为一个单调用轻量类型保留独立 runtime framework 文件。

#### Scenario: 局部 row 类型
- **WHEN** target-shot split 为可读性保留 row dataclass
- **THEN** dataclass MUST 位于 target-shot split owner 模块
- **AND** dataset runtime spec MUST 不要求通过该类型才能满足 flat sample 或 metadata contract

### Requirement: Dataset descriptor 可保持简单数据表
Dataset descriptor 机制 MUST 继续轻量可导入，但不要求使用多层 dataclass 或可扩展 framework。若当前 descriptor 只覆盖少量保留 dataset type，项目 MAY 使用简单 dict/table 和验证 helper 实现。

#### Scenario: descriptor 行为保持
- **WHEN** data factory 或 config validation 查询 DeepSense6G/MMW descriptor
- **THEN** 系统 MUST 返回 dataset family、storage kind、supported profiles 和 artifact boundary
- **AND** 查询 MUST 不导入 pandas、torch dataset、模型或训练模块

#### Scenario: descriptor 实现可简化
- **WHEN** dataclass descriptor 层只包装静态表且没有额外行为价值
- **THEN** 本 change MAY 将实现简化为 dict/table helper
- **AND** 公开查询函数和 config validation 行为 MUST 保持兼容

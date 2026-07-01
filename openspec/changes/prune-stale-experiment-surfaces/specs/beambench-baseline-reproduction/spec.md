## MODIFIED Requirements

### Requirement: BeamBench 数据接口检查
系统 MUST 提供 `kd_sensing.baselines.beambench.dataset_check` owner module 或等价包内入口，用于检查 BeamBench/DeepSense6G 数据目录和 CSV 字段。该检查 MUST 不修改、不移动、不删除真实数据文件。系统 MUST 不要求已删除的 `kd_sensing.cli.beambench_check_dataset` wrapper 存在。

#### Scenario: CSV 存在性与字段检查
- **WHEN** 用户运行 `conda run -n kd_mm_beam python -m kd_sensing.baselines.beambench.dataset_check --data-root <path> --csv <file>` 或等价保留入口
- **THEN** 系统 MUST 验证 CSV 文件存在且可读取
- **AND** 系统 MUST 报告 camera、LiDAR、radar、GPS、label、scene id、sample id 和 sequence id 相关字段是否存在或可由配置解析

#### Scenario: 传感器文件引用检查
- **WHEN** CSV 行包含 camera、LiDAR、radar 或 GPS 文件路径列
- **THEN** 系统 MUST 按数据根目录解析每一行引用的文件路径
- **AND** 系统 MUST 统计每个模态的存在数量、缺失数量、缺失比例和示例缺失路径

#### Scenario: label 和 beam index 检查
- **WHEN** CSV 或 beam label 文件提供 optimal beam index、future beam label 或等价目标
- **THEN** 系统 MUST 验证 label 存在且位于配置声明的合法范围内
- **AND** 系统 MUST 报告 label 总数、非法 label 数量、label 最小值、最大值和是否使用 0-based 或 1-based beam shift

#### Scenario: scene sample sequence 标识解析
- **WHEN** CSV 包含 scene、sample、timestamp、frame 或 sequence 字段
- **THEN** 系统 MUST 解析并报告 scene ID、sample ID、sequence ID 或 timestamp 的可用性
- **AND** 系统 MUST 在不能解析时报告缺失字段名和可选 fallback，而不是静默通过

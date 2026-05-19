## ADDED Requirements

### Requirement: MMW channel-derived mmWave power vectors
mmWave 预处理 MUST 支持从 MMW channel `_paths.npy` 或 `_paths.npz` 文件生成 64 维 receive-power vector。生成的 power vector MUST 与现有 `read_mmwave_power_vector()` 和 `build_mmwave_db_features()` 契约兼容，并 MUST 可作为 beam label 文件由 `argmax` 读取。

#### Scenario: 读取 MMW 派生 power txt
- **WHEN** MMW 准备流程写出包含 64 个 finite 数值的 power txt 文件
- **THEN** 现有 mmWave power vector 读取逻辑 MUST 返回形状为 `[64]` 的 `float32` 数组
- **AND** 现有 beam label 读取逻辑 MUST 能通过 `argmax` 得到 0-based beam 标签

#### Scenario: MMW channel 派生结果维度非法
- **WHEN** channel-to-beam 派生逻辑产生的 power vector 不是 64 个数值
- **THEN** 系统 MUST 拒绝写入该 beam power 文件
- **AND** 错误信息或 sanity report MUST 包含实际维度、期望维度和 channel 文件路径

### Requirement: Channel-to-beam metadata is versioned
系统 MUST 为 MMW channel-derived mmWave power vector 记录可复现 metadata，包括算法版本、codebook 类型、`num_beams`、发射端天线数量、接收端天线数量、使用的 channel 字段、输入 channel 文件路径和输出 power 文件路径。

#### Scenario: metadata 记录 codebook 设置
- **WHEN** MMW 准备流程生成任意 channel-derived power vector
- **THEN** metadata MUST 记录该 power vector 使用的 `num_beams`
- **AND** metadata MUST 记录 codebook 类型和算法版本
- **AND** metadata MUST 记录输入 channel 文件与输出 power 文件的相对路径映射

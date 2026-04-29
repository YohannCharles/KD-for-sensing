## ADDED Requirements

### Requirement: LiDAR BEV cache 原子写入
LiDAR BEV cache 写入 MUST 避免其它并行训练或评估进程读取到半成品文件。系统 MUST 使用临时文件加原子替换、文件锁或等价机制写入 `.npy` cache。

#### Scenario: 并发写入同一 LiDAR cache
- **WHEN** 两个训练进程在 `auto` policy 下同时遇到同一个缺失 LiDAR BEV cache
- **THEN** 任一进程写入时 MUST 不暴露半写入目标文件
- **AND** 最终存在的 cache 文件 MUST 可被 `np.load` 正常读取

#### Scenario: 读取已完成 LiDAR cache
- **WHEN** LiDAR BEV cache 文件已经完成写入
- **THEN** 后续 dataset 取样 MUST 直接读取该 cache
- **AND** 读取结果 MUST 保持与在线构造路径相同的 shape 和 dtype

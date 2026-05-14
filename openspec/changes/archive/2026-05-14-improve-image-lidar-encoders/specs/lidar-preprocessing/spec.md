## ADDED Requirements

### Requirement: LiDAR baseline profile 显式启用训练集统计归一化
LiDAR-only teacher baseline 配置 MUST 显式启用训练集 streaming stats 归一化，并 MUST 将同一 normalizer 复用于测试 split。该要求只约束 LiDAR baseline profile；未显式启用归一化的其它实验仍可保持既有默认行为。

#### Scenario: 默认 LiDAR teacher baseline 启用 streaming stats
- **WHEN** 用户运行默认 LiDAR teacher/no-KD baseline 配置
- **THEN** 配置 MUST 显式设置 LiDAR normalization 为 enabled
- **AND** normalization mode MUST 为 `streaming_stats` 或等价训练集统计模式
- **AND** 系统 MUST 只在 train split 上 fit normalizer
- **AND** test split MUST 复用 train split fit 得到的 normalizer

#### Scenario: LiDAR stats 工件可复用
- **WHEN** LiDAR baseline 训练完成 normalizer fit
- **THEN** 系统 MUST 将 LiDAR stats 或 normalizer 保存为运行工件
- **AND** 评估同一 checkpoint 时 MUST 优先复用该工件
- **AND** 系统 MUST 不在 test split 上重新 fit LiDAR normalizer

### Requirement: LiDAR baseline 输入质量诊断
LiDAR baseline 训练和评估 MUST 记录 BEV 输入质量摘要，使空 BEV、近常量通道、cache 参数混用或 ROI/FoV 异常可以被定位。

#### Scenario: 记录 BEV 非空率和通道统计
- **WHEN** 用户运行启用 LiDAR 的训练或评估
- **THEN** 系统 MUST 记录 LiDAR BEV 非空帧比例
- **AND** 系统 MUST 记录每个 BEV 通道的均值、标准差和零值比例摘要
- **AND** 摘要 MUST 区分 train/test split 或在 metadata 中标明来源 split

#### Scenario: 标记疑似退化输入
- **WHEN** LiDAR BEV 质量摘要显示大量全零帧或通道标准差低于实现定义的退化阈值
- **THEN** 系统 MUST 在运行报告中标记 LiDAR input degradation risk
- **AND** 报告 MUST 包含对应的 ROI、FoV、normalization 和 cache 参数

### Requirement: LiDAR cache 与 ROI/FoV 参数可追踪
LiDAR baseline profile MUST 使用参数隔离的 BEV cache，并 MUST 在训练输出中记录实际使用的 ROI、FoV、ground/background filter 和 cache 目录。

#### Scenario: cache key 区分 LiDAR 构造参数
- **WHEN** LiDAR BEV size、ROI、FoV、ground filter 或 background filter 参数变化
- **THEN** 系统 MUST 使用不同 cache key、目录或 metadata 约束避免错误复用旧 BEV cache

#### Scenario: final_config 记录 LiDAR profile
- **WHEN** 一次启用 LiDAR 的训练启动
- **THEN** final_config 或运行 metadata MUST 记录 LiDAR BEV size、ROI、FoV、normalization、cache policy 和 cache 目录

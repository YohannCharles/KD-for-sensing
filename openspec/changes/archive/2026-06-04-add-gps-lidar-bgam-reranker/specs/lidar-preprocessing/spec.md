## ADDED Requirements

### Requirement: BGAM LiDAR BEV grid metadata
LiDAR BEV cache 和 BGAM dataset MUST 记录足以复现 BGAM angle grid 的 BEV metadata。metadata MUST 包含 ROI、BEV size、grid size、cell center convention、FoV、ground/background filter、cache version 和参数 hash。

#### Scenario: BGAM 使用 BEV cache metadata
- **WHEN** BGAM dataset 读取 `lidar_bev_cache_path`
- **THEN** 系统 MUST 能获得该 BEV 的 ROI、height、width 和 cell center convention
- **AND** `GPSGuidedBGAM` MUST 使用同一 grid metadata 预计算 `theta_cell`
- **AND** run metadata MUST 记录实际使用的 BEV profile 和 cache hash

#### Scenario: cache 参数不匹配
- **WHEN** manifest 或配置中的 BGAM BEV grid 参数与 cache metadata 不一致
- **THEN** 系统 MUST 拒绝复用该 cache 或按配置在线重建
- **AND** 错误信息 MUST 包含不匹配的参数名

### Requirement: BGAM raw pillar pseudo-image fallback
系统 MUST 支持为 GPS+LiDAR BGAM workflow 构造 raw point cloud 到 pillar pseudo-image 的轻量 fallback。该 fallback MUST 使用现有 LiDAR reader/filter 逻辑，并 MUST 不引入重型点云依赖。

#### Scenario: raw 点云构造 pillar pseudo-image
- **WHEN** 配置 `lidar.profile=pillar6` 且样本提供 raw point cloud
- **THEN** 系统 MUST 按 ROI/grid 过滤并 pillarize 点云
- **AND** 每个 cell MUST 至少计算 point count normalized、mean z、max z、mean intensity、mean x offset 和 mean y offset
- **AND** 输出 MUST 为固定 shape `[6,H,W]` 或配置指定等价通道数

#### Scenario: 空点云 fallback
- **WHEN** raw 点云经过过滤后没有有效点
- **THEN** 系统 MUST 返回固定 shape 的全零 pillar pseudo-image
- **AND** 系统 MUST 不修改该样本的 label、candidate beams 或 GPS prior

### Requirement: BGAM LiDAR debug quality summary
GPS+LiDAR BGAM workflow MUST 记录 LiDAR BEV 输入质量摘要，以便判断 BGAM 是否被空 BEV、极端稀疏或 cache 混用影响。

#### Scenario: 记录 BGAM LiDAR 质量
- **WHEN** BGAM workflow 完成训练或评估
- **THEN** run metadata 或 diagnostics MUST 记录 raw/model input BEV 非空率、通道均值/标准差、零值比例、ROI、BEV size 和 cache path
- **AND** 质量摘要 MUST 区分 raw BEV 与模型实际输入 feature

#### Scenario: 标记退化风险
- **WHEN** LiDAR 质量摘要显示大量全零帧、近常量通道或 cache 参数异常
- **THEN** 系统 MUST 在 run metadata 或 report 中标记 `lidar_input_degradation_risk`
- **AND** report MUST 给出相关参数和受影响样本数
